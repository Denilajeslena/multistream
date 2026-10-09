import time
import cv2
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

from server.backend.config import settings
from server.backend.db.repository import Repository
from server.backend.db.models import GroundedEvidenceResult, DetectionItem, ParsedQuery
from server.backend.nlp.query_parser import query_parser
from server.backend.models_ai.embeddings import clip_embedder
from server.backend.models_ai.vector_index import vector_index
from server.backend.models_ai.detector import detector
from server.backend.evidence.clip_generator import clip_generator
from server.backend.retrieval.cross_camera import build_cross_camera_timeline

logger = logging.getLogger(__name__)


def _required_detection_terms(parsed: ParsedQuery) -> set[str]:
    terms = set((parsed.attributes or "").lower().split())
    if parsed.object:
        terms.add(parsed.object.lower())
    for concept in (
        "shirt", "jacket", "hoodie", "coat", "backpack", "bag", "handbag",
        "suitcase", "luggage", "umbrella", "cap", "hat", "clothes", "clothing",
        "uniform", "vest", "sweater", "trousers", "pants", "jeans", "shorts",
        "skirt", "dress", "shoes", "boots", "helmet", "sunglasses",
    ):
        if concept in parsed.semantic_query.lower().split():
            terms.add(concept)
    return terms


class GroundedRetriever:
    """
    Hybrid grounded retrieval pipeline:
    1. Parse query
    2. Resolve camera aliases
    3. Apply time filtering
    4. Retrieve semantic candidates (FAISS / CLIP)
    5. Verify with open-vocabulary detector (OWL-ViT)
    6. Rank candidates
    7. Group nearby timestamps
    8. Return grounded evidence
    """
    def retrieve(self, query_text: str) -> GroundedEvidenceResult:
        # Step 1: Parse query
        parsed: ParsedQuery = query_parser.parse(query_text)

        # Step 2: Handle unresolved location alias
        if parsed.unresolved_location:
            return GroundedEvidenceResult(
                matched=False,
                confidence=0.0,
                relevance=0.0,
                explanation=f"I don't know which camera '{parsed.unresolved_location}' refers to. Please identify the camera.",
                unresolved_alias_prompt=parsed.unresolved_location
            )

        # Step 3: Apply time filtering & camera target
        target_camera = parsed.resolved_camera_id
        start_t = parsed.start_time
        end_t = parsed.end_time

        # Step 4: Semantic candidate retrieval via vector embeddings
        query_text_semantic = parsed.semantic_query or parsed.raw_query
        q_vec = clip_embedder.encode_text(query_text_semantic)

        vector_hits = vector_index.search(
            query_vector=q_vec,
            top_k=25,
            camera_id=target_camera,
            start_time=start_t,
            end_time=end_t
        )

        # Build the semantic pool, then always add the newest indexed samples
        # from each relevant camera so an old global top-k cannot hide live data.
        candidate_frames = {}
        if vector_hits:
            for hit in vector_hits:
                frame_rec = Repository.get_frame(hit.frame_id)
                if frame_rec:
                    candidate_frames[hit.frame_id] = {
                        **frame_rec,
                        "semantic_score": hit.score
                    }
        else:
            db_frames = Repository.get_frames(camera_id=target_camera, start_time=start_t, end_time=end_t, limit=25)
            for f in db_frames:
                candidate_frames[f["frame_id"]] = {
                    **f,
                    "semantic_score": 0.5
                }

        now = time.time()
        live_start = max(
            now - settings.LIVE_QUERY_WINDOW_SEC,
            start_t if start_t is not None else now - settings.LIVE_QUERY_WINDOW_SEC,
        )
        live_end = min(now, end_t) if end_t is not None else now
        camera_ids = [target_camera] if target_camera else [
            camera["camera_id"] for camera in Repository.get_cameras()
        ]
        for camera_id in camera_ids:
            if not camera_id:
                continue
            recent_frames = Repository.get_frames(
                camera_id=camera_id,
                start_time=live_start,
                end_time=live_end,
                limit=settings.LIVE_QUERY_FRAMES_PER_CAMERA,
            )
            for frame in recent_frames:
                candidate_frames.setdefault(frame["frame_id"], {
                    **frame,
                    "semantic_score": 0.5,
                })

        if not candidate_frames:
            return GroundedEvidenceResult(
                matched=False,
                confidence=0.0,
                relevance=0.0,
                explanation="No matching visual evidence was found."
            )

        # Step 5: Verification with open-vocabulary visual detection
        detection_queries = [query_text_semantic]
        if parsed.object and not parsed.attributes and not parsed.action:
            detection_queries.append(parsed.object)
        detection_queries = list(dict.fromkeys(q.strip() for q in detection_queries if q.strip()))
        required_terms = _required_detection_terms(parsed)

        verified_candidates = []

        for cand in candidate_frames.values():
            frame_id = cand["frame_id"]
            cam_id = cand["camera_id"]
            ts = cand["timestamp"]
            sem_score = cand["semantic_score"]

            # Check pre-computed detections in SQLite
            stored_dets = Repository.get_detections_for_frame(frame_id)
            matching_dets: List[DetectionItem] = []

            for sd in stored_dets:
                label_terms = set(sd["label"].lower().split())
                if required_terms and required_terms.issubset(label_terms) and sd["confidence"] >= settings.DETECTION_THRESHOLD:
                    matching_dets.append(DetectionItem(
                        label=sd["label"],
                        confidence=sd["confidence"],
                        box=[sd["box_x1"], sd["box_y1"], sd["box_x2"], sd["box_y2"]]
                    ))

            # If no stored detections match, run open-vocabulary detection on the candidate frame
            if not matching_dets:
                frame_full_path = settings.BASE_DIR / cand["frame_path"]
                if frame_full_path.exists():
                    img = cv2.imread(str(frame_full_path))
                    if img is not None:
                        live_dets = detector.detect(img, detection_queries, threshold=settings.DETECTION_THRESHOLD)
                        for ld in live_dets:
                            matching_dets.append(DetectionItem(
                                label=ld.label,
                                confidence=ld.confidence,
                                box=ld.box
                            ))
                            # Persist newly verified detection
                            Repository.insert_detection(
                                frame_id=frame_id,
                                camera_id=cam_id,
                                timestamp=ts,
                                label=ld.label,
                                confidence=ld.confidence,
                                box=ld.box
                            )

            # Calculate composite grounded score - requires visual detection evidence
            if matching_dets:
                best_det_conf = max(d.confidence for d in matching_dets)
                is_live = ts >= now - settings.LIVE_QUERY_WINDOW_SEC
                recency_bonus = 0.05 if is_live else 0.0
                composite_score = min(1.0, (best_det_conf * 0.6) + (sem_score * 0.4) + recency_bonus)
                verified_candidates.append({
                    **cand,
                    "composite_score": composite_score,
                    "confidence": best_det_conf,
                    "is_live": is_live,
                    "detections": matching_dets
                })

        if not verified_candidates:
            return GroundedEvidenceResult(
                matched=False,
                confidence=0.0,
                relevance=0.0,
                explanation="No matching visual evidence was found."
            )

        # Step 6: Rank candidates
        verified_candidates.sort(
            key=lambda x: (x["composite_score"], x["timestamp"]),
            reverse=True
        )

        # Step 7: Group nearby timestamps (avoid duplicate contiguous detections)
        best_cand = verified_candidates[0]
        timeline = build_cross_camera_timeline(verified_candidates, query_text_semantic)

        # Step 8: Return grounded evidence with video clip
        cam_info = Repository.get_camera(best_cand["camera_id"])
        cam_name = cam_info["camera_name"] if cam_info else best_cand["camera_id"]

        # Generate on-demand evidence clip (±5s)
        clip_rel_path = clip_generator.generate_clip(
            camera_id=best_cand["camera_id"],
            center_timestamp=best_cand["timestamp"],
            frame_id=best_cand["frame_id"],
            half_window_sec=settings.EVIDENCE_HALF_WINDOW_SEC
        )

        time_str = datetime.fromtimestamp(best_cand["timestamp"], tz=timezone.utc).strftime("%H:%M:%S UTC")
        obj_desc = f"{parsed.attributes or ''} {parsed.object or query_text_semantic}".strip()
        explanation = f"Detected {obj_desc} at {cam_name} ({best_cand['camera_id']}) around {time_str} with {best_cand['confidence']:.2f} confidence."
        if timeline and len(timeline) > 1:
            transitions = " → ".join(f"{step['camera_name']} {datetime.fromtimestamp(step['timestamp'], tz=timezone.utc).strftime('%H:%M:%S')}" for step in timeline[:5])
            explanation = f"Detected {obj_desc}. Cross-camera continuity: {transitions}."

        return GroundedEvidenceResult(
            matched=True,
            camera_id=best_cand["camera_id"],
            camera_name=cam_name,
            timestamp=best_cand["timestamp"],
            timestamp_iso=best_cand["timestamp_iso"],
            frame_id=best_cand["frame_id"],
            frame_url=f"/{best_cand['frame_path']}",
            clip_url=f"/{clip_rel_path}" if clip_rel_path else None,
            confidence=round(best_cand["confidence"], 2),
            relevance=round(best_cand["composite_score"], 2),
            detections=best_cand["detections"],
            timeline=timeline,
            is_live_evidence=best_cand["is_live"],
            source_age_seconds=round(max(0.0, now - best_cand["timestamp"]), 1),
            explanation=explanation
        )

grounded_retriever = GroundedRetriever()

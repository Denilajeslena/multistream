import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (label or "").lower()).strip()


def _matches_query(label: str, query_tokens: Sequence[str]) -> bool:
    if not query_tokens:
        return True
    norm = _normalize_label(label)
    return any(token in norm for token in query_tokens if token)


def build_cross_camera_timeline(
    candidate_hits: Sequence[Dict[str, Any]],
    query_text: str,
    max_gap_sec: float = 600.0,
) -> List[Dict[str, Any]]:
    """
    Build a simple continuity timeline for a person/vehicle seen across multiple cameras.
    This intentionally keeps the logic lightweight and deterministic for demo use.
    """
    if len(candidate_hits) < 2:
        return []

    tokens = []
    cleaned = re.sub(r"[^a-z0-9\s]", " ", (query_text or "").lower())
    for token in cleaned.split():
        if token and token not in {"did", "the", "there", "is", "was", "show", "me", "a", "an", "any", "and", "or", "in", "at", "on", "of", "to", "for", "from"}:
            tokens.append(token)

    def _det_conf(d: Any) -> float:
        if isinstance(d, dict):
            return float(d.get("confidence", 0.0))
        return float(getattr(d, "confidence", 0.0))

    def _det_label(d: Any) -> str:
        if isinstance(d, dict):
            return str(d.get("label", "unknown"))
        return str(getattr(d, "label", "unknown"))

    timeline: List[Dict[str, Any]] = []
    for hit in sorted(candidate_hits, key=lambda x: float(x.get("timestamp", 0.0))):
        detections = hit.get("detections") or []
        if not detections:
            continue

        best_detection = max(detections, key=_det_conf)
        primary_label = _det_label(best_detection)
        if not tokens or _matches_query(primary_label, tokens):
            timestamp = float(hit.get("timestamp", 0.0))
            camera_id = hit.get("camera_id")
            camera_name = hit.get("camera_name") or camera_id
            timeline.append({
                "camera_id": camera_id,
                "camera_name": camera_name,
                "timestamp": timestamp,
                "timestamp_iso": hit.get("timestamp_iso") or datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
                "label": primary_label,
                "confidence": max((_det_conf(d) for d in detections), default=0.0),
            })

    if len(timeline) < 2:
        return []

    compact: List[Dict[str, Any]] = []
    prev = timeline[0]
    compact.append(prev)
    for step in timeline[1:]:
        if step["camera_id"] == prev["camera_id"] and (step["timestamp"] - prev["timestamp"]) <= max_gap_sec:
            continue
        if (step["timestamp"] - prev["timestamp"]) > max_gap_sec:
            compact.append(step)
            prev = step
            continue
        compact.append(step)
        prev = step

    return compact

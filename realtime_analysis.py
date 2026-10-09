import gc
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import cv2
import httpx
import numpy as np

from server.backend.config import settings
from server.backend.db.repository import Repository
from server.backend.ingestion.buffer_manager import buffer_manager
from server.backend.ingestion.camera_registry import camera_registry
from server.backend.models_ai.detector import detector
from server.backend.notifications.n8n_webhook import _format_telegram_message

logger = logging.getLogger(__name__)
_NOTIFICATION_STOP = object()


class RealtimeAnalysisService:
    """Single-camera, newest-frame analysis isolated from CCTV ingestion/indexing."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._shutdown = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._notification_worker: Optional[threading.Thread] = None
        self._notification_queue: queue.Queue = queue.Queue(maxsize=100)
        self._active = False
        self._camera_id: Optional[str] = None
        self._model = "existing"
        self._generation = 0
        self._last_sequence = 0
        self._last_processed_at = 0.0
        self._error: Optional[str] = None
        self._detections: List[Dict[str, Any]] = []
        self._current_events: List[Dict[str, Any]] = []
        self._frame_timestamp: Optional[float] = None
        self._frame_width = 0
        self._frame_height = 0
        self._notification_status = "disabled"
        self._last_notification: Optional[Dict[str, Any]] = None
        self._cooldowns: Dict[Tuple[str, str, str], float] = {}
        self._next_track_id = 1
        self._tracks: Dict[int, Dict[str, Any]] = {}
        self._departed_tracks: List[Dict[str, Any]] = []
        self._yolo: Any = None
        self._yolo_model_name: Optional[str] = None
        self._yolo_load_error: Optional[str] = None

    def start(self, camera_id: str, model: str) -> Dict[str, Any]:
        if not settings.REALTIME_ANALYSIS:
            raise RuntimeError("Real-time analysis is disabled.")
        camera_id = camera_id.strip().upper()
        if camera_registry.get_camera(camera_id) is None:
            raise ValueError(f"Camera {camera_id} is not configured.")
        if model not in {"existing", "yolo11n"}:
            raise ValueError("Model must be 'existing' or 'yolo11n'.")
        with self._lock:
            self._active = True
            self._camera_id = camera_id
            self._model = model
            self._generation += 1
            self._last_sequence = 0
            self._last_processed_at = 0.0
            self._error = None
            self._yolo_load_error = None
            self._detections = []
            self._current_events = []
            self._frame_timestamp = None
            self._frame_width = 0
            self._frame_height = 0
            self._reset_tracks()
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._analysis_loop,
                    name="realtime-analysis",
                    daemon=True,
                )
                self._worker.start()
            if settings.N8N_NOTIFICATIONS:
                self._ensure_notification_worker()
        return self.status()

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            self._active = False
            self._generation += 1
            self._detections = []
            self._current_events = []
            self._error = None
        return self.status()

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": settings.REALTIME_ANALYSIS,
                "active": self._active,
                "camera_id": self._camera_id,
                "model": self._model,
                "inference_fps": settings.REALTIME_INFERENCE_FPS,
                "frame_timestamp": self._frame_timestamp,
                "frame_width": self._frame_width,
                "frame_height": self._frame_height,
                "detections": list(self._detections),
                "current_events": list(self._current_events),
                "error": self._error,
                "notifications_enabled": settings.N8N_NOTIFICATIONS,
                "notification_status": self._notification_status,
                "last_notification": self._last_notification,
                "cooldown_seconds": settings.REALTIME_EVENT_COOLDOWN_SEC,
            }

    def events(self, camera_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        camera = camera_id.strip().upper() if camera_id else None
        return Repository.get_realtime_events(camera, min(max(limit, 1), 250))

    def shutdown(self) -> None:
        self._shutdown.set()
        try:
            self._notification_queue.put_nowait(_NOTIFICATION_STOP)
        except queue.Full:
            logger.warning("Real-time notification queue full during shutdown.")
        worker = self._worker
        if worker and worker.is_alive():
            worker.join(timeout=2)
        notification_worker = self._notification_worker
        if notification_worker and notification_worker.is_alive():
            notification_worker.join(timeout=2)

    def _reset_tracks(self) -> None:
        self._next_track_id = 1
        self._tracks.clear()
        self._departed_tracks = []

    def _analysis_loop(self) -> None:
        processed_generation = -1
        while not self._shutdown.is_set():
            with self._lock:
                active = self._active
                camera_id = self._camera_id
                model = self._model
                generation = self._generation
            if not active or camera_id is None:
                if self._yolo is not None:
                    self._release_yolo()
                self._shutdown.wait(0.1)
                continue
            if generation != processed_generation:
                processed_generation = generation
                self._last_sequence = 0
                self._reset_tracks()
                self._reset_yolo_tracking()
                if model != "yolo11n" and self._yolo is not None:
                    self._release_yolo()
            interval = 1.0 / max(settings.REALTIME_INFERENCE_FPS, 0.1)
            elapsed = time.monotonic() - self._last_processed_at
            if elapsed < interval:
                self._shutdown.wait(min(interval - elapsed, 0.05))
                continue
            latest = buffer_manager.get_latest_frame_details(camera_id)
            if latest is None or latest[0] == self._last_sequence:
                self._shutdown.wait(0.1)
                continue
            sequence, timestamp, encoded = latest
            self._last_sequence = sequence
            try:
                frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("Latest camera frame could not be decoded.")
                source_height, source_width = frame.shape[:2]
                scale = min(
                    1.0,
                    settings.REALTIME_FRAME_WIDTH / max(source_width, 1),
                    settings.REALTIME_FRAME_HEIGHT / max(source_height, 1),
                )
                if scale < 1.0:
                    frame = cv2.resize(
                        frame,
                        (max(1, int(source_width * scale)), max(1, int(source_height * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                if model == "yolo11n":
                    detections = self._detect_yolo(frame, source_width, source_height)
                else:
                    detections = self._detect_existing(frame, source_width, source_height)
                with self._lock:
                    if not self._active or self._generation != generation:
                        continue
                current_events = self._emit_events(
                    camera_id, timestamp, detections, source_width, source_height
                )
                with self._lock:
                    if self._active and self._generation == generation:
                        self._detections = detections
                        self._current_events = current_events
                        self._frame_timestamp = timestamp
                        self._frame_width = source_width
                        self._frame_height = source_height
                        self._error = None
                self._last_processed_at = time.monotonic()
            except Exception as exc:
                with self._lock:
                    previous_error = self._error
                    if self._generation == generation:
                        self._error = str(exc)
                if previous_error != str(exc):
                    logger.exception("Real-time analysis failed for %s.", camera_id)
                else:
                    logger.debug("Real-time analysis remains unavailable for %s: %s", camera_id, exc)
                self._last_processed_at = time.monotonic()
                self._shutdown.wait(1.0)
        self._release_yolo()

    def _detect_existing(
        self,
        frame: np.ndarray,
        source_width: int,
        source_height: int,
    ) -> List[Dict[str, Any]]:
        raw = detector.detect(
            frame,
            ["person", "car", "truck", "bicycle", "bus", "motorcycle", "bag", "backpack", "phone", "mobile phone"],
            threshold=settings.DETECTION_THRESHOLD,
        )
        sx = source_width / frame.shape[1]
        sy = source_height / frame.shape[0]
        values = [
            {
                "label": detection.label,
                "confidence": detection.confidence,
                "box": [
                    round(detection.box[0] * sx, 1),
                    round(detection.box[1] * sy, 1),
                    round(detection.box[2] * sx, 1),
                    round(detection.box[3] * sy, 1),
                ],
            }
            for detection in raw
        ]
        self._assign_centroid_tracks(values, source_width, source_height)
        return values

    def _detect_yolo(
        self,
        frame: np.ndarray,
        source_width: int,
        source_height: int,
    ) -> List[Dict[str, Any]]:
        if self._yolo is None or self._yolo_model_name != settings.REALTIME_YOLO_MODEL:
            if self._yolo_load_error:
                raise RuntimeError(self._yolo_load_error)
            try:
                from ultralytics import YOLO

                self._yolo = YOLO(settings.REALTIME_YOLO_MODEL)
                self._yolo_model_name = settings.REALTIME_YOLO_MODEL
            except Exception as exc:
                self._yolo_load_error = (
                    f"Could not load YOLO model '{settings.REALTIME_YOLO_MODEL}': {exc}"
                )
                raise RuntimeError(self._yolo_load_error) from exc
        with detector.inference_lock:
            result = self._yolo.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                imgsz=settings.REALTIME_YOLO_IMAGE_SIZE,
                device=settings.REALTIME_DEVICE,
                max_det=50,
            )[0]
        sx = source_width / frame.shape[1]
        sy = source_height / frame.shape[0]
        names = result.names
        detections: List[Dict[str, Any]] = []
        boxes = result.boxes
        if boxes is None:
            self._record_yolo_tracks([], source_width, source_height)
            return detections
        coordinates = boxes.xyxy.cpu().tolist()
        classes = boxes.cls.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(coordinates)
        for box, class_id, confidence, track_id in zip(coordinates, classes, confidences, track_ids):
            detections.append({
                "label": str(names[int(class_id)]),
                "confidence": float(confidence),
                "box": [round(box[0] * sx, 1), round(box[1] * sy, 1), round(box[2] * sx, 1), round(box[3] * sy, 1)],
                "tracking_id": int(track_id) if track_id is not None else None,
            })
        self._record_yolo_tracks(detections, source_width, source_height)
        return detections

    def _assign_centroid_tracks(
        self,
        detections: List[Dict[str, Any]],
        frame_width: int,
        frame_height: int,
    ) -> None:
        now = time.monotonic()
        self._collect_departed_tracks(now, frame_width, frame_height)
        used: set[int] = set()
        for detection in detections:
            box = detection["box"]
            center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            candidates = [
                (track_id, track)
                for track_id, track in self._tracks.items()
                if track_id not in used
                and track["label"] == detection["label"]
                and now - track["seen"] <= 2.0
            ]
            match = min(
                candidates,
                key=lambda item: (item[1]["center"][0] - center[0]) ** 2 + (item[1]["center"][1] - center[1]) ** 2,
                default=None,
            )
            max_distance = 75.0
            if match and (match[1]["center"][0] - center[0]) ** 2 + (match[1]["center"][1] - center[1]) ** 2 <= max_distance ** 2:
                track_id = match[0]
            else:
                track_id = self._next_track_id
                self._next_track_id += 1
            used.add(track_id)
            previous = self._tracks.get(track_id)
            stationary_anchor = previous["stationary_anchor"] if previous else center
            moved = previous and (
                (stationary_anchor[0] - center[0]) ** 2
                + (stationary_anchor[1] - center[1]) ** 2
            ) > 12 ** 2
            self._tracks[track_id] = {
                "label": detection["label"],
                "center": center,
                "seen": now,
                "box": box,
                "confidence": detection["confidence"],
                "new": previous is None,
                "stationary_since": now if previous is None or moved else previous["stationary_since"],
                "stationary_anchor": center if previous is None or moved else stationary_anchor,
            }
            detection["tracking_id"] = track_id
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if now - track["seen"] <= 2.0
        }

    def _record_yolo_tracks(
        self,
        detections: List[Dict[str, Any]],
        frame_width: int,
        frame_height: int,
    ) -> None:
        now = time.monotonic()
        self._collect_departed_tracks(now, frame_width, frame_height)
        for detection in detections:
            track_id = detection.get("tracking_id")
            if track_id is None:
                continue
            box = detection["box"]
            center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            previous = self._tracks.get(track_id)
            anchor = previous["stationary_anchor"] if previous else center
            moved = previous and (
                (anchor[0] - center[0]) ** 2 + (anchor[1] - center[1]) ** 2 > 12 ** 2
            )
            self._tracks[track_id] = {
                "label": detection["label"],
                "center": center,
                "seen": now,
                "box": box,
                "confidence": detection["confidence"],
                "new": previous is None,
                "stationary_since": now if previous is None or moved else previous["stationary_since"],
                "stationary_anchor": center if previous is None or moved else anchor,
            }
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if now - track["seen"] <= 2.0
        }

    def _collect_departed_tracks(
        self,
        now: float,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> None:
        self._departed_tracks = []
        expired = [
            (track_id, track)
            for track_id, track in self._tracks.items()
            if now - track["seen"] > 2.0
        ]
        for track_id, track in expired:
            if (
                track["label"].lower() == "person"
                and frame_width
                and frame_height
                and self._is_at_frame_edge(track["box"], frame_width, frame_height)
            ):
                self._departed_tracks.append({
                    "label": track["label"],
                    "confidence": track["confidence"],
                    "box": track["box"],
                    "tracking_id": track_id,
                })
            self._tracks.pop(track_id, None)

    @staticmethod
    def _is_at_frame_edge(box: List[float], width: int, height: int) -> bool:
        return (
            box[0] <= width * 0.03
            or box[1] <= height * 0.03
            or box[2] >= width * 0.97
            or box[3] >= height * 0.97
        )

    def _emit_events(
        self,
        camera_id: str,
        timestamp: float,
        detections: List[Dict[str, Any]],
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        people = [item for item in detections if item["label"].lower() == "person"]
        candidates: List[Tuple[str, List[Dict[str, Any]]]] = []
        candidates.extend(("PERSON_EXITED", [track]) for track in self._departed_tracks)
        for detection in detections:
            label = detection["label"].lower()
            if label == "person":
                candidates.append(("PERSON_DETECTED", [detection]))
                track = self._tracks.get(detection.get("tracking_id"))
                if (
                    track
                    and track.get("new")
                    and frame_width
                    and frame_height
                    and self._is_at_frame_edge(track["box"], frame_width, frame_height)
                ):
                    candidates.append(("PERSON_ENTERED", [detection]))
            elif label in {"car", "truck", "bus", "motorcycle", "bicycle", "vehicle"}:
                candidates.append(("VEHICLE_DETECTED", [detection]))
            if label != "person" and detection["tracking_id"] is not None:
                track = self._tracks.get(detection["tracking_id"])
                if track and timestamp > 0 and time.monotonic() - track["stationary_since"] >= 15:
                    candidates.append(("STATIONARY_OBJECT", [detection]))
        carryable = {"bag", "backpack", "handbag", "suitcase"}
        for person in people:
            px1, py1, px2, py2 = person["box"]
            for item in detections:
                if item["label"].lower() not in carryable:
                    continue
                ix1, iy1, ix2, iy2 = item["box"]
                overlap_w = max(0.0, min(px2, ix2) - max(px1, ix1))
                overlap_h = max(0.0, min(py2, iy2) - max(py1, iy1))
                item_area = max(1.0, (ix2 - ix1) * (iy2 - iy1))
                if overlap_w * overlap_h / item_area >= 0.5:
                    candidates.append(("POSSIBLE_CARRYING_EVENT", [person, item]))
        if len(people) >= 3:
            centers = [
                ((item["box"][0] + item["box"][2]) / 2, (item["box"][1] + item["box"][3]) / 2)
                for item in people
            ]
            heights = [max(1.0, item["box"][3] - item["box"][1]) for item in people]
            group_span_x = max(center[0] for center in centers) - min(center[0] for center in centers)
            group_span_y = max(center[1] for center in centers) - min(center[1] for center in centers)
            typical_height = sum(heights) / len(heights)
        else:
            group_span_x = group_span_y = typical_height = 0.0
        if len(people) >= 3 and group_span_x <= typical_height * 5 and group_span_y <= typical_height * 3:
            candidates.append(("CROWD_DETECTED", people))

        emitted: List[Dict[str, Any]] = []
        now = time.monotonic()
        cooldown = max(settings.REALTIME_EVENT_COOLDOWN_SEC, 0)
        for event_type, items in candidates:
            tracking_ids = sorted({
                item["tracking_id"] for item in items if item.get("tracking_id") is not None
            })
            cooldown_id = ",".join(map(str, tracking_ids)) or "none"
            key = (camera_id, event_type, cooldown_id)
            with self._lock:
                if now - self._cooldowns.get(key, float("-inf")) < cooldown:
                    continue
                self._cooldowns[key] = now
                if len(self._cooldowns) > 5000:
                    self._cooldowns = {
                        cooldown_key: seen_at
                        for cooldown_key, seen_at in self._cooldowns.items()
                        if now - seen_at < cooldown
                    }
            event = {
                "camera_id": camera_id,
                "timestamp": timestamp,
                "event_type": event_type,
                "objects": sorted({item["label"] for item in items}),
                "confidence": max(item["confidence"] for item in items),
                "tracking_ids": tracking_ids,
                "bounding_box": items[0]["box"] if len(items) == 1 else None,
            }
            try:
                Repository.insert_realtime_event(event)
            except Exception:
                logger.exception("Could not persist real-time event for %s.", camera_id)
                continue
            emitted.append(event)
            if settings.N8N_NOTIFICATIONS:
                self._enqueue_notification(event)
        return emitted

    def _ensure_notification_worker(self) -> None:
        if self._notification_worker is None or not self._notification_worker.is_alive():
            self._notification_worker = threading.Thread(
                target=self._notification_loop,
                name="realtime-n8n-notifier",
                daemon=True,
            )
            self._notification_worker.start()
            self._notification_status = "disconnected"

    def _reset_yolo_tracking(self) -> None:
        predictor = getattr(self._yolo, "predictor", None)
        trackers = getattr(predictor, "trackers", [])
        for tracker in trackers:
            reset = getattr(tracker, "reset", None)
            if reset:
                reset()

    def _release_yolo(self) -> None:
        if self._yolo is None:
            return
        self._yolo = None
        self._yolo_model_name = None
        gc.collect()
        if settings.REALTIME_DEVICE.lower().startswith("cuda"):
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                logger.debug("Could not release cached CUDA allocations for YOLO.", exc_info=True)

    def _enqueue_notification(self, event: Dict[str, Any]) -> None:
        timestamp_iso = datetime.fromtimestamp(event["timestamp"], timezone.utc).isoformat()
        payload = {
            **event,
            "source": "realtime_analysis",
            "timestamp": timestamp_iso,
            "text": _format_telegram_message({
                "camera": event["camera_id"],
                "timestamp": timestamp_iso,
                "answer": f"{event['event_type']} · {', '.join(event['objects'])}",
                "confidence": event["confidence"],
                "detections": [{"label": label, "confidence": event["confidence"]} for label in event["objects"]],
            }),
        }
        try:
            self._notification_queue.put_nowait(payload)
        except queue.Full:
            logger.warning("Real-time n8n notification queue is full; dropping event notification.")

    def _notification_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                payload = self._notification_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                if payload is _NOTIFICATION_STOP:
                    return
                timeout = httpx.Timeout(3.0, connect=1.0)
                response = httpx.post(
                    settings.N8N_WEBHOOK_URL,
                    json=payload,
                    timeout=timeout,
                )
                if response.is_error:
                    logger.warning("Real-time n8n webhook returned HTTP %s.", response.status_code)
                    status = "disconnected"
                else:
                    status = "connected"
                with self._lock:
                    self._notification_status = status
                    self._last_notification = {
                        "timestamp": payload["timestamp"],
                        "event_type": payload["event_type"],
                        "status": status,
                    }
            except httpx.HTTPError as exc:
                logger.warning("Real-time n8n notification failed: %s", exc)
                with self._lock:
                    self._notification_status = "disconnected"
                    self._last_notification = {
                        "timestamp": payload["timestamp"],
                        "event_type": payload["event_type"],
                        "status": "failed",
                    }
            finally:
                self._notification_queue.task_done()


realtime_analysis_service = RealtimeAnalysisService()

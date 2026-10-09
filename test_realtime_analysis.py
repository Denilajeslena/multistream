import time
from types import SimpleNamespace

import cv2
import numpy as np

from server.backend.config import settings
from server.backend.ingestion import realtime_analysis as analysis_module
from server.backend.ingestion.realtime_analysis import RealtimeAnalysisService
from server.backend.models_ai.interfaces import DetectionResult


def test_existing_model_boxes_are_scaled_and_tracked(monkeypatch):
    service = RealtimeAnalysisService()
    monkeypatch.setattr(
        analysis_module.detector,
        "detect",
        lambda *_args, **_kwargs: [
            DetectionResult(label="person", confidence=0.91, box=[10, 12, 40, 70])
        ],
    )
    first = service._detect_existing(np.zeros((180, 320, 3), dtype=np.uint8), 640, 360)
    second = service._detect_existing(np.zeros((180, 320, 3), dtype=np.uint8), 640, 360)

    assert first[0]["box"] == [20, 24, 80, 140]
    assert first[0]["tracking_id"] == second[0]["tracking_id"]


def test_yolo_adapter_uses_persistent_bytetrack(monkeypatch):
    calls = []

    class Boxes:
        xyxy = SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: [[1, 2, 10, 20]]))
        cls = SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: [0]))
        conf = SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: [0.9]))
        id = SimpleNamespace(int=lambda: SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: [12])))

    class FakeYOLO:
        def __init__(self, model_name):
            assert model_name == settings.REALTIME_YOLO_MODEL

        def track(self, frame, **kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(names={0: "person"}, boxes=Boxes())]

    monkeypatch.setitem(
        __import__("sys").modules,
        "ultralytics",
        SimpleNamespace(YOLO=FakeYOLO),
    )
    service = RealtimeAnalysisService()
    result = service._detect_yolo(np.zeros((180, 320, 3), dtype=np.uint8), 640, 360)

    assert result == [{
        "label": "person",
        "confidence": 0.9,
        "box": [2, 4, 20, 40],
        "tracking_id": 12,
    }]
    assert calls[0]["persist"] is True
    assert calls[0]["tracker"] == "bytetrack.yaml"
    assert calls[0]["device"] == settings.REALTIME_DEVICE
    assert calls[0]["imgsz"] == settings.REALTIME_YOLO_IMAGE_SIZE
    assert calls[0]["max_det"] == 50


def test_event_cooldown_prevents_duplicate_records(monkeypatch):
    service = RealtimeAnalysisService()
    persisted = []
    notifications = []
    monkeypatch.setattr(settings, "REALTIME_EVENT_COOLDOWN_SEC", 20)
    monkeypatch.setattr(settings, "N8N_NOTIFICATIONS", True)
    monkeypatch.setattr(analysis_module.Repository, "insert_realtime_event", lambda event: persisted.append(event))
    monkeypatch.setattr(service, "_enqueue_notification", lambda event: notifications.append(event))
    detection = {
        "label": "person",
        "confidence": 0.92,
        "box": [1, 2, 30, 60],
        "tracking_id": 12,
    }

    first = service._emit_events("CAM-01", time.time(), [detection])
    second = service._emit_events("CAM-01", time.time() + 1, [detection])

    assert len(first) == 1
    assert first[0]["event_type"] == "PERSON_DETECTED"
    assert first[0]["tracking_ids"] == [12]
    assert second == []
    assert len(persisted) == len(notifications) == 1


def test_spatial_events_require_detection_and_tracking_evidence(monkeypatch):
    service = RealtimeAnalysisService()
    persisted = []
    monkeypatch.setattr(settings, "REALTIME_EVENT_COOLDOWN_SEC", 20)
    monkeypatch.setattr(settings, "N8N_NOTIFICATIONS", False)
    monkeypatch.setattr(analysis_module.Repository, "insert_realtime_event", lambda event: persisted.append(event))

    edge_person = {
        "label": "person",
        "confidence": 0.9,
        "box": [0, 10, 25, 80],
        "tracking_id": 7,
    }
    service._record_yolo_tracks([edge_person], 100, 100)
    events = service._emit_events("CAM-01", time.time(), [edge_person], 100, 100)
    assert {event["event_type"] for event in events} == {"PERSON_DETECTED", "PERSON_ENTERED"}

    bag = {"label": "backpack", "confidence": 0.85, "box": [5, 20, 20, 50], "tracking_id": 8}
    person_with_bag = {
        "label": "person",
        "confidence": 0.92,
        "box": [0, 10, 25, 80],
        "tracking_id": 7,
    }
    service._record_yolo_tracks([person_with_bag, bag], 100, 100)
    carrying_events = service._emit_events(
        "CAM-01", time.time() + 1, [person_with_bag, bag], 100, 100
    )
    assert any(event["event_type"] == "POSSIBLE_CARRYING_EVENT" for event in carrying_events)

    service._tracks[9] = {
        "label": "person",
        "center": (2, 50),
        "seen": time.monotonic() - 3,
        "box": [0, 30, 15, 80],
        "confidence": 0.8,
        "stationary_since": time.monotonic(),
        "stationary_anchor": (2, 50),
    }
    service._collect_departed_tracks(time.monotonic(), 100, 100)
    departed = service._emit_events("CAM-01", time.time() + 2, [], 100, 100)
    assert any(event["event_type"] == "PERSON_EXITED" for event in departed)
    assert len(persisted) == len(events) + len(carrying_events) + len(departed)


def test_notification_payload_is_structured_and_formatted(monkeypatch):
    service = RealtimeAnalysisService()
    service._notification_queue = __import__("queue").Queue(maxsize=1)
    event = {
        "camera_id": "CAM-01",
        "timestamp": 1700000000.0,
        "event_type": "PERSON_DETECTED",
        "objects": ["person"],
        "confidence": 0.91,
        "tracking_ids": [12],
        "bounding_box": [1, 2, 30, 60],
    }

    service._enqueue_notification(event)
    payload = service._notification_queue.get_nowait()

    assert payload["source"] == "realtime_analysis"
    assert payload["camera_id"] == "CAM-01"
    assert payload["timestamp"].startswith("2023-11-14T")
    assert payload["tracking_ids"] == [12]
    assert "PERSON_DETECTED" in payload["text"]
    assert "Detected: person 91%" in payload["text"]


def test_live_worker_consumes_latest_buffered_frame(monkeypatch):
    service = RealtimeAnalysisService()
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    encoded_ok, encoded = cv2.imencode(".jpg", frame)
    assert encoded_ok
    processed = []
    monkeypatch.setattr(settings, "REALTIME_ANALYSIS", True)
    monkeypatch.setattr(settings, "REALTIME_INFERENCE_FPS", 10)
    monkeypatch.setattr(
        analysis_module.buffer_manager,
        "get_latest_frame_details",
        lambda _camera: (7, 1700000000.0, encoded.tobytes()),
    )
    monkeypatch.setattr(
        service,
        "_detect_existing",
        lambda *_args: processed.append(True) or [{
            "label": "person",
            "confidence": 0.9,
            "box": [1, 2, 30, 60],
            "tracking_id": 1,
        }],
    )
    monkeypatch.setattr(analysis_module.Repository, "insert_realtime_event", lambda _event: None)

    try:
        service.start("CAM-01", "existing")
        deadline = time.monotonic() + 2
        while not processed and time.monotonic() < deadline:
            time.sleep(0.02)
        status = service.status()
        assert processed
        assert status["active"] is True
        assert status["camera_id"] == "CAM-01"
        assert status["frame_timestamp"] == 1700000000.0
        assert status["detections"][0]["tracking_id"] == 1
    finally:
        service.shutdown()

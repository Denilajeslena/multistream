import os
import time
import cv2
import pytest
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure sys.path includes project root
import sys
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from server.backend.config import settings
from server.backend.main import app
from server.backend.db.database import init_db, get_connection
from server.backend.db.repository import Repository
from server.backend.ingestion.camera_registry import camera_registry
from server.backend.nlp.alias_resolver import alias_resolver
from server.backend.nlp.query_parser import query_parser
from server.backend.retrieval.retriever import grounded_retriever
from server.backend.retrieval.cross_camera import build_cross_camera_timeline
from server.backend.evidence.clip_generator import clip_generator

client = TestClient(app)

@pytest.fixture(autouse=True)
def disable_external_n8n_webhook(monkeypatch):
    from server.backend.routers import query as query_router

    async def no_external_request(_payload):
        return None

    monkeypatch.setattr(query_router, "send_answer_notification", no_external_request)

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    settings.ensure_directories()
    init_db()
    yield

def make_test_frame_bytes(color=(0, 0, 255), label_text="RED CAR"):
    """Creates a synthetic 640x360 test frame with specified color and label."""
    img = np.zeros((settings.FRAME_HEIGHT, settings.FRAME_WIDTH, 3), dtype=np.uint8)
    img[:] = (60, 60, 60)
    # Draw object
    cv2.rectangle(img, (150, 100), (350, 240), color, -1)
    cv2.putText(img, label_text, (160, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes(), img

# 1. Server Health
def test_1_server_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "uptime_seconds" in data
    assert data["cameras_registered"] >= 3
    assert "storage" in data
    assert data["vision_models"]["detector"] in {"owlvit", "fallback", "not_loaded"}

# 2. CAM-01 Connection
def test_2_cam01_connection():
    frame_bytes, _ = make_test_frame_bytes(color=(0, 0, 220), label_text="RED CAR")
    files = {"frame": ("frame1.jpg", frame_bytes, "image/jpeg")}
    data = {"camera_id": "CAM-01", "camera_name": "Main Gate", "timestamp": str(time.time())}
    resp = client.post("/api/cameras/frame", data=data, files=files)
    assert resp.status_code == 200
    assert resp.json()["status"] == "received"
    assert resp.json()["camera_id"] == "CAM-01"

    # Verify online status
    cam_info = client.get("/api/cameras/CAM-01").json()
    assert cam_info["status"] == "ONLINE"
    assert cam_info["camera_id"] == "CAM-01"

# 3. CAM-02 Connection
def test_3_cam02_connection():
    frame_bytes, _ = make_test_frame_bytes(color=(220, 80, 20), label_text="BLUE SHIRT")
    files = {"frame": ("frame2.jpg", frame_bytes, "image/jpeg")}
    data = {"camera_id": "CAM-02", "camera_name": "Parking Area", "timestamp": str(time.time())}
    resp = client.post("/api/cameras/frame", data=data, files=files)
    assert resp.status_code == 200
    assert resp.json()["camera_id"] == "CAM-02"

    cam_info = client.get("/api/cameras/CAM-02").json()
    assert cam_info["status"] == "ONLINE"

# 4. CAM-03 Connection
def test_4_cam03_connection():
    frame_bytes, _ = make_test_frame_bytes(color=(240, 240, 240), label_text="WHITE TRUCK")
    files = {"frame": ("frame3.jpg", frame_bytes, "image/jpeg")}
    data = {"camera_id": "CAM-03", "camera_name": "Exit Gate", "timestamp": str(time.time())}
    resp = client.post("/api/cameras/frame", data=data, files=files)
    assert resp.status_code == 200
    assert resp.json()["camera_id"] == "CAM-03"

    cam_info = client.get("/api/cameras/CAM-03").json()
    assert cam_info["status"] == "ONLINE"

# 5. Camera Disconnect Handling
def test_5_camera_disconnect_handling():
    # Ensure CAM-01 and CAM-02 are freshly updated
    camera_registry.register_frame("CAM-01", time.time())
    camera_registry.register_frame("CAM-02", time.time())

    # Simulate CAM-03 having disconnected 30 seconds ago
    cam3 = camera_registry._cameras.get("CAM-03")
    assert cam3 is not None
    cam3.last_seen = time.time() - 30.0
    cam3.check_heartbeat(timeout_sec=5.0)

    # CAM-03 should now be OFFLINE, while CAM-01 and CAM-02 remain ONLINE
    cam3_dict = camera_registry.get_camera("CAM-03")
    assert cam3_dict["status"] == "OFFLINE"

    cam1_dict = camera_registry.get_camera("CAM-01")
    assert cam1_dict["status"] == "ONLINE"

# 6. Frame Ingestion
def test_6_frame_ingestion():
    frame_bytes, _ = make_test_frame_bytes(color=(0, 0, 220), label_text="RED CAR INGESTION")
    ts = time.time()
    files = {"frame": ("ingest.jpg", frame_bytes, "image/jpeg")}
    data = {"camera_id": "CAM-01", "timestamp": str(ts)}
    resp = client.post("/api/cameras/frame", data=data, files=files)
    assert resp.status_code == 200

    # Verify frame is accessible via latest endpoint
    resp_latest = client.get("/api/cameras/CAM-01/latest")
    assert resp_latest.status_code == 200
    assert resp_latest.headers["content-type"] == "image/jpeg"

# 7. Timestamp Storage
def test_7_timestamp_storage():
    frames = Repository.get_frames(camera_id="CAM-01", limit=10)
    assert len(frames) > 0
    f = frames[0]
    assert "timestamp" in f
    assert "timestamp_iso" in f
    assert f["timestamp"] > 0
    assert "T" in f["timestamp_iso"]

# 8. Detection Storage
def test_8_detection_storage():
    frames = Repository.get_frames(camera_id="CAM-01", limit=1)
    assert len(frames) > 0
    fid = frames[0]["frame_id"]

    Repository.insert_detection(
        frame_id=fid,
        camera_id="CAM-01",
        timestamp=frames[0]["timestamp"],
        label="red car",
        confidence=0.94,
        box=[150.0, 100.0, 350.0, 240.0]
    )

    stored_dets = Repository.get_detections_for_frame(fid)
    assert len(stored_dets) > 0
    assert any(d["label"] == "red car" and d["confidence"] >= 0.9 for d in stored_dets)

# 9. Semantic Search
def test_9_semantic_search():
    from server.backend.models_ai.embeddings import clip_embedder
    from server.backend.models_ai.vector_index import vector_index

    # Search for "red car"
    query_vec = clip_embedder.encode_text("red car")
    results = vector_index.search(query_vec, top_k=5)
    assert len(results) > 0
    assert results[0].frame_id is not None

# 10. Camera Alias Persistence
def test_10_camera_alias_persistence():
    # Test built-in aliases
    cid, is_unknown = alias_resolver.resolve("main gate")
    assert cid == "CAM-01"
    assert is_unknown is False

    cid, is_unknown = alias_resolver.resolve("parking area")
    assert cid == "CAM-02"

    # Test unknown alias
    conn = get_connection()
    conn.execute("DELETE FROM camera_aliases WHERE alias = 'north gate'")
    conn.commit()

    cid, is_unknown = alias_resolver.resolve("north gate")
    assert cid is None
    assert is_unknown is True

    # Register new alias
    registered = alias_resolver.register_alias("north gate", "CAM-01")
    assert registered is True

    # Verify newly saved alias resolves immediately
    cid, is_unknown = alias_resolver.resolve("north gate")
    assert cid == "CAM-01"
    assert is_unknown is False

# 11. Grounded Answer Generation
def test_11_grounded_answer_generation():
    # Positive query that matches indexed red car on CAM-01
    resp = client.post("/api/query", json={"query": "Did a red car pass through the main gate in the last hour?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["matched"] is True
    assert data["camera_id"] == "CAM-01"
    assert data["confidence"] > 0
    assert data["frame_url"] is not None
    assert "explanation" in data
    assert "red car" in data["explanation"].lower() or "car" in data["explanation"].lower()

    # Negative query with no evidence
    resp_neg = client.post("/api/query", json={"query": "Did an extraterrestrial flying saucer land at CAM-01?"})
    assert resp_neg.status_code == 200
    data_neg = resp_neg.json()
    assert data_neg["matched"] is False
    assert "No matching visual evidence was found." in data_neg["explanation"]

# 12. Evidence Generation
def test_12_evidence_generation():
    frames = Repository.get_frames(camera_id="CAM-01", limit=1)
    assert len(frames) > 0
    f = frames[0]

    clip_path = clip_generator.generate_clip(
        camera_id="CAM-01",
        center_timestamp=f["timestamp"],
        frame_id=f["frame_id"],
        half_window_sec=3.0
    )
    assert clip_path is not None
    full_path = settings.BASE_DIR / clip_path
    assert full_path.exists()
    assert full_path.stat().st_size > 0

# 13. Server Restart Persistence
def test_13_server_restart_persistence():
    # Check that database connection restart retains frames, aliases, and detections
    conn = get_connection()
    count_before = conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
    assert count_before > 0

    # Re-run init_db (simulating server restart)
    init_db()

    conn_after = get_connection()
    count_after = conn_after.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
    assert count_after == count_before

    # Verify persisted alias survived restart
    cid, is_unknown = alias_resolver.resolve("north gate")
    assert cid == "CAM-01"
    assert is_unknown is False

# 14. Unknown Location Clarification

def test_14_unknown_location_requires_clarification():
    parsed = query_parser.parse("Was there a car near the reception?")
    assert parsed.location == "reception"
    assert parsed.unresolved_location == "reception"

    resp = client.post("/api/query", json={"query": "Was there a car near the reception?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["matched"] is False
    assert data["unresolved_alias_prompt"] == "reception"
    assert "I don't know which camera 'reception' refers to" in data["explanation"]


def test_15_cross_camera_continuity_timeline():
    hits = [
        {
            "camera_id": "CAM-01",
            "camera_name": "Main Gate",
            "timestamp": 1000.0,
            "timestamp_iso": "2026-10-08T09:14:00+00:00",
            "detections": [{"label": "red car", "confidence": 0.88}],
        },
        {
            "camera_id": "CAM-02",
            "camera_name": "Parking Area",
            "timestamp": 1200.0,
            "timestamp_iso": "2026-10-08T09:16:00+00:00",
            "detections": [{"label": "red car", "confidence": 0.82}],
        },
        {
            "camera_id": "CAM-03",
            "camera_name": "Exit Gate",
            "timestamp": 1500.0,
            "timestamp_iso": "2026-10-08T09:22:00+00:00",
            "detections": [{"label": "red car", "confidence": 0.79}],
        },
    ]

    timeline = build_cross_camera_timeline(hits, "red car")
    assert len(timeline) >= 2
    assert [step["camera_id"] for step in timeline] == ["CAM-01", "CAM-02", "CAM-03"]


def test_16_person_attribute_queries_keep_the_full_visual_concept():
    yellow_shirt = query_parser.parse("Is there any guy wearing a yellow shirt?")
    assert yellow_shirt.object == "person"
    assert "yellow" in (yellow_shirt.attributes or "")
    assert "person" in yellow_shirt.semantic_query
    assert "yellow" in yellow_shirt.semantic_query
    assert "shirt" in yellow_shirt.semantic_query

    large_bag = query_parser.parse("Was there someone carrying a large bag?")
    assert large_bag.object == "person"
    assert large_bag.action == "carrying"
    assert "large bag" in large_bag.semantic_query
    assert query_parser.parse("Was there a car near the reception?").semantic_query == "car"


def test_18_parser_supports_expanded_objects_actions_and_mixed_colors():
    mixed_color_person = query_parser.parse(
        "Find the person wearing a red and navy blue striped shirt"
    )
    assert mixed_color_person.object == "person"
    assert "red" in (mixed_color_person.attributes or "")
    assert "navy blue" in (mixed_color_person.attributes or "")
    assert "striped" in (mixed_color_person.attributes or "")
    assert "red and navy blue striped shirt" in mixed_color_person.semantic_query

    vehicle = query_parser.parse("Show me a delivery truck backing up")
    assert vehicle.object == "delivery truck"
    assert vehicle.action == "backing up"
    assert "backing up" in vehicle.semantic_query
    assert query_parser.parse("Show me a car").object == "car"
    assert "person" in query_parser.parse("Find me a yellow-shirt person").semantic_query

    carried_item = query_parser.parse("Find someone carrying a bright orange duffel bag")
    assert carried_item.object == "person"
    assert carried_item.action == "carrying"
    assert "bright orange duffel bag" in carried_item.semantic_query


def test_17_query_checks_recent_indexed_frames_for_person_attributes(monkeypatch):
    from server.backend.models_ai.interfaces import DetectionResult
    from server.backend.models_ai.detector import detector
    from server.backend.models_ai.vector_index import vector_index

    captured_queries = []

    monkeypatch.setattr(vector_index, "search", lambda **kwargs: [])

    def fake_detect(image, labels, threshold=0.15):
        captured_queries.extend(labels)
        if any("yellow shirt" in label for label in labels):
            return [DetectionResult(
                label="person wearing a yellow shirt",
                confidence=0.91,
                box=[100.0, 30.0, 300.0, 350.0],
            )]
        return []

    monkeypatch.setattr(detector, "detect", fake_detect)
    response = client.post("/api/query", json={"query": "Is there any guy wearing a yellow shirt?"})

    assert response.status_code == 200
    result = response.json()
    assert result["matched"] is True
    assert result["camera_id"] in {"CAM-01", "CAM-02", "CAM-03"}
    assert result["frame_url"]
    assert result["is_live_evidence"] is True
    assert any("person wearing yellow shirt" in label for label in captured_queries)


def test_19_successful_query_schedules_grounded_n8n_notification(monkeypatch):
    from server.backend.db.models import GroundedEvidenceResult
    from server.backend.routers import query as query_router

    sent_payloads = []

    async def capture_notification(payload):
        sent_payloads.append(payload)

    monkeypatch.setattr(query_router.grounded_retriever, "retrieve", lambda _query: GroundedEvidenceResult(
        matched=True,
        camera_id="CAM-02",
        camera_name="Parking Area",
        timestamp=1791535320.0,
        timestamp_iso="2026-10-09T05:02:00+00:00",
        frame_id="CAM-02_test",
        frame_url="/data/frames/CAM-02_test.jpg",
        confidence=0.9,
        relevance=0.9,
        explanation="Detected a person at Parking Area (CAM-02).",
    ))
    monkeypatch.setattr(query_router, "send_answer_notification", capture_notification)

    response = client.post("/api/query", json={"query": "Did a person enter the parking area?"})

    assert response.status_code == 200
    assert response.json()["matched"] is True
    assert sent_payloads[0] == {
        "question": "Did a person enter the parking area?",
        "answer": "Detected a person at Parking Area (CAM-02).",
        "camera": "Parking Area (CAM-02)",
        "camera_id": "CAM-02",
        "camera_name": "Parking Area",
        "timestamp": "2026-10-09T05:02:00+00:00",
        "confidence": 0.9,
        "relevance": 0.9,
        "evidence_frame": "/data/frames/CAM-02_test.jpg",
        "evidence_clip": None,
        "detections": [],
        "is_live_evidence": False,
        "source_age_seconds": None,
        "status": "answer_found",
    }


def test_20_unmatched_query_does_not_schedule_n8n_notification(monkeypatch):
    from server.backend.db.models import GroundedEvidenceResult
    from server.backend.routers import query as query_router

    sent_payloads = []

    async def capture_notification(payload):
        sent_payloads.append(payload)

    monkeypatch.setattr(query_router.grounded_retriever, "retrieve", lambda _query: GroundedEvidenceResult(
        matched=False,
        explanation="No matching visual evidence was found.",
    ))
    monkeypatch.setattr(query_router, "send_answer_notification", capture_notification)

    response = client.post("/api/query", json={"query": "Is there a purple bus?"})

    assert response.status_code == 200
    assert response.json()["matched"] is False
    assert sent_payloads == []


def test_21_n8n_unavailable_does_not_raise(monkeypatch):
    import asyncio
    import httpx
    from server.backend.notifications import n8n_webhook

    class UnavailableClient:
        def __init__(self, timeout):
            assert timeout.connect == 2.0
            assert timeout.read == 5.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def post(self, _url, json):
            raise httpx.ConnectError("test offline")

    monkeypatch.setattr(n8n_webhook.httpx, "AsyncClient", UnavailableClient)
    asyncio.run(n8n_webhook.send_answer_notification({"status": "answer_found"}))


def test_22_n8n_sender_posts_payload_to_configured_webhook(monkeypatch):
    import asyncio
    from server.backend.notifications import n8n_webhook

    requests_sent = []

    class SuccessResponse:
        is_error = False
        status_code = 200

    class SuccessClient:
        def __init__(self, timeout):
            assert timeout.connect == 2.0
            assert timeout.read == 5.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def post(self, url, json):
            requests_sent.append((url, json))
            return SuccessResponse()

    monkeypatch.setattr(n8n_webhook.httpx, "AsyncClient", SuccessClient)
    payload = {
        "question": "Find the yellow shirt person",
        "answer": "Person detected.",
        "camera": "Main Gate (CAM-01)",
        "timestamp": "2026-10-09T05:02:00+00:00",
        "confidence": 0.9,
        "relevance": 0.8,
        "evidence_frame": "/data/frames/frame.jpg",
        "status": "answer_found",
    }

    asyncio.run(n8n_webhook.send_answer_notification(payload))

    posted_payload = requests_sent[0][1]
    assert requests_sent[0][0] == n8n_webhook.N8N_WEBHOOK_URL
    assert {key: value for key, value in posted_payload.items() if key != "text"} == payload
    assert posted_payload["text"] == (
        "📍 Main Gate (CAM-01) · 2026-10-09 05:02:00 UTC\n"
        "Person detected.\n"
        "Q: Find the yellow shirt person\n"
        "Confidence 90% · Relevance 80%\n"
        "Frame: /data/frames/frame.jpg"
    )


def test_23_n8n_telegram_text_includes_detection_and_evidence_details():
    from server.backend.notifications.n8n_webhook import _format_telegram_message

    message = _format_telegram_message({
        "camera": "Parking Area (CAM-02)",
        "timestamp": "2026-10-09T05:02:00+00:00",
        "answer": "Person detected.",
        "question": "Did someone enter?",
        "confidence": 0.9,
        "relevance": 0.8,
        "is_live_evidence": False,
        "source_age_seconds": 90,
        "detections": [{"label": "person", "confidence": 0.95}],
        "evidence_frame": "https://example.test/frame.jpg",
        "evidence_clip": "https://example.test/clip.mp4",
    })

    assert "2026-10-09 05:02:00 UTC" in message
    assert "Historical · 1.5 min old" in message
    assert "Detected: person 95%" in message
    assert "Frame: https://example.test/frame.jpg" in message
    assert "Clip: https://example.test/clip.mp4" in message


def test_camera_ai_indexing_does_not_block_frame_ingestion(monkeypatch):
    import threading
    from server.backend.routers import cameras as cameras_router

    indexing_started = threading.Event()
    allow_indexing_to_finish = threading.Event()

    def slow_index(_frame_info):
        indexing_started.set()
        allow_indexing_to_finish.wait(timeout=5)

    monkeypatch.setattr(cameras_router, "_index_frame", slow_index)
    frame_bytes, _ = make_test_frame_bytes(label_text="ASYNC INDEX")
    response = client.post(
        "/api/cameras/frame",
        data={"camera_id": "CAM-LATENCY-TEST", "timestamp": str(time.time())},
        files={"frame": ("latency.jpg", frame_bytes, "image/jpeg")},
    )

    try:
        assert response.status_code == 200
        assert response.json()["sampled_for_indexing"] is True
        assert indexing_started.wait(timeout=2)
        assert client.get("/health").status_code == 200
    finally:
        allow_indexing_to_finish.set()
        cameras_router._ai_index_queue.join()


def test_camera_buffer_only_reports_new_preview_frames():
    from server.backend.ingestion.buffer_manager import CameraBufferManager

    buffer = CameraBufferManager(max_seconds=2)
    buffer.add_frame("CAM-01", 1.0, b"first")
    first = buffer.get_latest_frame_with_sequence("CAM-01")
    assert first is not None
    assert first[1] == b"first"
    assert buffer.get_latest_frame_with_sequence("CAM-01") == first

    buffer.add_frame("CAM-01", 1.1, b"second")
    second = buffer.get_latest_frame_with_sequence("CAM-01")
    assert second is not None
    assert second[0] > first[0]
    assert second[1] == b"second"


def test_mjpeg_camera_stream_yields_each_new_buffered_frame(monkeypatch):
    import asyncio
    from server.backend.ingestion.buffer_manager import CameraBufferManager
    from server.backend.routers import cameras as cameras_router

    buffer = CameraBufferManager(max_seconds=2)
    buffer.add_frame("CAM-01", time.time(), b"first-frame")
    monkeypatch.setattr(cameras_router, "buffer_manager", buffer)
    response = asyncio.run(cameras_router.mjpeg_stream("CAM-01"))

    async def collect_frames():
        first = await anext(response.body_iterator)
        buffer.add_frame("CAM-01", time.time(), b"second-frame")
        second = await anext(response.body_iterator)
        await response.body_iterator.aclose()
        return first, second

    first, second = asyncio.run(collect_frames())
    assert b"first-frame" in first
    assert b"second-frame" in second
    assert response.headers["cache-control"] == "no-cache, no-store"

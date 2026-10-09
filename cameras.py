import time
import asyncio
import logging
import queue
import threading
from typing import Optional
import cv2
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Response
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from server.backend.config import settings
from server.backend.ingestion.camera_registry import camera_registry
from server.backend.ingestion.buffer_manager import buffer_manager
from server.backend.ingestion.frame_sampler import frame_sampler
from server.backend.nlp.alias_resolver import alias_resolver
from server.backend.db.models import CameraAliasModel
from server.backend.models_ai.embeddings import clip_embedder
from server.backend.models_ai.vector_index import vector_index
from server.backend.models_ai.detector import detector
from server.backend.db.repository import Repository

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/cameras", tags=["Cameras"])
_ai_index_queue: queue.Queue = queue.Queue()
_ai_worker_lock = threading.Lock()
_ai_worker_started = False


def _index_frame(frame_info: dict) -> None:
    """
    AI indexing callback for sampled frames:
    1. Computes CLIP visual embedding and indexes in FAISS.
    2. Runs open-vocabulary detector on common security classes and saves detections.
    """
    try:
        img = cv2.imread(frame_info["full_path"])
        if img is None:
            raise OSError(f"Could not read sampled frame {frame_info['full_path']}.")
        fid = frame_info["frame_id"]
        cid = frame_info["camera_id"]
        ts = frame_info["timestamp"]

        # 1. Visual embedding & FAISS indexing
        emb = clip_embedder.encode_image(img)
        vector_index.add(emb, frame_id=fid, camera_id=cid, timestamp=ts)

        # 2. General baseline open-vocabulary detection
        common_classes = ["car", "truck", "person", "bag", "bicycle"]
        dets = detector.detect(img, common_classes, threshold=settings.DETECTION_THRESHOLD)
        for d in dets:
            Repository.insert_detection(
                frame_id=fid,
                camera_id=cid,
                timestamp=ts,
                label=d.label,
                confidence=d.confidence,
                box=d.box
            )
    except Exception as e:
        logger.error(f"Error in background AI indexing for {frame_info.get('frame_id')}: {e}")


def _ai_index_worker() -> None:
    while True:
        frame_info = _ai_index_queue.get()
        try:
            _index_frame(frame_info)
        finally:
            _ai_index_queue.task_done()


def _ai_indexing_worker(frame_info: dict) -> None:
    global _ai_worker_started
    queue_item = {
        key: frame_info[key]
        for key in ("frame_id", "camera_id", "timestamp", "full_path")
    }
    with _ai_worker_lock:
        if not _ai_worker_started:
            threading.Thread(
                target=_ai_index_worker,
                name="cctv-ai-indexer",
                daemon=True,
            ).start()
            _ai_worker_started = True
    _ai_index_queue.put_nowait(queue_item)


def _process_ingested_frame(
    camera_id: str,
    timestamp: Optional[float],
    camera_name: Optional[str],
    frame_bytes: bytes,
) -> dict:
    cid = camera_id.strip().upper()
    ts = timestamp or time.time()

    if not frame_bytes:
        raise HTTPException(status_code=400, detail="Empty frame received")

    camera_registry.register_frame(cid, ts, camera_name)
    buffer_manager.add_frame(cid, ts, frame_bytes)
    sampled = frame_sampler.process_and_save_frame(
        camera_id=cid,
        timestamp=ts,
        frame_bytes=frame_bytes,
        ai_pipeline_callback=_ai_indexing_worker,
    )

    return {
        "status": "received",
        "camera_id": cid,
        "timestamp": ts,
        "sampled_for_indexing": sampled is not None
    }


@router.post("/frame")
async def ingest_frame(
    camera_id: str = Form(...),
    timestamp: Optional[float] = Form(None),
    camera_name: Optional[str] = Form(None),
    frame: UploadFile = File(...)
):
    """
    Primary ingestion endpoint for CCTV CAM-01, CAM-02, CAM-03 laptops.
    Distinguishes streams, maintains heartbeat, buffers frames, and samples 1 FPS for AI.
    """
    cid = camera_id.strip().upper()
    ts = timestamp or time.time()
    frame_bytes = await frame.read()

    return await run_in_threadpool(
        _process_ingested_frame,
        cid,
        ts,
        camera_name,
        frame_bytes,
    )

@router.get("")
def list_cameras():
    """Returns registry of all cameras with status ONLINE/OFFLINE, last seen, FPS, frame count."""
    camera_registry.check_all_heartbeats()
    return camera_registry.get_all_cameras()

@router.get("/{camera_id}")
def get_camera(camera_id: str):
    cid = camera_id.strip().upper()
    cam = camera_registry.get_camera(cid)
    if not cam:
        raise HTTPException(status_code=404, detail=f"Camera {cid} not found")
    return cam

@router.get("/{camera_id}/latest")
def get_latest_frame(camera_id: str):
    """Returns latest JPEG frame from memory buffer or disk."""
    cid = camera_id.strip().upper()
    frame_bytes = buffer_manager.get_latest_frame(cid)
    if frame_bytes:
        return Response(content=frame_bytes, media_type="image/jpeg")

    # Fallback to latest stored frame on disk
    frames = Repository.get_frames(camera_id=cid, limit=1)
    if frames:
        f_path = settings.BASE_DIR / frames[0]["frame_path"]
        if f_path.exists():
            with open(f_path, "rb") as f:
                return Response(content=f.read(), media_type="image/jpeg")

    raise HTTPException(status_code=404, detail="No frame available for camera")

@router.get("/{camera_id}/stream")
async def mjpeg_stream(camera_id: str):
    """MJPEG live streaming endpoint for browser viewing."""
    cid = camera_id.strip().upper()

    async def frame_generator():
        last_sequence = 0
        while True:
            latest = buffer_manager.get_latest_frame_with_sequence(cid)
            if latest is not None and latest[0] != last_sequence:
                last_sequence, frame_bytes = latest
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            await asyncio.sleep(0.03)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store",
            "X-Accel-Buffering": "no",
        },
    )

# Camera Alias Management
@router.get("/aliases/all")
def get_aliases():
    return alias_resolver.get_all_mappings()

@router.post("/alias")
def register_alias(payload: CameraAliasModel):
    success = alias_resolver.register_alias(payload.alias, payload.camera_id)
    if not success:
        raise HTTPException(status_code=400, detail="Invalid alias or camera ID")
    return {
        "status": "success",
        "alias": payload.alias.lower().strip(),
        "camera_id": payload.camera_id.upper().strip()
    }

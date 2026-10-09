import time
from fastapi import APIRouter
from server.backend.config import settings
from server.backend.db.database import get_connection
from server.backend.ingestion.camera_registry import camera_registry
from server.backend.models_ai.detector import detector
from server.backend.models_ai.embeddings import clip_embedder

router = APIRouter(tags=["Health"])

START_TIME = time.time()

@router.get("/health")
def health_check():
    camera_registry.check_all_heartbeats()
    cameras = camera_registry.get_all_cameras()

    # Query indexed frames count
    conn = get_connection()
    frame_count = conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
    detections_count = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]

    device = settings.get_device()
    detector_mode = (
        "owlvit"
        if detector._model is not None
        else "fallback"
        if detector._fallback_mode
        else "not_loaded"
    )
    embedding_mode = (
        "clip"
        if clip_embedder._model is not None
        else "fallback"
        if clip_embedder._fallback_mode
        else "not_loaded"
    )
    gpu_available = False
    gpu_name = None
    if device == "cuda":
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass

    return {
        "status": "healthy",
        "service": "CCTV Intelligence Server",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "device": device,
        "vision_models": {
            "detector": detector_mode,
            "embeddings": embedding_mode,
        },
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "cameras_registered": len(cameras),
        "cameras_online": sum(1 for c in cameras if c["status"] == "ONLINE"),
        "total_indexed_frames": frame_count,
        "total_detections": detections_count,
        "storage": {
            "sampling_fps": settings.SAMPLING_FPS,
            "resolution": f"{settings.FRAME_WIDTH}x{settings.FRAME_HEIGHT}",
            "frames_dir": str(settings.FRAMES_DIR),
            "evidence_dir": str(settings.EVIDENCE_DIR)
        }
    }

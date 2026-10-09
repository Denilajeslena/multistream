from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from server.backend.config import settings
from server.backend.db.database import get_connection
from server.backend.db.repository import Repository

router = APIRouter(prefix="/api/evidence", tags=["Evidence"])

@router.get("/list")
def list_evidence_clips():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM evidence_clips ORDER BY created_at DESC LIMIT 50").fetchall()
    return [dict(r) for r in rows]

@router.get("/{clip_id}")
def get_evidence_clip_info(clip_id: str):
    clip = Repository.get_evidence_clip(clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Evidence clip not found")
    return clip

@router.get("/{clip_id}/video")
def stream_evidence_clip_video(clip_id: str):
    clip = Repository.get_evidence_clip(clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Evidence clip not found")
    file_path = settings.BASE_DIR / clip["clip_path"]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found on disk")
    return FileResponse(str(file_path), media_type="video/mp4")

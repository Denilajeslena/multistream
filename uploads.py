import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from server.backend.config import settings
from server.backend.ingestion.uploaded_video import (
    ALLOWED_VIDEO_EXTENSIONS,
    MAX_VIDEO_BYTES,
    create_job,
    get_job,
    public_job,
    query_video_async,
    release_processing_slot,
    reserve_processing_slot,
    submit_video_processing,
)
from server.backend.notifications.n8n_webhook import send_answer_notification

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/uploads", tags=["Uploaded Videos"])
UPLOAD_CHUNK_BYTES = 1024 * 1024


class UploadedVideoQuery(BaseModel):
    question: str


@router.post("/videos", status_code=202)
async def upload_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
):
    filename = Path(video.filename or "").name
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format. Use one of: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}.",
        )

    if not reserve_processing_slot():
        raise HTTPException(
            status_code=429,
            detail="Uploaded video processing is busy. Try again when the current upload finishes.",
        )

    job_id = uuid.uuid4().hex
    try:
        job_dir = settings.UPLOADS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        video_path = job_dir / f"source{extension}"

        total_bytes = 0
        with video_path.open("wb") as output:
            while chunk := await video.read(UPLOAD_CHUNK_BYTES):
                total_bytes += len(chunk)
                if total_bytes > MAX_VIDEO_BYTES:
                    raise HTTPException(status_code=413, detail="Video exceeds the 512 MB upload limit.")
                output.write(chunk)
    except Exception:
        if "video_path" in locals():
            video_path.unlink(missing_ok=True)
        if "job_dir" in locals():
            try:
                job_dir.rmdir()
            except OSError:
                logger.warning("Could not remove incomplete video upload directory %s.", job_dir)
        release_processing_slot()
        raise
    finally:
        await video.close()

    if total_bytes == 0:
        video_path.unlink(missing_ok=True)
        job_dir.rmdir()
        release_processing_slot()
        raise HTTPException(status_code=400, detail="Uploaded video is empty.")

    create_job(job_id, filename, video_path, job_dir)
    background_tasks.add_task(submit_video_processing, job_id)
    job = get_job(job_id)
    assert job is not None
    return public_job(job)


@router.get("/{job_id}")
def get_upload_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Uploaded video job not found.")
    return public_job(job)


@router.get("/{job_id}/frames/{frame_name}")
def get_uploaded_frame(job_id: str, frame_name: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Uploaded video job not found.")
    frame_path = job["job_dir"] / "frames" / Path(frame_name).name
    allowed_frames = {Path(frame["path"]).name for frame in job["frames"]}
    if Path(frame_name).name not in allowed_frames or not frame_path.is_file():
        raise HTTPException(status_code=404, detail="Uploaded evidence frame not found.")
    return FileResponse(str(frame_path), media_type="image/jpeg")


@router.post("/{job_id}/query")
async def query_uploaded_video(
    job_id: str,
    payload: UploadedVideoQuery,
    background_tasks: BackgroundTasks,
):
    question = payload.question
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Uploaded video job not found.")
    if job["status"] == "processing":
        raise HTTPException(status_code=409, detail="Uploaded video is still processing.")
    if job["status"] == "failed":
        raise HTTPException(
            status_code=422,
            detail=job.get("error") or "Uploaded video processing failed.",
        )

    try:
        result = await query_video_async(job_id, question.strip())
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if result["matched"] and result["frame_url"] and result["timestamp_iso"]:
        notification = {
            "source": "uploaded_video",
            "question": question.strip(),
            "answer": result["explanation"],
            "camera": "Uploaded Video",
            "timestamp": result["timestamp_iso"],
            "confidence": result["confidence"],
            "relevance": result["relevance"],
            "evidence_frame": result["frame_url"],
            "detections": result["detections"],
            "status": "answer_found",
        }
        background_tasks.add_task(send_answer_notification, notification)
    return result

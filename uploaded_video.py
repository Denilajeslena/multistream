import asyncio
import logging
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2

from server.backend.config import settings
from server.backend.models_ai.detector import detector
from server.backend.nlp.query_parser import query_parser

logger = logging.getLogger(__name__)

MAX_VIDEO_BYTES = 512 * 1024 * 1024
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
MAX_QUEUED_VIDEO_JOBS = 1

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_processing_slots = threading.BoundedSemaphore(MAX_QUEUED_VIDEO_JOBS + 1)
_video_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="uploaded-video")


def create_job(job_id: str, filename: str, video_path: Path, job_dir: Path) -> None:
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "filename": filename,
            "video_path": video_path,
            "job_dir": job_dir,
            "status": "processing",
            "progress": 0,
            "total_frames": 0,
            "frame_count": 0,
            "duration_seconds": 0.0,
            "frames": [],
            "query_cache": {},
            "query_lock": threading.Lock(),
            "error": None,
        }


def get_job(job_id: str) -> dict[str, Any] | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        key: job[key]
        for key in (
            "job_id",
            "filename",
            "status",
            "progress",
            "total_frames",
            "frame_count",
            "duration_seconds",
            "error",
        )
    }


def reserve_processing_slot() -> bool:
    return _processing_slots.acquire(blocking=False)


def release_processing_slot() -> None:
    _processing_slots.release()


def submit_video_processing(job_id: str) -> None:
    def run_job() -> None:
        try:
            process_video(job_id)
        finally:
            _processing_slots.release()

    try:
        _video_executor.submit(run_job)
    except RuntimeError as exc:
        _processing_slots.release()
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job.update({"status": "failed", "error": str(exc)})
                job["video_path"].unlink(missing_ok=True)
        logger.exception("Could not queue uploaded video job %s.", job_id)
        raise


def process_video(job_id: str) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            logger.error("Cannot process missing uploaded video job %s.", job_id)
            return
        video_path = job["video_path"]
        job_dir = job["job_dir"]

    capture = None
    frames_dir = None
    processing_failed = False
    try:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError("The uploaded file could not be opened as a video.")

        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 25.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_seconds = total_frames / fps if total_frames else 0.0
        sample_interval_frames = max(1, round(fps / max(settings.SAMPLING_FPS, 0.1)))
        frame_index = 0
        last_sampled_image = None
        last_sampled_seconds = -1.0
        sampled_frames: list[dict[str, Any]] = []
        frames_dir = job_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        with _jobs_lock:
            job = _jobs[job_id]
            job["total_frames"] = total_frames
            job["duration_seconds"] = duration_seconds

        while True:
            success = capture.grab()
            if not success:
                break

            if frame_index % sample_interval_frames == 0:
                success, image = capture.retrieve()
                if not success:
                    frame_index += 1
                    continue

                if (
                    last_sampled_image is not None
                    and image.shape == last_sampled_image.shape
                    and cv2.norm(image, last_sampled_image, cv2.NORM_INF) == 0
                ):
                    frame_index += 1
                    continue

                last_sampled_image = image
                reported_seconds = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                timestamp_seconds = (
                    reported_seconds
                    if reported_seconds > last_sampled_seconds
                    else frame_index / fps
                )
                if image.shape[1] != settings.FRAME_WIDTH or image.shape[0] != settings.FRAME_HEIGHT:
                    image = cv2.resize(
                        image,
                        (settings.FRAME_WIDTH, settings.FRAME_HEIGHT),
                        interpolation=cv2.INTER_AREA,
                    )

                image_name = f"frame-{len(sampled_frames):06d}.jpg"
                image_path = frames_dir / image_name
                if not cv2.imwrite(
                    str(image_path),
                    image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), settings.JPEG_QUALITY],
                ):
                    raise OSError(f"Could not save sampled frame {image_name}.")
                sampled_frames.append({
                    "timestamp_seconds": max(0.0, timestamp_seconds),
                    "path": image_path,
                })
                last_sampled_seconds = timestamp_seconds

            frame_index += 1
            if frame_index % 100 == 0 or frame_index == total_frames:
                with _jobs_lock:
                    job = _jobs.get(job_id)
                    if job is not None:
                        job["progress"] = (
                            min(99, int(frame_index * 100 / total_frames))
                            if total_frames
                            else 0
                        )
                        job["frame_count"] = len(sampled_frames)

        if frame_index == 0:
            raise ValueError("The uploaded video contains no readable frames.")

        with _jobs_lock:
            job = _jobs[job_id]
            job.update({
                "status": "completed",
                "progress": 100,
                "total_frames": total_frames or frame_index,
                "duration_seconds": duration_seconds or frame_index / fps,
                "frame_count": len(sampled_frames),
                "frames": sampled_frames,
            })
    except Exception as exc:
        processing_failed = True
        logger.exception("Uploaded video processing failed for job %s.", job_id)
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job.update({"status": "failed", "error": str(exc)})
    finally:
        if capture is not None:
            capture.release()
        try:
            video_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove processed upload source %s.", video_path)
        if processing_failed and frames_dir is not None:
            try:
                shutil.rmtree(frames_dir)
                job_dir.rmdir()
            except OSError:
                logger.warning("Could not remove partial artifacts for video job %s.", job_id)


def format_video_timestamp(seconds: float) -> str:
    whole_seconds = max(0, int(seconds))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"
    return f"{minutes:02d}:{seconds_part:02d}"


def query_video(job_id: str, question: str) -> dict[str, Any]:
    job = get_job(job_id)
    if job is None:
        raise KeyError(job_id)
    if job["status"] != "completed":
        raise RuntimeError(f"Video processing is {job['status']}.")

    cache_key = question.strip().casefold()
    with job["query_lock"]:
        cached_result = job["query_cache"].get(cache_key)
        if cached_result is not None:
            return dict(cached_result)
        result = _query_video_frames(job, question)
        if len(job["query_cache"]) >= 16:
            job["query_cache"].pop(next(iter(job["query_cache"])))
        job["query_cache"][cache_key] = result
        return dict(result)


async def query_video_async(job_id: str, question: str) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_video_executor, query_video, job_id, question)


def _query_video_frames(job: dict[str, Any], question: str) -> dict[str, Any]:
    job_id = job["job_id"]
    parsed_query = query_parser.parse(question)
    detection_labels = list(dict.fromkeys(
        label.strip()
        for label in (parsed_query.semantic_query, parsed_query.object or "")
        if label.strip()
    ))
    best_match: dict[str, Any] | None = None

    for frame in job["frames"]:
        image = cv2.imread(str(frame["path"]))
        if image is None:
            raise OSError(f"Could not read processed frame {frame['path'].name}.")
        matches = detector.detect(
            image,
            detection_labels,
            threshold=settings.DETECTION_THRESHOLD,
        )
        if not matches:
            continue

        best_detection = max(matches, key=lambda detection: detection.confidence)
        timestamp_seconds = frame["timestamp_seconds"]
        candidate = {
            "timestamp_seconds": timestamp_seconds,
            "timestamp": format_video_timestamp(timestamp_seconds),
            "confidence": round(float(best_detection.confidence), 2),
            "detection": best_detection.model_dump(),
            "frame_url": f"/api/uploads/{job_id}/frames/{Path(frame['path']).name}",
            "frame_id": Path(frame["path"]).stem,
        }
        if best_match is None or (
            candidate["confidence"], candidate["timestamp_seconds"]
        ) > (
            best_match["confidence"], best_match["timestamp_seconds"]
        ):
            best_match = candidate

    if best_match is None:
        return {
            "matched": False,
            "camera_id": "Uploaded Video",
            "camera_name": "Uploaded Video",
            "timestamp": None,
            "timestamp_iso": None,
            "frame_id": None,
            "frame_url": None,
            "clip_url": None,
            "confidence": 0.0,
            "relevance": 0.0,
            "detections": [],
            "is_live_evidence": False,
            "explanation": "No matching visual evidence was found in the uploaded video.",
            "source": "uploaded_video",
        }

    timestamp = best_match["timestamp"]
    description = parsed_query.semantic_query.strip().lower()
    description = re.sub(r"^person\s+use\b", "a person using", description)
    description = re.sub(r"^person\b", "a person", description)
    description = re.sub(
        r"\b(using|holding|carrying|with)\s+(mobile phone|phone|laptop|bag|backpack|handbag|bottle|cup)\b",
        r"\1 a \2",
        description,
    )
    if not description.startswith(("a ", "an ", "the ")):
        article = "an" if description[:1] in "aeiou" else "a"
        description = f"{article} {description}"
    return {
        "matched": True,
        "camera_id": "Uploaded Video",
        "camera_name": "Uploaded Video",
        "timestamp": best_match["timestamp_seconds"],
        "timestamp_iso": timestamp,
        "frame_id": best_match["frame_id"],
        "frame_url": best_match["frame_url"],
        "clip_url": None,
        "confidence": best_match["confidence"],
        "relevance": best_match["confidence"],
        "detections": [best_match["detection"]],
        "is_live_evidence": False,
        "explanation": f"Yes, {description} was detected at {timestamp}.",
        "source": "uploaded_video",
    }

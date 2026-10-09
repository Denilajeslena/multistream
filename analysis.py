from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from server.backend.config import settings
from server.backend.ingestion.realtime_analysis import realtime_analysis_service

router = APIRouter(prefix="/api/analysis", tags=["Real-time Analysis"])


class AnalysisSelection(BaseModel):
    camera_id: str
    model: Literal["existing", "yolo11n"] = "existing"


@router.get("/status")
def get_analysis_status():
    return realtime_analysis_service.status()


@router.post("/start")
def start_analysis(selection: AnalysisSelection):
    if not settings.REALTIME_ANALYSIS:
        raise HTTPException(status_code=404, detail="Real-time analysis is disabled.")
    try:
        return realtime_analysis_service.start(selection.camera_id, selection.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/stop")
def stop_analysis():
    if not settings.REALTIME_ANALYSIS:
        raise HTTPException(status_code=404, detail="Real-time analysis is disabled.")
    return realtime_analysis_service.stop()


@router.get("/events")
def get_analysis_events(
    camera_id: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=250),
):
    if not settings.REALTIME_ANALYSIS:
        return []
    return realtime_analysis_service.events(camera_id, limit)

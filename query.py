import logging
from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, BackgroundTasks, HTTPException

from server.backend.retrieval.retriever import grounded_retriever
from server.backend.nlp.alias_resolver import alias_resolver
from server.backend.db.models import GroundedEvidenceResult
from server.backend.notifications.n8n_webhook import send_answer_notification

router = APIRouter(prefix="/api/query", tags=["Query"])
logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    query: str

class AliasResolutionRequest(BaseModel):
    alias: str
    camera_id: str
    original_query: Optional[str] = None


def _notification_payload(query_text: str, result: GroundedEvidenceResult) -> dict:
    camera_name = result.camera_name or result.camera_id
    return {
        "question": query_text,
        "answer": result.explanation,
        "camera": f"{camera_name} ({result.camera_id})",
        "camera_id": result.camera_id,
        "camera_name": camera_name,
        "timestamp": result.timestamp_iso,
        "confidence": result.confidence,
        "relevance": result.relevance,
        "evidence_frame": result.frame_url,
        "evidence_clip": result.clip_url,
        "detections": [detection.model_dump() for detection in result.detections],
        "is_live_evidence": result.is_live_evidence,
        "source_age_seconds": result.source_age_seconds,
        "status": "answer_found",
    }


@router.post("", response_model=GroundedEvidenceResult)
def handle_query(payload: QueryRequest, background_tasks: BackgroundTasks):
    """
    Executes grounded natural-language retrieval:
    Parses question, resolves camera alias, searches vector index, verifies with detector,
    and returns matching frame + evidence clip or 'No matching visual evidence was found.'
    """
    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    query_text = payload.query.strip()
    result = grounded_retriever.retrieve(query_text)
    if result.matched and result.camera_id and result.timestamp_iso and result.frame_url:
        logger.info("Scheduling n8n notification for grounded CCTV answer from %s.", result.camera_id)
        background_tasks.add_task(send_answer_notification, _notification_payload(query_text, result))
    return result

@router.post("/alias-resolve", response_model=GroundedEvidenceResult)
def resolve_alias_and_retry(payload: AliasResolutionRequest, background_tasks: BackgroundTasks):
    """
    Saves an unknown camera alias to SQLite (survives restarts) and optionally re-runs query.
    """
    alias_resolver.register_alias(payload.alias, payload.camera_id)

    if payload.original_query:
        query_text = payload.original_query.strip()
        result = grounded_retriever.retrieve(query_text)
        if result.matched and result.camera_id and result.timestamp_iso and result.frame_url:
            logger.info("Scheduling n8n notification for grounded CCTV answer from %s.", result.camera_id)
            background_tasks.add_task(send_answer_notification, _notification_payload(query_text, result))
        return result

    return GroundedEvidenceResult(
        matched=False,
        confidence=1.0,
        relevance=1.0,
        explanation=f"Alias '{payload.alias}' has been successfully mapped to {payload.camera_id} and saved permanently."
    )

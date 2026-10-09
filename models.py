from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class CameraModel(BaseModel):
    camera_id: str
    camera_name: str
    status: str = "OFFLINE"
    last_seen: float = 0.0
    fps: float = 0.0
    frame_count: int = 0
    preview_url: Optional[str] = None

class CameraAliasModel(BaseModel):
    alias: str
    camera_id: str

class DetectionItem(BaseModel):
    label: str
    confidence: float
    box: List[float] = Field(..., description="[x1, y1, x2, y2] normalized or pixel coords")

class FrameRecord(BaseModel):
    frame_id: str
    camera_id: str
    timestamp: float
    timestamp_iso: str
    frame_path: str
    width: int = 640
    height: int = 360

class ParsedQuery(BaseModel):
    raw_query: str
    object: Optional[str] = None
    attributes: Optional[str] = None
    location: Optional[str] = None
    time_range_seconds: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    action: Optional[str] = None
    semantic_query: str
    resolved_camera_id: Optional[str] = None
    unresolved_location: Optional[str] = None

class GroundedEvidenceResult(BaseModel):
    matched: bool
    camera_id: Optional[str] = None
    camera_name: Optional[str] = None
    timestamp: Optional[float] = None
    timestamp_iso: Optional[str] = None
    frame_id: Optional[str] = None
    frame_url: Optional[str] = None
    clip_url: Optional[str] = None
    confidence: float = 0.0
    relevance: float = 0.0
    detections: List[DetectionItem] = Field(default_factory=list)
    timeline: List[Dict[str, Any]] = Field(default_factory=list)
    is_live_evidence: bool = False
    source_age_seconds: Optional[float] = None
    explanation: str
    unresolved_alias_prompt: Optional[str] = None

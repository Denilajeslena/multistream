import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import dotenv_values

_LOCAL_ENV = dotenv_values(Path(__file__).resolve().parent.parent / ".env")


def _env_value(name: str, default: str) -> str:
    return os.getenv(name, _LOCAL_ENV.get(name) or default)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# Base directory for server
SERVER_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseModel):
    # Network
    SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8000"))

    # Directories
    BASE_DIR: Path = SERVER_DIR
    DATA_DIR: Path = SERVER_DIR / os.getenv("DATA_DIR", "data")
    MODELS_DIR: Path = SERVER_DIR / os.getenv("MODELS_DIR", "models")
    FRAMES_DIR: Path = DATA_DIR / "frames"
    EVIDENCE_DIR: Path = DATA_DIR / "evidence"
    UPLOADS_DIR: Path = DATA_DIR / "uploads"
    INDEX_DIR: Path = DATA_DIR / "index"
    DB_PATH: Path = DATA_DIR / "database.sqlite3"

    # Frame processing & sampling
    SAMPLING_FPS: float = float(os.getenv("SAMPLING_FPS", "1.0"))
    LIVE_QUERY_WINDOW_SEC: float = float(os.getenv("LIVE_QUERY_WINDOW_SEC", "120"))
    LIVE_QUERY_FRAMES_PER_CAMERA: int = int(os.getenv("LIVE_QUERY_FRAMES_PER_CAMERA", "5"))
    FRAME_WIDTH: int = int(os.getenv("FRAME_WIDTH", "640"))
    FRAME_HEIGHT: int = int(os.getenv("FRAME_HEIGHT", "360"))
    JPEG_QUALITY: int = int(os.getenv("JPEG_QUALITY", "80"))
    BUFFER_SECONDS: int = int(os.getenv("BUFFER_SECONDS", "20"))
    EVIDENCE_HALF_WINDOW_SEC: float = float(os.getenv("EVIDENCE_HALF_WINDOW_SEC", "5.0"))

    # AI Models (No API key needed)
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "openai/clip-vit-base-patch32")
    DETECTOR_MODEL_NAME: str = os.getenv("DETECTOR_MODEL_NAME", "google/owlvit-base-patch32")
    DETECTION_THRESHOLD: float = float(os.getenv("DETECTION_THRESHOLD", "0.15"))
    SEMANTIC_SEARCH_THRESHOLD: float = float(os.getenv("SEMANTIC_SEARCH_THRESHOLD", "0.18"))
    DEVICE: str = os.getenv("DEVICE", "auto")

    # Optional, isolated real-time analysis
    REALTIME_ANALYSIS: bool = _env_bool(
        "REALTIME_ANALYSIS",
        _env_value("REALTIME_ANALYSIS", "false").lower() in {"1", "true", "yes", "on"},
    )
    REALTIME_INFERENCE_FPS: float = float(_env_value("REALTIME_INFERENCE_FPS", "3.0"))
    REALTIME_FRAME_WIDTH: int = int(_env_value("REALTIME_FRAME_WIDTH", "640"))
    REALTIME_FRAME_HEIGHT: int = int(_env_value("REALTIME_FRAME_HEIGHT", "360"))
    REALTIME_EVENT_COOLDOWN_SEC: float = float(_env_value("REALTIME_EVENT_COOLDOWN_SEC", "20"))
    REALTIME_YOLO_MODEL: str = _env_value("REALTIME_YOLO_MODEL", "yolo11n.pt")
    REALTIME_YOLO_IMAGE_SIZE: int = int(_env_value("REALTIME_YOLO_IMAGE_SIZE", "416"))
    REALTIME_DEVICE: str = _env_value("REALTIME_DEVICE", "cpu")
    N8N_NOTIFICATIONS: bool = _env_bool(
        "N8N_NOTIFICATIONS",
        _env_value("N8N_NOTIFICATIONS", "false").lower() in {"1", "true", "yes", "on"},
    )
    N8N_WEBHOOK_URL: str = _env_value(
        "N8N_WEBHOOK_URL",
        "https://nitharsh.app.n8n.cloud/webhook/video-events",
    )

    # Inactivity / Disconnect timeout (seconds)
    CAMERA_OFFLINE_TIMEOUT_SEC: float = float(os.getenv("CAMERA_OFFLINE_TIMEOUT_SEC", "6.0"))

    def get_device(self) -> str:
        if self.DEVICE != "auto":
            return self.DEVICE
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def ensure_directories(self):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        self.EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        self.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_directories()

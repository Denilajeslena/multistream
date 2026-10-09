from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from pydantic import BaseModel

class DetectionResult(BaseModel):
    label: str
    confidence: float
    box: List[float]  # [x1, y1, x2, y2]

class VectorMetadata(BaseModel):
    index_id: int
    frame_id: str
    camera_id: str
    timestamp: float

class VectorSearchResult(BaseModel):
    frame_id: str
    camera_id: str
    timestamp: float
    score: float

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray, labels: List[str], threshold: float = 0.15) -> List[DetectionResult]:
        """Detect objects with open-vocabulary text labels."""
        pass

class BaseEmbeddingModel(ABC):
    @abstractmethod
    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """Compute normalized L2 image embedding."""
        pass

    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        """Compute normalized L2 text embedding."""
        pass

class BaseVectorIndex(ABC):
    @abstractmethod
    def add(self, vector: np.ndarray, frame_id: str, camera_id: str, timestamp: float):
        """Add vector and its frame mapping to index."""
        pass

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        camera_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[VectorSearchResult]:
        """Search vector index with optional camera and time range filter."""
        pass

    @abstractmethod
    def save(self):
        """Persist index and metadata to disk."""
        pass

    @abstractmethod
    def load(self):
        """Load index and metadata from disk."""
        pass

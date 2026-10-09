import os
import json
import logging
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from server.backend.config import settings
from server.backend.models_ai.interfaces import BaseVectorIndex, VectorSearchResult

logger = logging.getLogger(__name__)

class FaissVectorIndex(BaseVectorIndex):
    """
    FAISS-based vector index with disk persistence and metadata mapping.
    Maps every vector to: frame_id, camera_id, timestamp.
    Supports camera and time filtering.
    """
    def __init__(self, dimension: int = 512, index_dir: Optional[Path] = None):
        self.dimension = dimension
        self.index_dir = index_dir or settings.INDEX_DIR
        self.index_file = self.index_dir / "embeddings.faiss"
        self.meta_file = self.index_dir / "metadata.json"
        self._index = None
        self._metadata: List[Dict[str, Any]] = []
        self._vectors: List[np.ndarray] = []  # In-memory backup for fallback
        self._lock = threading.Lock()
        self._has_faiss = False

        self._init_index()

    def _init_index(self):
        try:
            import faiss
            self._index = faiss.IndexFlatIP(self.dimension)
            self._has_faiss = True
        except Exception as e:
            logger.warning(f"FAISS not available ({e}). Using NumPy fallback vector index.")
            self._has_faiss = False
        self.load()

    def add(self, vector: np.ndarray, frame_id: str, camera_id: str, timestamp: float):
        with self._lock:
            # Ensure 1D float32 normalized vector
            v = vector.astype(np.float32).flatten()
            norm = np.linalg.norm(v)
            if norm > 1e-8:
                v = v / norm

            # Prevent duplicate frame entries
            for m in self._metadata:
                if m["frame_id"] == frame_id:
                    return

            idx_id = len(self._metadata)
            meta_entry = {
                "index_id": idx_id,
                "frame_id": frame_id,
                "camera_id": camera_id,
                "timestamp": timestamp
            }
            self._metadata.append(meta_entry)

            if self._has_faiss and self._index is not None:
                v_2d = np.ascontiguousarray(v.reshape(1, -1), dtype=np.float32)
                self._index.add(v_2d)
            else:
                self._vectors.append(v)

            # Periodically auto-save
            if len(self._metadata) % 10 == 0:
                self.save()

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        camera_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[VectorSearchResult]:
        with self._lock:
            if not self._metadata:
                return []

            q = query_vector.astype(np.float32).flatten()
            norm = np.linalg.norm(q)
            if norm > 1e-8:
                q = q / norm

            results = []

            if self._has_faiss and self._index is not None and self._index.ntotal > 0:
                # Query FAISS
                k_to_fetch = min(len(self._metadata), max(top_k * 3, 50))
                q_2d = np.ascontiguousarray(q.reshape(1, -1), dtype=np.float32)
                distances, indices = self._index.search(q_2d, k_to_fetch)

                for score, idx in zip(distances[0], indices[0]):
                    if idx < 0 or idx >= len(self._metadata):
                        continue
                    m = self._metadata[idx]
                    if camera_id and m["camera_id"] != camera_id:
                        continue
                    if start_time is not None and m["timestamp"] < start_time:
                        continue
                    if end_time is not None and m["timestamp"] > end_time:
                        continue
                    results.append(VectorSearchResult(
                        frame_id=m["frame_id"],
                        camera_id=m["camera_id"],
                        timestamp=m["timestamp"],
                        score=float(score)
                    ))
                    if len(results) >= top_k:
                        break
            else:
                # NumPy fallback cosine search
                if not self._vectors:
                    return []
                mat = np.array(self._vectors, dtype=np.float32)
                scores = np.dot(mat, q)
                sorted_indices = np.argsort(scores)[::-1]

                for idx in sorted_indices:
                    m = self._metadata[idx]
                    if camera_id and m["camera_id"] != camera_id:
                        continue
                    if start_time is not None and m["timestamp"] < start_time:
                        continue
                    if end_time is not None and m["timestamp"] > end_time:
                        continue
                    results.append(VectorSearchResult(
                        frame_id=m["frame_id"],
                        camera_id=m["camera_id"],
                        timestamp=m["timestamp"],
                        score=float(scores[idx])
                    ))
                    if len(results) >= top_k:
                        break

            return results

    def save(self):
        try:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            if self._has_faiss and self._index is not None:
                import faiss
                faiss.write_index(self._index, str(self.index_file))

            with open(self.meta_file, "w", encoding="utf-8") as f:
                json.dump({
                    "metadata": self._metadata,
                    "dimension": self.dimension
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving vector index: {e}")

    def load(self):
        with self._lock:
            if not self.meta_file.exists():
                return
            try:
                with open(self.meta_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._metadata = data.get("metadata", [])
                    self.dimension = data.get("dimension", self.dimension)

                if self._has_faiss and self.index_file.exists():
                    import faiss
                    self._index = faiss.read_index(str(self.index_file))
                logger.info(f"Loaded {len(self._metadata)} vector entries from index.")
            except Exception as e:
                logger.error(f"Error loading vector index: {e}")

vector_index = FaissVectorIndex()

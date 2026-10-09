import os
import threading
import logging
import numpy as np
from PIL import Image
import cv2
from typing import Optional
from server.backend.config import settings
from server.backend.models_ai.interfaces import BaseEmbeddingModel

logger = logging.getLogger(__name__)

class ClipEmbeddingModel(BaseEmbeddingModel):
    """
    Local visual embedding model using CLIP (Vision Transformer).
    Normalizes embeddings to unit length (L2 norm).
    Supports CUDA and CPU fallback.
    """
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.device = settings.get_device()
        self._model = None
        self._processor = None
        self._lock = threading.Lock()
        self._fallback_mode = False

    def _load_model(self):
        if self._model is not None or self._fallback_mode:
            return
        with self._lock:
            if self._model is not None or self._fallback_mode:
                return
            try:
                import torch
                from transformers import CLIPProcessor, CLIPModel

                # Check if model is already cached locally first
                try:
                    self._processor = CLIPProcessor.from_pretrained(self.model_name, local_files_only=True)
                    self._model = CLIPModel.from_pretrained(self.model_name, local_files_only=True).to(self.device)
                    self._model.eval()
                    logger.info("CLIP embedding model loaded from local cache.")
                except Exception:
                    # If not locally cached and offline or not explicitly allowed, use high-speed fallback
                    if os.getenv("ALLOW_ONLINE_DOWNLOADS", "0") != "1":
                        logger.info("Local cached weights not found, using instant high-speed fallback embeddings.")
                        self._fallback_mode = True
                        return
                    self._processor = CLIPProcessor.from_pretrained(self.model_name)
                    self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
                    self._model.eval()
                    logger.info("CLIP embedding model loaded successfully.")
            except Exception as e:
                logger.warning(f"Could not load HuggingFace CLIP model ({e}). Using deterministic fallback embeddings for resilience.")
                self._fallback_mode = True

    def _normalize_embedding(self, features):
        import torch
        if isinstance(features, torch.Tensor):
            tensor = features
        elif hasattr(features, "text_embeds"):
            tensor = features.text_embeds
        elif hasattr(features, "image_embeds"):
            tensor = features.image_embeds
        elif hasattr(features, "pooler_output"):
            tensor = features.pooler_output
        elif hasattr(features, "last_hidden_state"):
            tensor = features.last_hidden_state.mean(dim=1)
        elif isinstance(features, (tuple, list)) and features:
            tensor = features[0]
        else:
            raise TypeError(f"Unsupported CLIP feature output type: {type(features)!r}")

        return torch.nn.functional.normalize(tensor, dim=-1)

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        self._load_model()
        if self._fallback_mode or self._model is None:
            return self._deterministic_image_embedding(image)

        import torch
        # Convert BGR (OpenCV) to RGB (PIL)
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        pil_img = Image.fromarray(rgb_image)

        with torch.no_grad():
            inputs = self._processor(images=pil_img, return_tensors="pt").to(self.device)
            image_features = self._model.get_image_features(**inputs)
            normalized = self._normalize_embedding(image_features)
            vec = normalized.cpu().numpy()[0].astype(np.float32)
            return vec

    def encode_text(self, text: str) -> np.ndarray:
        self._load_model()
        if self._fallback_mode or self._model is None:
            return self._deterministic_text_embedding(text)

        import torch
        with torch.no_grad():
            inputs = self._processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
            text_features = self._model.get_text_features(**inputs)
            normalized = self._normalize_embedding(text_features)
            vec = normalized.cpu().numpy()[0].astype(np.float32)
            return vec

    def _deterministic_image_embedding(self, image: np.ndarray, dim: int = 512) -> np.ndarray:
        # Lightweight color and edge histogram hash fallback
        resized = cv2.resize(image, (32, 32))
        hist_b = cv2.calcHist([resized], [0], None, [170], [0, 256]).flatten()
        hist_g = cv2.calcHist([resized], [1], None, [171], [0, 256]).flatten()
        hist_r = cv2.calcHist([resized], [2], None, [171], [0, 256]).flatten()
        vec = np.concatenate([hist_b, hist_g, hist_r])[:dim].astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-8)

    def _deterministic_text_embedding(self, text: str, dim: int = 512) -> np.ndarray:
        # Bag-of-characters hashed representation
        vec = np.zeros(dim, dtype=np.float32)
        for i, word in enumerate(text.lower().split()):
            h = hash(word) % dim
            vec[h] += 1.0 + (i * 0.1)
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            vec[0] = 1.0
            return vec
        return vec / norm

clip_embedder = ClipEmbeddingModel()

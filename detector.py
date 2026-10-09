import os
import cv2
import threading
import logging
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from PIL import Image

from server.backend.config import settings
from server.backend.models_ai.interfaces import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)

class OwlViTDetector(BaseDetector):
    """
    Open-vocabulary visual object detector using OWL-ViT (Vision Transformer for Open-World Localization).
    Accepts arbitrary text queries (e.g. 'red car', 'person in blue shirt', 'large bag').
    GPU accelerated with CPU fallback.
    """
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.DETECTOR_MODEL_NAME
        self.device = settings.get_device()
        self._model = None
        self._processor = None
        self._lock = threading.Lock()
        self.inference_lock = threading.Lock()
        self._fallback_mode = False

    def _load_model(self):
        if self._model is not None or self._fallback_mode:
            return
        with self._lock:
            if self._model is not None or self._fallback_mode:
                return
            try:
                import torch
                from transformers import OwlViTProcessor, OwlViTForObjectDetection

                try:
                    self._processor = OwlViTProcessor.from_pretrained(self.model_name, local_files_only=True)
                    self._model = OwlViTForObjectDetection.from_pretrained(self.model_name, local_files_only=True).to(self.device)
                    self._model.eval()
                    logger.info("OWL-ViT detector loaded from local cache.")
                except Exception:
                    if os.getenv("ALLOW_ONLINE_DOWNLOADS", "0") != "1":
                        logger.info("Local detector weights not cached, using instant computer vision fallback detector.")
                        self._fallback_mode = True
                        return
                    self._processor = OwlViTProcessor.from_pretrained(self.model_name)
                    self._model = OwlViTForObjectDetection.from_pretrained(self.model_name).to(self.device)
                    self._model.eval()
                    logger.info("OWL-ViT detector loaded successfully.")
            except Exception as e:
                logger.warning(f"Could not load HuggingFace OWL-ViT detector ({e}). Enabling heuristic fallback detector.")
                self._fallback_mode = True

    def detect(self, image: np.ndarray, labels: List[str], threshold: float = 0.15) -> List[DetectionResult]:
        if not labels:
            return []

        self._load_model()
        if self._fallback_mode or self._model is None:
            return self._heuristic_fallback_detect(image, labels, threshold)

        import torch
        # Format image
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        pil_img = Image.fromarray(rgb_image)
        h, w = image.shape[:2]

        clean_labels = [l.strip() for l in labels if l.strip()]
        if not clean_labels:
            return []

        try:
            with self.inference_lock, torch.no_grad():
                inputs = self._processor(text=[clean_labels], images=pil_img, return_tensors="pt").to(self.device)
                outputs = self._model(**inputs)

                target_sizes = torch.Tensor([[h, w]]).to(self.device)
                results = self._processor.post_process_object_detection(
                    outputs=outputs,
                    target_sizes=target_sizes,
                    threshold=threshold
                )[0]

                detections: List[DetectionResult] = []
                boxes = results["boxes"].cpu().numpy()
                scores = results["scores"].cpu().numpy()
                pred_labels = results["labels"].cpu().numpy()

                for box, score, label_idx in zip(boxes, scores, pred_labels):
                    if label_idx < len(clean_labels):
                        label_name = clean_labels[label_idx]
                        x1, y1, x2, y2 = box.tolist()
                        detections.append(DetectionResult(
                            label=label_name,
                            confidence=float(score),
                            box=[round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
                        ))

                return detections
        except Exception as e:
            logger.error(f"OWL-ViT inference error: {e}. Falling back to heuristic detector.")
            return self._heuristic_fallback_detect(image, labels, threshold)

    def _heuristic_fallback_detect(self, image: np.ndarray, labels: List[str], threshold: float = 0.15) -> List[DetectionResult]:
        """
        Deterministic computer vision fallback for testing and offline environments.
        Detects color attributes and basic contours for cars, persons, bags, trucks.
        """
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        detections = []

        for label in labels:
            lbl = label.lower().strip()
            # Color detection heuristics
            mask = None
            if "red" in lbl:
                m1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
                m2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
                mask = cv2.bitwise_or(m1, m2)
            elif "blue" in lbl:
                mask = cv2.inRange(hsv, np.array([100, 70, 50]), np.array([130, 255, 255]))
            elif "white" in lbl:
                mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))
            elif "green" in lbl:
                mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
            elif "black" in lbl:
                mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 60]))

            if mask is not None:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > (w * h * 0.005):  # Object covers at least 0.5% of frame
                        x, y, bw, bh = cv2.boundingRect(cnt)
                        conf = min(0.95, round(0.5 + min(area / (w * h * 0.2), 0.45), 2))
                        if conf >= threshold:
                            detections.append(DetectionResult(
                                label=lbl,
                                confidence=conf,
                                box=[float(x), float(y), float(x + bw), float(y + bh)]
                            ))
            else:
                # Only check edge contours for legitimate surveillance target objects
                known_shapes = ["car", "truck", "person", "bag", "backpack", "vehicle", "bicycle", "van", "automobile"]
                if any(k in lbl for k in known_shapes):
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    edged = cv2.Canny(gray, 50, 150)
                    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:2]:
                        area = cv2.contourArea(cnt)
                        if area > (w * h * 0.02):
                            x, y, bw, bh = cv2.boundingRect(cnt)
                            detections.append(DetectionResult(
                                label=lbl,
                                confidence=0.75,
                                box=[float(x), float(y), float(x + bw), float(y + bh)]
                            ))

        return detections

detector = OwlViTDetector()

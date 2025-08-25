from typing import List
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import cv2


class VideoMAEAnomalyScorer:
    def __init__(self, model_id: str, device: str = "auto", input_size: int = 224):
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageClassification.from_pretrained(model_id)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        self.input_size = input_size
        self.model.eval()

    def _preprocess_clip(self, frames_bgr: List[np.ndarray]) -> torch.Tensor:
        imgs = []
        for f in frames_bgr:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (self.input_size, self.input_size))
            imgs.append(resized)
        inputs = self.processor(images=imgs, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    @torch.no_grad()
    def score_clip(self, frames_bgr: List[np.ndarray]) -> float:
        inputs = self._preprocess_clip(frames_bgr)
        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        # Heuristic: if model is fine-tuned for normal vs abnormal or multi-class crimes,
        # take the highest non-normal probability as anomaly score when labels are available.
        score = float(1.0 - probs.max().item())
        return score



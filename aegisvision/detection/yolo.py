from typing import List, Tuple, Dict, Any
import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover - optional import for docs build
    YOLO = None  # type: ignore


class YOLODetector:
    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.25, iou: float = 0.5, device: str = "auto"):
        if YOLO is None:
            raise ImportError("Ultralytics YOLO is not installed. Please install 'ultralytics'.")
        self.model = YOLO(model_name)
        if device != "auto":
            self.model.to(device)
        self.conf = conf
        self.iou = iou

    def detect(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Run detection on a single BGR frame. Returns list of dicts with bbox, cls, conf.

        bbox format: (x1, y1, x2, y2)
        """
        results = self.model.predict(image_bgr, conf=self.conf, iou=self.iou, verbose=False)[0]
        detections: List[Dict[str, Any]] = []
        for box, cls, conf in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy(), results.boxes.conf.cpu().numpy()):
            detections.append({
                "bbox": tuple(map(float, box.tolist())),
                "class_id": int(cls),
                "confidence": float(conf),
            })
        return detections



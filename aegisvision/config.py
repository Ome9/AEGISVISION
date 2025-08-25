from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DetectionConfig:
    model_name: str = "yolov8n.pt"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.5
    device: str = "auto"  # "cuda", "cpu", or "auto"


@dataclass
class VideoMAEConfig:
    model_id: str = "OPear/videomae-large-finetuned-UCF-Crime"
    sample_rate: int = 4           # use every Nth frame
    clip_len: int = 16             # number of frames per clip
    input_size: int = 224          # resize to square
    anomaly_threshold: float = 0.7 # score to trigger alert
    device: str = "auto"


@dataclass
class HeuristicsConfig:
    loiter_seconds: float = 30.0
    abandonment_seconds: float = 20.0
    stationary_pixel_tolerance: int = 10
    proximity_pixels: int = 50
    fps_hint: Optional[float] = None  # if None, inferred from video


@dataclass
class AppConfig:
    output_dir: str = "output"
    alerts_json: str = "alerts.json"
    annotated_video: str = "annotated.mp4"


@dataclass
class Config:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    videomae: VideoMAEConfig = field(default_factory=VideoMAEConfig)
    heuristics: HeuristicsConfig = field(default_factory=HeuristicsConfig)
    app: AppConfig = field(default_factory=AppConfig)



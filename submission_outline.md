### I. Proposed Solution

An AI-powered surveillance system that combines frame-level object/person detection (YOLO) with video-level anomaly understanding (VideoMAE). We track entities over time and apply heuristics for loitering and object abandonment, while VideoMAE flags unusual movement. Outputs include an annotated video and a timestamped alert log, viewable in a Streamlit dashboard.

Innovation: lightweight tracking + heuristics provide interpretable signals (loitering/abandonment), while a pretrained video transformer (VideoMAE) complements with learned anomaly cues from temporal context. Synthetic data (bonus) can be generated for rare edge cases to improve robustness.

### II. Technical Approach

- Technologies: Python, OpenCV, PyTorch, Transformers, Ultralytics YOLO, Streamlit
- Flow:
  1. Ingest video → sample frames
  2. YOLO detections → tracking → dwell/stationary estimation
  3. Clip sampling → VideoMAE anomaly score
  4. Heuristics + anomaly threshold → alerts with timestamps
  5. Annotated video + JSON alert log → dashboard visualization

### III. Feasibility and Viability

- Feasibility: Uses widely-available pretrained models; runs on CPU (slower) or GPU (preferred). Modular design supports different environments and scaling.
- Risks: Domain shift to new scenes, poor lighting, crowded scenes, fast motion blur.
- Mitigations: Threshold tuning, finetuning VideoMAE on Avenue/UCSD, data augmentation, synthetic video for rare cases.

### IV. Research and References

- YOLO object detection (Ultralytics)
- VideoMAE: Masked Autoencoders Are Data-Efficient Learners for Video
- UCF-Crime: Large-scale real-world anomaly detection dataset
- Avenue Dataset, UCSD Anomaly Detection Dataset (for finetuning/eval)


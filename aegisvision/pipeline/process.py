from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from aegisvision.config import Config
from aegisvision.detection.yolo import YOLODetector
from aegisvision.models.videomae import VideoMAEAnomalyScorer
from aegisvision.tracking.tracker import SimpleTracker
from aegisvision.analytics.anomaly import evaluate_heuristics, Alert


CLASS_PERSON = 0  # COCO
CLASS_BAG = 24    # COCO: backpack
CLASS_SUITCASE = 28


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def process_video(input_video: str, cfg: Config, output_dir: str | None = None) -> Path:
    out_dir = Path(output_dir or cfg.app.output_dir)
    ensure_dir(out_dir)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    annotated_path = out_dir / cfg.app.annotated_video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, (width, height))

    detector = YOLODetector(
        model_name=cfg.detection.model_name,
        conf=cfg.detection.conf_threshold,
        iou=cfg.detection.iou_threshold,
        device=cfg.detection.device,
    )
    scorer = VideoMAEAnomalyScorer(
        model_id=cfg.videomae.model_id,
        device=cfg.videomae.device,
        input_size=cfg.videomae.input_size,
    )
    person_tracker = SimpleTracker()
    object_tracker = SimpleTracker()

    frame_idx = 0
    clip_buffer: List[np.ndarray] = []
    alerts: List[Alert] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        detections = detector.detect(frame)
        person_boxes: List[Tuple[float, float, float, float]] = []
        object_boxes: List[Tuple[float, float, float, float]] = []
        for det in detections:
            cid = det["class_id"]
            if cid == CLASS_PERSON:
                person_boxes.append(det["bbox"])
            elif cid in (CLASS_BAG, CLASS_SUITCASE):
                object_boxes.append(det["bbox"])

        p_tracks = person_tracker.update(frame_idx, person_boxes)
        o_tracks = object_tracker.update(frame_idx, object_boxes)

        # Heuristics
        alerts.extend(
            evaluate_heuristics(
                frame_idx=frame_idx,
                fps=cfg.heuristics.fps_hint or fps,
                person_tracks=p_tracks,
                object_tracks=o_tracks,
                loiter_seconds=cfg.heuristics.loiter_seconds,
                abandonment_seconds=cfg.heuristics.abandonment_seconds,
                stationary_pixel_tolerance=cfg.heuristics.stationary_pixel_tolerance,
            )
        )

        # VideoMAE scoring on sampled frames
        if frame_idx % cfg.videomae.sample_rate == 0:
            clip_buffer.append(frame.copy())
            if len(clip_buffer) >= cfg.videomae.clip_len:
                score = scorer.score_clip(clip_buffer[-cfg.videomae.clip_len :])
                if score >= cfg.videomae.anomaly_threshold:
                    alerts.append(Alert(frame_idx=frame_idx, timestamp_s=frame_idx / fps, type="unusual_movement", track_id=None, score=score))

        # Draw annotations
        for tid, tr in p_tracks.items():
            x1, y1, x2, y2 = map(int, tr.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"person#{tid}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        for tid, tr in o_tracks.items():
            x1, y1, x2, y2 = map(int, tr.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.putText(frame, f"object#{tid}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)

        for al in alerts[-5:]:  # show recent alerts
            if abs(al.frame_idx - frame_idx) < fps * 2:
                cv2.putText(frame, f"ALERT: {al.type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        writer.write(frame)

    cap.release()
    writer.release()

    # Save alerts JSON
    alerts_path = out_dir / cfg.app.alerts_json
    with open(alerts_path, "w", encoding="utf-8") as f:
        json.dump([asdict(a) for a in alerts], f, indent=2)

    return annotated_path



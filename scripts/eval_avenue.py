"""
Evaluate VideoMAE on the Avenue dataset (frame-level anomaly scoring to ROC-AUC).
Assumes a directory with video files and a matching ground-truth .m or .npy with frame-level labels.
This script sketches a typical approach; adapt paths and GT loader to your setup.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import cv2
from transformers import AutoImageProcessor, VideoMAEForVideoClassification
from sklearn.metrics import roc_auc_score


def load_gt_labels(gt_path: Path) -> dict:
    """Load ground-truth frame labels per video into a dict: {video_stem: [0/1,...]}
    Implement according to your GT format. Here supports .npy maps by video stem.
    """
    if gt_path.is_file() and gt_path.suffix == ".npy":
        return np.load(gt_path, allow_pickle=True).item()
    raise ValueError("Please provide a .npy mapping file with frame-level labels per video.")


def sample_frames_for_clip(cap, frame_indices, size=(224, 224)):
    imgs = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for idx in frame_indices:
        idx = max(0, min(total - 1, idx))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, size)
        imgs.append(frame)
    return imgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_dir", required=True)
    parser.add_argument("--gt_map", required=True, help=".npy dict: {video_stem: [0/1,...]}")
    parser.add_argument("--model_id", default="OPear/videomae-large-finetuned-UCF-Crime")
    parser.add_argument("--clip_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = VideoMAEForVideoClassification.from_pretrained(args.model_id).to(device)
    model.eval()

    gt_map = load_gt_labels(Path(args.gt_map))
    print(f"Loaded {len(gt_map)} ground-truth entries from {args.gt_map}")

    video_files = []
    for ext in ["*.avi", "*.mp4", "*.mov"]:
        video_files.extend(Path(args.videos_dir).glob(ext))
    print(f"Found {len(video_files)} videos in {args.videos_dir}")

    all_scores = []
    all_labels = []

    for vid in sorted(video_files):
        stem = vid.stem
        if stem not in gt_map:
            print(f"Warning: No ground-truth for video {vid.name}, skipping.")
            continue
        labels = np.array(gt_map[stem], dtype=int)
        cap = cv2.VideoCapture(str(vid))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_scores = np.zeros(total, dtype=float)
        frame_counts = np.zeros(total, dtype=int)

        for start in range(0, total - args.clip_len + 1, args.stride):
            idxs = list(range(start, start + args.clip_len))
            imgs = sample_frames_for_clip(cap, idxs)
            if len(imgs) < args.clip_len:
                break
            inputs = processor(images=imgs, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                # Use 1 - max prob as anomaly score per clip
                score = float(1.0 - probs.max().item())
            for i in idxs:
                frame_scores[i] += score
                frame_counts[i] += 1

        cap.release()
        # Average overlapping scores
        nonzero = frame_counts > 0
        frame_scores[nonzero] /= frame_counts[nonzero]
        valid_len = min(len(labels), len(frame_scores))
        all_scores.extend(frame_scores[:valid_len].tolist())
        all_labels.extend(labels[:valid_len].tolist())

    if not all_labels or not all_scores:
        print("No samples processed. Check video paths, ground-truth map, and file formats.")
        return

    auc = roc_auc_score(all_labels, all_scores)
    print(f"Avenue frame-level ROC-AUC: {auc:.4f}")


if __name__ == "__main__":
    main()



"""
Evaluate VideoMAE on UCSD frame folders using frame-level labels.

Inputs:
- --test_root: path containing Test001, Test002, ... subfolders with frame images (.tif/.png/.jpg)
- --gt_map: .npy dict mapping {"TestNNN": [0/1,...]} frame-level labels

Outputs:
- Prints frame-level ROC-AUC across all sequences
"""

import argparse
import re
from pathlib import Path
from typing import List

import numpy as np
import torch
import cv2
from transformers import AutoImageProcessor, VideoMAEForVideoClassification
from sklearn.metrics import roc_auc_score


def natural_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def load_images(seq_dir: Path, size=(224, 224)) -> List[np.ndarray]:
    frames = []
    imgs = sorted([p for p in seq_dir.iterdir() if p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}], key=lambda p: natural_key(p.name))
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        frames.append(img)
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_root", required=True)
    parser.add_argument("--gt_map", required=True)
    parser.add_argument("--model_id", default="OPear/videomae-large-finetuned-UCF-Crime")
    parser.add_argument("--clip_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = VideoMAEForVideoClassification.from_pretrained(args.model_id).to(device)
    model.eval()

    gt_map = np.load(args.gt_map, allow_pickle=True).item()

    all_scores: List[float] = []
    all_labels: List[int] = []

    test_root = Path(args.test_root)
    seq_dirs = sorted([p for p in test_root.iterdir() if p.is_dir() and p.name.lower().startswith("test")], key=lambda p: natural_key(p.name))

    for seq in seq_dirs:
        stem = seq.name
        if stem not in gt_map:
            continue
        labels = np.array(gt_map[stem], dtype=int)
        frames = load_images(seq)
        total = len(frames)
        if total == 0:
            continue
        scores = np.zeros(total, dtype=float)
        counts = np.zeros(total, dtype=int)

        for start in range(0, max(1, total - args.clip_len + 1), args.stride):
            clip = frames[start : start + args.clip_len]
            if len(clip) < args.clip_len:
                break
            inputs = processor(images=clip, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
                probs = torch.softmax(out.logits, dim=-1)
                score = float(1.0 - probs.max().item())
            for i in range(start, start + args.clip_len):
                scores[i] += score
                counts[i] += 1

        nonzero = counts > 0
        scores[nonzero] /= counts[nonzero]
        valid_len = min(len(labels), len(scores))
        all_scores.extend(scores[:valid_len].tolist())
        all_labels.extend(labels[:valid_len].tolist())

    auc = roc_auc_score(all_labels, all_scores)
    print(f"UCSD frame-level ROC-AUC: {auc:.4f}")


if __name__ == "__main__":
    main()



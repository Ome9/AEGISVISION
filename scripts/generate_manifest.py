"""
Generate manifest files (train.jsonl, val.jsonl) for VideoMAE fine-tuning from Avenue and UCSD .npy mappings.
Each line: {"video_path": ..., "label": ...} or {"frame_path": ..., "label": ...}
"""

import argparse
import numpy as np
from pathlib import Path
import json

def create_avenue_manifest(npy_path, video_dir, output_jsonl):
    mapping = np.load(npy_path, allow_pickle=True).item()
    entries = []
    for stem, labels in mapping.items():
        video_path = str(Path(video_dir) / f"{stem}.avi")
        label = int(any(labels))
        entries.append({"video_path": video_path, "label": label})
    return entries


def create_ucsd_manifest(npy_path, frame_root, output_jsonl):
    mapping = np.load(npy_path, allow_pickle=True).item()
    entries = []
    for clip, labels in mapping.items():
        for i, label in enumerate(labels, start=1):
            frame_path = str(Path(frame_root) / clip / f"{i:03d}.tif")
            entries.append({"frame_path": frame_path, "label": int(label)})
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--avenue_train_npy', help='Avenue training .npy')
    parser.add_argument('--avenue_train_videos', help='Avenue training videos dir')
    parser.add_argument('--avenue_test_npy', help='Avenue test .npy')
    parser.add_argument('--avenue_test_videos', help='Avenue test videos dir')
    parser.add_argument('--ucsd_train_npy', nargs='*', help='UCSD training .npy files (multiple allowed)')
    parser.add_argument('--ucsd_train_root', nargs='*', help='UCSD training frames roots (multiple allowed)')
    parser.add_argument('--ucsd_test_npy', nargs='*', help='UCSD test .npy files (multiple allowed)')
    parser.add_argument('--ucsd_test_root', nargs='*', help='UCSD test frames roots (multiple allowed)')
    parser.add_argument('--output_train', default='train.jsonl')
    parser.add_argument('--output_val', default='val.jsonl')
    args = parser.parse_args()


    train_entries = []
    val_entries = []

    # Avenue train
    if args.avenue_train_npy and args.avenue_train_videos:
        train_entries.extend(create_avenue_manifest(args.avenue_train_npy, args.avenue_train_videos, args.output_train))
    # Avenue test
    if args.avenue_test_npy and args.avenue_test_videos:
        val_entries.extend(create_avenue_manifest(args.avenue_test_npy, args.avenue_test_videos, args.output_val))

    # UCSD train (multiple)
    if args.ucsd_train_npy and args.ucsd_train_root:
        for npy_path, root in zip(args.ucsd_train_npy, args.ucsd_train_root):
            train_entries.extend(create_ucsd_manifest(npy_path, root, args.output_train))
    # UCSD test (multiple)
    if args.ucsd_test_npy and args.ucsd_test_root:
        for npy_path, root in zip(args.ucsd_test_npy, args.ucsd_test_root):
            val_entries.extend(create_ucsd_manifest(npy_path, root, args.output_val))

    # Write all entries to output files
    with open(args.output_train, 'w', encoding='utf-8') as f:
        for entry in train_entries:
            f.write(json.dumps(entry) + '\n')
    with open(args.output_val, 'w', encoding='utf-8') as f:
        for entry in val_entries:
            f.write(json.dumps(entry) + '\n')

if __name__ == '__main__':
    main()

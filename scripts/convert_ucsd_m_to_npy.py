import os
import numpy as np
import cv2
from pathlib import Path

def process_train(train_dir, output_file, img_size=(224,224)):
    mapping = {}
    for vid_folder in sorted(Path(train_dir).iterdir()):
        if not vid_folder.is_dir():
            continue
        frames = sorted(vid_folder.glob("*.tif"))
        mapping[vid_folder.name] = {
            "frames": [str(f) for f in frames],
            "labels": np.zeros(len(frames), dtype=int).tolist()
        }
        print(f"[TRAIN] {vid_folder.name}: {len(frames)} frames, labels all zeros")
    np.save(output_file, mapping, allow_pickle=True)
    print(f"Saved training map: {output_file}")


def process_test(test_dir, output_file, img_size=(224,224)):
    mapping = {}
    test_folders = sorted([p for p in Path(test_dir).iterdir() if p.is_dir()])
    
    for vid_folder in test_folders:
        if vid_folder.name.endswith("_gt"):
            # skip gt folders
            continue
        
        frames = sorted(vid_folder.glob("*.tif"))
        labels = np.zeros(len(frames), dtype=int)

        # check for corresponding _gt folder
        gt_folder = vid_folder.parent / f"{vid_folder.name}_gt"
        if gt_folder.exists():
            gt_frames = sorted(gt_folder.glob("*.bmp"))
            # Map frames one-to-one
            for i, gt_f in enumerate(gt_frames):
                if i >= len(labels):
                    break
                mask = cv2.imread(str(gt_f), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                if np.any(mask > 0):
                    labels[i] = 1

        mapping[vid_folder.name] = {
            "frames": [str(f) for f in frames],
            "labels": labels.tolist()
        }
        print(f"[TEST] {vid_folder.name}: {len(frames)} frames, {labels.sum()} anomalies")

    np.save(output_file, mapping, allow_pickle=True)
    print(f"Saved testing map: {output_file}")


if __name__ == "__main__":
    # Ped1
    ped1_train_dir = "AegisVision Dataset/UCSD_Anomaly_Dataset/UCSDped1/Train"
    ped1_test_dir  = "AegisVision Dataset/UCSD_Anomaly_Dataset/UCSDped1/Test"
    process_train(ped1_train_dir, "ucsd_ped1_train_map.npy")
    process_test(ped1_test_dir, "ucsd_ped1_test_map.npy")

    # Ped2
    ped2_train_dir = "AegisVision Dataset/UCSD_Anomaly_Dataset/UCSDped2/Train"
    ped2_test_dir  = "AegisVision Dataset/UCSD_Anomaly_Dataset/UCSDped2/Test"
    process_train(ped2_train_dir, "ucsd_ped2_train_map.npy")
    process_test(ped2_test_dir, "ucsd_ped2_test_map.npy")

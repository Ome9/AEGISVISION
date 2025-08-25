import os
import json
import numpy as np
import cv2
from pathlib import Path

# --- Parameters ---
N_FRAMES = 16   # clip length
STRIDE = 8      # frame stride
TEMP_DIR = "temp_frames"  # temporary folder to save images for manifest
OUTPUT_TRAIN = "combined_train.jsonl"
OUTPUT_TEST = "combined_test.jsonl"

# --- Helper functions ---
def save_clip_frames(frames, base_name):
    """
    Save frames as images to TEMP_DIR and return list of paths.
    Handles frames as np.ndarray or already as file paths (str).
    """
    frame_paths = []
    for idx, frame in enumerate(frames):
        # If the frame is already a string (path), just use it
        if isinstance(frame, str):
            frame_paths.append(frame)
            continue

        out_path = Path(TEMP_DIR) / f"{base_name}_{idx:04d}.jpg"
        os.makedirs(out_path.parent, exist_ok=True)

        # Convert float32 0-1 to uint8 0-255 if needed
        if frame.dtype != np.uint8:
            frame_uint8 = (frame * 255).astype(np.uint8)
        else:
            frame_uint8 = frame
        cv2.imwrite(str(out_path), frame_uint8)
        frame_paths.append(str(out_path))
    return frame_paths

def npy_to_clips(npy_file, clip_len=N_FRAMES, stride=STRIDE):
    """
    Convert a single npy file to a list of clips for JSONL.
    Each clip is {"frames": [...], "label": 0 or 1}.
    """
    data = np.load(npy_file, allow_pickle=True).item()
    out_clips = []

    for vid_name, content in data.items():
        frames = content["frames"]
        labels = content["labels"]
        total_frames = len(frames)
        for start in range(0, total_frames - clip_len + 1, stride):
            clip_frames = frames[start:start + clip_len]
            clip_labels = labels[start:start + clip_len]
            # Label for the clip: 1 if any frame is anomalous, else 0
            clip_label = int(np.any(clip_labels))
            frame_paths = save_clip_frames(clip_frames, f"{vid_name}_{start}")
            out_clips.append({"frames": frame_paths, "label": clip_label})
    return out_clips

# --- Main processing ---
def main():
    npy_files = list(Path(".").glob("*.npy"))
    train_clips = []
    test_clips = []

    for npy_file in npy_files:
        fname = npy_file.name.lower()
        print(f"Processing {npy_file} ...")
        clips = npy_to_clips(npy_file)
        if "train" in fname:
            train_clips.extend(clips)
        elif "test" in fname:
            test_clips.extend(clips)

    # Save combined manifests
    with open(OUTPUT_TRAIN, "w") as f:
        for clip in train_clips:
            f.write(json.dumps(clip) + "\n")
    with open(OUTPUT_TEST, "w") as f:
        for clip in test_clips:
            f.write(json.dumps(clip) + "\n")

    print(f"Saved {len(train_clips)} train clips to {OUTPUT_TRAIN}")
    print(f"Saved {len(test_clips)} test clips to {OUTPUT_TEST}")

if __name__ == "__main__":
    main()

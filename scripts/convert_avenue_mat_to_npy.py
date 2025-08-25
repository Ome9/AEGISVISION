import os
from pathlib import Path
import numpy as np
import cv2
from scipy.io import loadmat

IMG_SIZE = (224, 224)  # resize frames for consistency

def load_video_frames(video_path: Path) -> np.ndarray:
    """Load video frames, resize, normalize to [0,1]."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, IMG_SIZE)
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
    cap.release()
    return np.array(frames, dtype=np.float32)

def mat_to_frame_labels(mat_file: Path) -> np.ndarray:
    """Convert a .mat file (volLabel / vol / gt) into 1D frame-level 0/1 labels."""
    data = loadmat(mat_file)

    if "volLabel" in data:
        vol = np.array(data["volLabel"])
        if vol.dtype == object:
            vol = np.stack([np.array(v).squeeze() for v in vol], axis=-1)
        if vol.ndim == 3:
            labels = (vol.max(axis=(0,1)) > 0).astype(np.uint8)
        else:
            raise ValueError(f"Unexpected volLabel shape: {vol.shape}")
    elif "vol" in data:
        vol = np.array(data["vol"])
        if vol.ndim == 3:
            labels = (vol.max(axis=(0,1)) > 0).astype(np.uint8)
        else:
            raise ValueError(f"Unexpected vol shape: {vol.shape}")
    elif "gt" in data:
        labels = data["gt"].squeeze()
        if labels.ndim > 1:
            labels = labels.max(axis=0)
        labels = labels.astype(np.uint8)
    else:
        raise ValueError(f"No supported keys in {mat_file}")
    return labels

def build_train_map(videos_dir: str, output_file: str):
    """Build avenue_train_map.npy with frames + all-zero labels."""
    videos_dir = Path(videos_dir)
    video_files = sorted(videos_dir.glob("*.*"))
    mapping = {}

    for vid in video_files:
        if vid.suffix.lower() not in {".avi", ".mp4", ".mov", ".mkv"}:
            continue
        frames = load_video_frames(vid)
        labels = np.zeros(len(frames), dtype=np.uint8)
        mapping[vid.stem] = {"frames": frames, "labels": labels}
        print(f"[TRAIN] {vid.stem}: {len(frames)} frames, labels all zeros")

    np.save(output_file, mapping)
    print(f"\nSaved training map: {output_file}")

def build_test_map(videos_dir: str, mat_dir: str, output_file: str):
    """Build avenue_test_map.npy with frames + labels from .mat files."""
    videos_dir = Path(videos_dir)
    mat_dir = Path(mat_dir)
    video_files = sorted([p for p in videos_dir.glob("*.*") if p.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}])
    mat_files = sorted(mat_dir.glob("*.mat"))

    mapping = {}
    for vid, mat_file in zip(video_files, mat_files):
        frames = load_video_frames(vid)
        labels = mat_to_frame_labels(mat_file)
        # Ensure frames and labels match
        min_len = min(len(frames), len(labels))
        frames = frames[:min_len]
        labels = labels[:min_len]
        mapping[vid.stem] = {"frames": frames, "labels": labels}
        print(f"[TEST] {vid.stem}: {len(frames)} frames, {labels.sum()} anomalies")

    np.save(output_file, mapping)
    print(f"\nSaved testing map: {output_file}")

if __name__ == "__main__":
    BASE_DIR = "AegisVision Dataset\Avenue_Dataset"

    # 1Training map
    build_train_map(
        videos_dir=os.path.join(BASE_DIR, "training_videos"),
        output_file=os.path.join(BASE_DIR, "avenue_train_map.npy")
    )

    # Testing map
    build_test_map(
        videos_dir=os.path.join(BASE_DIR, "testing_videos"),
        mat_dir=os.path.join(BASE_DIR, "testing_vol"),
        output_file=os.path.join(BASE_DIR, "avenue_test_map.npy")
    )

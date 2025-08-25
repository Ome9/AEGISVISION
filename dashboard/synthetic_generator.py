import torch
import cv2
import numpy as np
from pathlib import Path
from RIFE_HDv3 import Model
import tempfile
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load RIFE model once
rife_model = Model()
rife_model.load_model("flownet.pkl")  # path to your pretrained weights
rife_model.eval()
rife_model.device()

def generate_synthetic_video(input_path, output_path, factor=2):
    """
    Generate enhanced synthetic video using RIFE frame interpolation.
    Returns: frames (list of frames), alerts (empty list for compatibility)
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps * factor, (w, h))
    frames = []
    alerts = []  # keep for compatibility

    ret, frame1 = cap.read()
    if not ret:
        return [], []

    frame1_t = torch.from_numpy(frame1).permute(2,0,1).unsqueeze(0).float().to(device)/255.

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
        frame2_t = torch.from_numpy(frame2).permute(2,0,1).unsqueeze(0).float().to(device)/255.

        # RIFE inference to generate middle frame
        mid = rife_model.inference(frame1_t, frame2_t)

        # Properly remove batch dim and convert to HWC for OpenCV
        mid = mid.clamp(0,1).detach().cpu().squeeze(0).permute(1,2,0).numpy()
        mid = (mid * 255).astype(np.uint8)

        out.write(frame1)
        out.write(mid)
        frames.append(frame1)
        frames.append(mid)

        frame1 = frame2
        frame1_t = frame2_t

    out.write(frame1)
    frames.append(frame1)
    cap.release()
    out.release()

    return frames, alerts

import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, VideoMAEForVideoClassification

# Define video directory
video_folder = "sample_data"

# Define class mapping
class_mapping = {
    "Abuse": 0, "Arrest": 1, "Arson": 2, "Assault": 3, "Burglary": 4,
    "Explosion": 5, "Fighting": 6, "Normal Videos": 7, "Road Accidents": 8,
    "Robbery": 9, "Shooting": 10, "Shoplifting": 11, "Stealing": 12, "Vandalism": 13
}
reverse_mapping = {v: k for k, v in class_mapping.items()}

# Load VideoMAE model and processor
model_name = "OPear/videomae-large-finetuned-UCF-Crime"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoImageProcessor.from_pretrained(model_name)
model = VideoMAEForVideoClassification.from_pretrained(
    model_name,
    label2id=class_mapping,
    id2label=reverse_mapping,
    ignore_mismatched_sizes=True,
).to(device)
model.eval()

# Video processing function
def load_video_frames(video_path, num_frames=16, size=(224, 224)):
    """
    Load video frames from a given path and resize them to (224, 224).
    Converts video into a tensor of shape [num_frames, 3, height, width].
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, size)
            frames.append(frame)
    
    cap.release()

    if len(frames) < num_frames:  # Pad if not enough frames
        frames.extend([frames[-1]] * (num_frames - len(frames)))

    frames = np.stack(frames, axis=0)  # [num_frames, H, W, 3] RGB uint8
    return frames

# Custom Dataset
class VideoDataset(Dataset):
    def __init__(self, video_folder: str, num_frames: int = 16, size=(224, 224)):
        self.video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.lower().endswith((".mp4", ".avi", ".mov"))]
        self.num_frames = num_frames
        self.size = size
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        frames = load_video_frames(video_path, num_frames=self.num_frames, size=self.size)
        return {"frames": frames, "filename": os.path.basename(video_path)}

# Load dataset
def main():
    global video_folder
    test_dataset = VideoDataset(video_folder)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Run inference
    with torch.no_grad():
        for idx, sample in enumerate(test_loader):
            frames = sample["frames"][0].numpy()  # [T, H, W, 3] RGB
            inputs = processor(images=[frame for frame in frames], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_label = torch.argmax(probs, dim=-1).item()
            filename = sample["filename"][0]
            print(f"Video {idx}: {filename} - Predicted label = {reverse_mapping[predicted_label]}")


if __name__ == "__main__":
    main()
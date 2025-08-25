"""
Evaluate the trained VideoMAE model
"""

import json
import torch
import cv2
import numpy as np
from pathlib import Path
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

class VideoDataset:
    def __init__(self, jsonl_path, processor, input_size=224):
        self.processor = processor
        self.input_size = input_size
        self.samples = []
        
        # Load samples
        for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
            sample = json.loads(line)
            self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and preprocess frames
        imgs = []
        for frame_path in sample["frames"]:
            img = cv2.imread(frame_path)
            if img is None:
                print(f"Warning: Could not load {frame_path}")
                # Create a black frame as fallback
                img = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.input_size, self.input_size))
            imgs.append(img)
        
        # Process with VideoMAE processor
        inputs = self.processor(images=imgs, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        return inputs, sample["label"]

def evaluate_model(model_path, test_jsonl, device="cuda"):
    """Evaluate model on test set"""
    
    # Load model and processor
    print(f"Loading model from {model_path}...")
    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    model = VideoMAEForVideoClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Load test dataset
    print(f"Loading test data from {test_jsonl}...")
    test_dataset = VideoDataset(test_jsonl, processor)
    
    predictions = []
    true_labels = []
    
    print("Running evaluation...")
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="Evaluating"):
            inputs, true_label = test_dataset[i]
            
            # Move inputs to device
            pixel_values = inputs["pixel_values"].unsqueeze(0).to(device)
            
            # Get prediction
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            predicted_label = torch.argmax(logits, dim=-1).cpu().item()
            
            predictions.append(predicted_label)
            true_labels.append(true_label)
    
    # Calculate metrics
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    # Overall accuracy
    accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Detailed classification report
    print("\nClassification Report:")
    report = classification_report(true_labels, predictions, 
                                 target_names=["Normal", "Anomaly"],
                                 digits=4)
    print(report)
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, predictions)
    print(f"           Predicted")
    print(f"         Normal Anomaly")
    print(f"Normal     {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"Anomaly    {cm[1,0]:4d}   {cm[1,1]:4d}")
    
    # Per-class metrics
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], cm[0,1], cm[1,0], cm[1,1])
    
    precision_normal = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_normal = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision_anomaly = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_anomaly = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\nDetailed Metrics:")
    print(f"Normal Detection - Precision: {precision_normal:.4f}, Recall: {recall_normal:.4f}")
    print(f"Anomaly Detection - Precision: {precision_anomaly:.4f}, Recall: {recall_anomaly:.4f}")
    
    return {
        "accuracy": accuracy,
        "predictions": predictions,
        "true_labels": true_labels,
        "confusion_matrix": cm
    }

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--test_data", required=True, help="Path to test JSONL file")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    results = evaluate_model(args.model_path, args.test_data, device)
    
    print(f"\n✅ Evaluation complete!")
    print(f"Final accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")

if __name__ == "__main__":
    main()
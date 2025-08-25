"""
Optimized VideoMAE finetuning for Avenue + UCSD datasets
- Combines multiple JSONL manifests
- Shuffles clips
- Resumes from last checkpoint folder
- Memory-efficient for 16GB RAM + 4GB GPU
- Shows progress bar for training
"""

from dataclasses import dataclass
from typing import List
from pathlib import Path
import json
import torch
from torch.utils.data import Dataset
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import cv2
import random
from tqdm import tqdm

@dataclass
class Sample:
    frames: List[str]
    label: int

class ClipDataset(Dataset):
    def __init__(self, samples: List[Sample], processor, input_size: int = 224):
        self.samples = samples
        self.processor = processor
        self.input_size = input_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        imgs = []
        for p in sample.frames:
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.input_size, self.input_size))
            imgs.append(img)
        inputs = self.processor(images=imgs, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = torch.tensor(sample.label, dtype=torch.long)
        return inputs

def load_samples_from_jsonls(jsonl_paths: List[Path]) -> List[Sample]:
    samples = []
    for p in jsonl_paths:
        print(f"Processing manifest: {p}")
        for line in tqdm(p.read_text(encoding="utf-8").splitlines(), desc="Reading samples", leave=False):
            js = json.loads(line)
            samples.append(Sample(frames=js["frames"], label=int(js["label"])))
    random.shuffle(samples)
    return samples

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="MCG-NJU/videomae-base")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--train_manifests", nargs="+", required=True)
    parser.add_argument("--val_manifests", nargs="+", required=True)
    parser.add_argument("--output_dir", default="checkpoints/videomae_finetuned")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accum_steps", type=int, default=2)
    parser.add_argument("--resume_checkpoint", default=None, help="Path to last checkpoint folder")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    print(f"Using device: {device}")

    processor = VideoMAEImageProcessor.from_pretrained(args.model_id)
    model = VideoMAEForVideoClassification.from_pretrained(args.model_id)
    model.to(device)

    # Load and combine samples
    train_ds = ClipDataset(load_samples_from_jsonls([Path(p) for p in args.train_manifests]), processor)
    val_ds = ClipDataset(load_samples_from_jsonls([Path(p) for p in args.val_manifests]), processor)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum_steps,
        num_train_epochs=args.epochs,
        fp16=True,
        logging_steps=10,
        report_to=[],
        save_strategy="epoch",
        save_total_limit=3,
        dataloader_num_workers=4,
        eval_strategy="epoch",
        load_best_model_at_end=True,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        # Regularization to prevent overfitting
        learning_rate=1e-5,  # Much lower learning rate
        weight_decay=0.01,   # L2 regularization
        warmup_steps=100,    # Shorter warmup for smaller dataset
        lr_scheduler_type="cosine",  # Better LR scheduling
        # Model selection
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
    )

    def compute_metrics(eval_pred):
        import numpy as np
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = (preds == labels).mean()
        return {"accuracy": acc}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=processor,  # Updated from tokenizer
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    if args.resume_checkpoint:
        print(f"Resuming from checkpoint folder: {args.resume_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_checkpoint)
    else:
        print("Starting training from scratch...")
        trainer.train()

    trainer.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
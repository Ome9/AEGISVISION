import numpy as np

train_map = np.load("AegisVision Dataset/Avenue_Dataset/avenue_train_map.npy", allow_pickle=True).item()
test_map = np.load("AegisVision Dataset/Avenue_Dataset/avenue_test_map.npy", allow_pickle=True).item()

print("Training videos:", list(train_map.keys()))
print("Testing videos:", list(test_map.keys()))

# Example: check a single video
vid = "01"
print("Train frames:", train_map[vid]["frames"].shape, "Labels sum:", train_map[vid]["labels"].sum())
print("Test frames:", test_map[vid]["frames"].shape, "Labels sum:", test_map[vid]["labels"].sum())
# Check total anomalies in test set
total_anomalies = sum(v["labels"].sum() for v in test_map.values())
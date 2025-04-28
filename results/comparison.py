import json
import matplotlib.pyplot as plt

# File paths (update if needed)
files = {
    "DETR": "evaluation_results_detr_predictions.json",
    "YOLOv11n": "evaluation_results_yolo11n_predictions_anno_only.json",
    "YOLOv8n": "evaluation_results_yolov8n_predictions_anno_only.json"
}

# AP keys to compare
ap_keys = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]

# Load AP values
ap_data = {}
for label, path in files.items():
    with open(path, "r") as f:
        data = json.load(f)
        ap_data[label] = [data[key] for key in ap_keys]

# Plotting
x = range(len(ap_keys))
bar_width = 0.25
plt.figure(figsize=(10, 6))

for i, (label, values) in enumerate(ap_data.items()):
    plt.bar([p + bar_width * i for p in x],
            values, width=bar_width, label=label)

plt.xticks([p + bar_width for p in x], ap_keys)
plt.ylabel("Score")
plt.title("AP Comparison Across DETR, YOLOv11n, and YOLOv8n")
plt.legend()
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.savefig("ap_comparison.png")

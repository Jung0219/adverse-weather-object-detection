# YOLO on ExDark Dataset

This repository contains scripts to run, evaluate, and visualize YOLO object detection models on the ExDark (Exclusively Dark) dataset.

## Setup

1. Make sure you have the required dependencies installed:

   ```
   pip install torch torchvision torchaudio transformers ultralytics opencv-python numpy matplotlib tqdm pillow pycocotools
   ```

2. Download the ExDark dataset and place it in the `ExDark` directory in the root of this repository.

3. Download YOLO models:
   - YOLOv8n (default): `yolov8n.pt`
   - YOLOv11n: `yolo11n.pt`

## Available Scripts

### Run YOLO on ExDark Dataset

The `run_yolo.py` script processes the ExDark dataset with a YOLO model and saves the predictions in COCO format.

```bash
python predictions/run_yolo.py \
  --exdark_root ExDark \
  --model_path yolov8n.pt \
  --confidence 0.25 \
  --gt_coco_path ground_truth/ground_truth.json \
  --output yolo_predictions.json
```

**Arguments:**

- `--exdark_root`: Path to ExDark dataset root directory (default: ".")
- `--model_path`: Path to YOLO model (.pt file) (default: "yolo11n.pt")
- `--confidence`: Confidence threshold for detections (default: 0.25)
- `--gt_coco_path`: Path to ground truth COCO JSON file (default: "exdark_coco.json")
- `--output`: Custom output filename for YOLO predictions (optional)

This script will save the predictions in a JSON file with COCO format.

### Run DETR on ExDark Dataset

The `run_detr.py` script processes the ExDark dataset with a DETR model (`facebook/detr-resnet-50`) and saves the predictions in COCO format.

```bash
python predictions/run_detr.py \
  --exdark_root ExDark \
  --confidence 0.25 \
  --gt_coco_path ground_truth/ground_truth.json \
  --output detr_predictions.json
```

**Arguments:**

- `--exdark_root`: Path to ExDark dataset root directory (required)
- `--confidence`: Confidence threshold for detections (default: 0.25)
- `--gt_coco_path`: Path to ground truth COCO JSON file (optional)
- `--output`: Custom output filename for DETR predictions (optional)

This script will save the DETR model's predictions in a JSON file with COCO annotation format.  
It automatically maps DETR's output classes to the ExDark dataset's 12 target classes.


### Visualize YOLO Detections

The `visualize_yolo.py` script creates visualizations of YOLO detections on ExDark images.

```bash
python visualize_yolo.py \
  --exdark_root . \
  --model_path yolov8n.pt \
  --image_index 0 \
  --confidence 0.25 \
  --gt_coco_path ground_truth/ground_truth.json \
  --output_dir visualized_detections
```

**Arguments:**

- `--exdark_root`: Path to ExDark dataset root directory (default: ".")
- `--model_path`: Path to YOLO model (.pt file) (default: "yolov8n.pt")
- `--image_index`: Index of image to use from each class directory (default: 0)
- `--confidence`: Confidence threshold for detections (default: 0.25)
- `--gt_coco_path`: Path to ground truth COCO JSON file (default: "exdark_coco.json")
- `--output_dir`: Directory to save visualized images (default: "visualized_detections")

This script will process one image from each of the 12 ExDark classes and save visualization images with:

- Blue boxes: YOLO detections
- Green boxes: Ground truth (if provided)

The script will save images to the specified output directory (creates if it doesn't exist). If the directory already exists, new images will be added (potentially overwriting existing files with the same names).

### Evaluate YOLO Predictions

The `eval.py` script evaluates YOLO predictions against ground truth annotations.

```bash
python eval.py \
  --gt_coco ground_truth/ground_truth.json \
  --pred_coco yolo_predictions.json \
  --by_category \
  --output eval_results_yolov8n
```

**Arguments:**

- `--gt_coco`: Path to ground truth COCO JSON file (required)
- `--pred_coco`: Path to predictions COCO JSON file (required)
- `--by_category`: Include to evaluate performance by category
- `--output`: Custom output filename for evaluation results (optional)

This script will:

1. Calculate overall COCO metrics (AP, AP50, AP75, etc.)
2. Generate per-category metrics if `--by_category` is specified
3. Create a bar chart comparing performance across categories
4. Save results to JSON files

## Example Workflow

1. Generate COCO format predictions using YOLOv8n:

   ```
   python predictions/run_yolo.py --model_path yolov8n.pt
   ```

2. Visualize detections on sample images:

   ```
   python visualize_yolo.py --model_path yolov8n.pt --image_index 5
   ```

3. Evaluate the predictions:
   ```
   python eval.py --gt_coco exdark_coco.json --pred_coco yolov8n_predictions.json --by_category
   ```

## Notes

- The scripts map YOLO class names to ExDark classes automatically
- YOLO detections for classes not in the ExDark dataset (12 classes) are filtered out
- Visualization images include both model predictions and ground truth boxes in different colors

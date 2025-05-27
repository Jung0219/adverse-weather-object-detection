# ExDark Dataset Pipeline

This project includes preprocessing, inference, and evaluation tools for object detection models (YOLOv8, YOLOv11, and DETR) on the ExDark dataset.

---

## Preprocessing

### 1. Download the ExDark Dataset

- **Images**: [Download from Google Drive](https://drive.google.com/file/d/1BHmPgu8EsHoFDDkMGLVoXIlCth2dW6Yx/view)
- **Annotations**: [Download from Google Drive](https://drive.google.com/file/d/1P3iO3UYn7KoBi5jiUkogJq96N6maZS1i/view)
- **Class List**: [Download imageclasslist.txt](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/blob/master/Groundtruth/imageclasslist.txt)

Organize the files as follows:

```
data/ExDark/
├── images/
├── annotations/
└── imageclasslist.txt
```

---

### 2. Convert Annotations to COCO Format

```bash
python src/preprocessing/exdark_to_coco.py   --exdark_root data/ExDark   --output data/ExDark/ground_truth.json
```

This will generate `ground_truth.json` for COCO-style evaluation/training.

---

## Running Inference

### 1. Download YOLO Models

Place the following models in the `weights/` directory:

- `yolov8n.pt` – [Download from Ultralytics](https://github.com/ultralytics/ultralytics) or use:

```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -P weights/
```

- `yolov11.pt` – Place your custom model here.

---

### 2. Run YOLO Inference

```bash
python src/inference/run_yolo.py \
  --exdark_root data/ExDark \
  --model_path weights/yolov8n.pt \
  --confidence 0.25 \
  --gt_coco_path data/ExDark/ground_truth.json
```

Output:

```
data/outputs/predictions/predictions_<modelname>.json
```

---

### 3. Run DETR Inference

```bash
python src/inference/run_detr.py \
  --exdark_root data/ExDark \
  --confidence 0.25 \
  --gt_coco_path data/ExDark/ground_truth.json
```

Output:

```
data/outputs/predictions/predictions_detr-resnet-50.json
```

---

## Evaluation

Evaluate predictions against ground truth:

```bash
python src/evaluation/eval.py \
  --gt_coco data/ExDark/ground_truth.json \
  --pred_coco data/outputs/predictions/yolov8n_predictions.json \
  --by_category
```

Evaluation results are saved in:

```
data/outputs/results/
├── evaluation_results_yolov8n_predictions.json
├── category_evaluation_yolov8n_predictions.json
└── category_performance_yolov8n_predictions.png
```

> Note: `eval.py` currently saves multiple intermediate files. Consider modifying it to reduce clutter or support silent mode.

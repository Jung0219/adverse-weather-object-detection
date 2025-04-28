# run_detr.py

import os
import json
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import DetrImageProcessor, DetrForObjectDetection


def run_detr_on_exdark(exdark_root, confidence=0.25, gt_coco_path=None, output_path=None):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load DETR model and processor
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50").to(device)
    model.eval()

    # Define ExDark class names and map to IDs
    class_names = [
        'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair', 'Cup',
        'Dog', 'Motorbike', 'People', 'Table'
    ]
    class_id_map = {name: i+1 for i, name in enumerate(class_names)}

    # Map alternative names to ExDark names
    name_variations = {
        'bicycle': 'Bicycle', 'bike': 'Bicycle', 'boat': 'Boat', 'ship': 'Boat',
        'vessel': 'Boat', 'bottle': 'Bottle', 'bus': 'Bus', 'car': 'Car',
        'automobile': 'Car', 'vehicle': 'Car', 'cat': 'Cat', 'chair': 'Chair',
        'cup': 'Cup', 'mug': 'Cup', 'glass': 'Cup', 'dog': 'Dog',
        'motorbike': 'Motorbike', 'motorcycle': 'Motorbike',
        'people': 'People', 'person': 'People', 'human': 'People', 'pedestrian': 'People',
        'dining table': 'Table', 'desk': 'Table'
    }

    # Initialize COCO JSON structure
    coco_data = {
        "info": {
            "description": "DETR detections on ExDark dataset",
            "version": "1.0",
            "year": 2025,
            "contributor": "facebook/detr-resnet-50"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": class_id_map[name], "name": name, "supercategory": "object"}
            for name in class_names
        ]
    }

    # Load image list from GT COCO or walk the dataset folders
    if gt_coco_path and os.path.exists(gt_coco_path):
        with open(gt_coco_path) as f:
            gt_data = json.load(f)
        coco_data["images"] = gt_data["images"]
        images_info = gt_data["images"]
    else:
        image_dir = Path(exdark_root) / "ExDark"
        images_info = []
        image_id = 0
        for class_name in class_names:
            class_dir = image_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in class_dir.glob("*.*"):
                if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue
                with Image.open(img_path) as im:
                    width, height = im.size
                image_id += 1
                images_info.append({
                    "id": image_id,
                    "file_name": img_path.name,
                    "width": width,
                    "height": height,
                    "path": str(img_path)
                })
                coco_data["images"].append({
                    "id": image_id,
                    "file_name": img_path.name,
                    "width": width,
                    "height": height,
                    "license": 0,
                    "date_captured": "",
                    "coco_url": "",
                    "flickr_url": ""
                })

    # Process each image
    annotation_id = 0
    print("Running DETR on dataset...")
    for img_info in tqdm(images_info):
        # Resolve image path
        img_path = Path(exdark_root) / "ExDark"
        if "path" in img_info:
            img_path = Path(img_info["path"])
        else:
            for class_name in class_names:
                potential_path = Path(exdark_root) / "ExDark" / \
                    class_name / img_info["file_name"]
                if potential_path.exists():
                    img_path = potential_path
                    break

        # Load and preprocess image
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        # Run DETR inference
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor(
            [[img_info["height"], img_info["width"]]]).to(device)
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=confidence)[0]

        # Convert to COCO annotation format
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            name = model.config.id2label[label.item()].lower()
            if name in name_variations:
                exdark_name = name_variations[name]
                category_id = class_id_map[exdark_name]
            else:
                continue  # skip if not mappable

            x1, y1, x2, y2 = box.tolist()
            width = x2 - x1
            height = y2 - y1
            annotation_id += 1
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": img_info["id"],
                "category_id": category_id,
                "bbox": [x1, y1, width, height],
                "area": width * height,
                "iscrowd": 0,
                "score": score.item()
            })

    # Save predictions to JSON
    pred_path = output_path or "detr_predictions.json"
    with open(pred_path, "w") as f:
        json.dump(coco_data["annotations"], f, indent=2)

    print(f"Saved DETR predictions to {pred_path}")
    return pred_path


def main():
    # CLI for arguments
    parser = argparse.ArgumentParser(
        description="Run DETR on ExDark dataset and save in COCO format")
    parser.add_argument("--exdark_root", type=str, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--gt_coco_path", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    run_detr_on_exdark(
        exdark_root=args.exdark_root,
        confidence=args.confidence,
        gt_coco_path=args.gt_coco_path,
        output_path=args.output
    )


if __name__ == "__main__":
    main()

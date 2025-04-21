#!/usr/bin/env python3
"""
Merged script to run YOLO on ExDark dataset and convert to COCO format.
"""
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np


def run_yolo_on_exdark(exdark_root, model_path, confidence=0.25, gt_coco_path=None, output_path=None):
    """
    Run YOLO on ExDark dataset and convert predictions to COCO format

    Args:
        exdark_root: Path to ExDark dataset root
        model_path: Path to YOLO model (.pt file)
        confidence: Confidence threshold for detections
        gt_coco_path: Path to ground truth COCO JSON file (optional)
        output_path: Custom output path for predictions (optional)

    Returns:
        Path to predictions COCO JSON file
    """
    print(f"Loading YOLO model from {model_path}")
    model = YOLO(model_path)

    # ExDark class names
    class_names = [
        'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat',
        'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table'
    ]

    # Create a mapping dictionary for YOLO to ExDark name variations
    name_variations = {
        'bicycle': 'Bicycle',
        'bike': 'Bicycle',
        'boat': 'Boat',
        'ship': 'Boat',
        'vessel': 'Boat',
        'bottle': 'Bottle',
        'bus': 'Bus',
        'car': 'Car',
        'automobile': 'Car',
        'vehicle': 'Car',
        'cat': 'Cat',
        'chair': 'Chair',
        'cup': 'Cup',
        'mug': 'Cup',
        'glass': 'Cup',
        'dog': 'Dog',
        'motorbike': 'Motorbike',
        'motorcycle': 'Motorbike',
        'people': 'People',
        'person': 'People',
        'human': 'People',
        'pedestrian': 'People',
        'dining table': 'Table',
        'desk': 'Table'
    }

    class_id_map = {name: i+1 for i, name in enumerate(class_names)}
    yolo_class_map = {}  # To be filled after first inference

    # Initialize COCO structure
    coco_data = {
        "info": {
            "description": f"YOLO {Path(model_path).stem} detections on ExDark dataset",
            "version": "1.0",
            "year": 2023,
            "contributor": f"YOLO {Path(model_path).stem}"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": class_id_map[name], "name": name, "supercategory": "object"}
            for name in class_names
        ]
    }

    # If ground truth is provided, use its image list
    if gt_coco_path and os.path.exists(gt_coco_path):
        print(f"Using images from ground truth COCO file: {gt_coco_path}")
        with open(gt_coco_path, 'r') as f:
            gt_data = json.load(f)
        coco_data["images"] = gt_data["images"]

        # Process images from ground truth
        image_dir = os.path.join(exdark_root, "ExDark")
        annotation_id = 0

        # Process images
        print(f"Running YOLO on ExDark dataset images...")
        for img_info in tqdm(gt_data["images"]):
            image_id = img_info["id"]
            file_name = img_info["file_name"]

            # Find the image path (could be in any class directory)
            img_path = None
            for class_name in class_names:
                potential_path = os.path.join(image_dir, class_name, file_name)
                if os.path.exists(potential_path):
                    img_path = potential_path
                    break

            if img_path is None:
                print(f"Warning: Image {file_name} not found. Skipping.")
                continue

            # Run inference
            results = model(img_path, conf=confidence)[0]

            # Map YOLO class ids to ExDark class ids on first run
            if not yolo_class_map and len(results.boxes) > 0:
                yolo_names = results.names
                print("YOLO model class names:")
                for yolo_id, yolo_name in yolo_names.items():
                    print(f"  {yolo_id}: {yolo_name}")
                    yolo_name_lower = yolo_name.lower()

                    # Check direct match
                    if yolo_name_lower in [name.lower() for name in class_names]:
                        for exdark_name, exdark_id in class_id_map.items():
                            if yolo_name_lower == exdark_name.lower():
                                yolo_class_map[yolo_id] = exdark_id
                                break
                    # Check name variations
                    elif yolo_name_lower in name_variations:
                        exdark_name = name_variations[yolo_name_lower]
                        yolo_class_map[yolo_id] = class_id_map[exdark_name]

                print("Mapping from YOLO classes to ExDark classes:")
                for yolo_id, exdark_id in yolo_class_map.items():
                    yolo_name = yolo_names[yolo_id]
                    exdark_name = next(
                        name for name, id in class_id_map.items() if id == exdark_id)
                    print(
                        f"  YOLO class {yolo_id} ({yolo_name}) → ExDark class {exdark_id} ({exdark_name})")

                # Check for missing mappings
                missing_classes = []
                for exdark_name in class_names:
                    exdark_id = class_id_map[exdark_name]
                    if exdark_id not in yolo_class_map.values():
                        missing_classes.append(exdark_name)

                if missing_classes:
                    print(
                        f"Warning: No mapping found for ExDark classes: {', '.join(missing_classes)}")
                    print("Manual mapping may be required for these classes.")

            # Convert YOLO detections to COCO format
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                width = x2 - x1
                height = y2 - y1

                yolo_class_id = int(box.cls[0].item())
                confidence_score = float(box.conf[0].item())

                # Map YOLO class to ExDark class if possible
                if yolo_class_id in yolo_class_map:
                    category_id = yolo_class_map[yolo_class_id]
                else:
                    # Try to map using class name
                    yolo_name = results.names[yolo_class_id].lower()
                    if yolo_name in name_variations:
                        exdark_name = name_variations[yolo_name]
                        category_id = class_id_map[exdark_name]
                        yolo_class_map[yolo_class_id] = category_id
                    else:
                        # Skip classes not mappable to ExDark
                        continue

                annotation_id += 1
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "score": confidence_score,
                    "iscrowd": 0
                })
    # The rest of the function remains unchanged
    else:
        print(
            "No ground truth COCO file provided. Scanning all images in ExDark directories.")
        image_dir = os.path.join(exdark_root, "ExDark")
        image_id = 0
        annotation_id = 0

        # Process each class directory
        for class_name in class_names:
            class_img_dir = os.path.join(image_dir, class_name)

            if not os.path.exists(class_img_dir):
                print(
                    f"Warning: Image directory for class {class_name} not found.")
                continue

            # Get all image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(Path(class_img_dir).glob(ext))

            print(f"Found {len(image_files)} images in {class_name} directory")

            # Process each image
            for img_path in tqdm(image_files, desc=f"Processing {class_name}"):
                # Get image dimensions
                img = Image.open(img_path)
                width, height = img.size
                img.close()

                # Add image to COCO format
                image_id += 1
                file_name = os.path.basename(img_path)

                coco_data["images"].append({
                    "id": image_id,
                    "file_name": file_name,
                    "width": width,
                    "height": height,
                    "license": 0,
                    "date_captured": "",
                    "coco_url": "",
                    "flickr_url": ""
                })

                # Run inference
                results = model(str(img_path), conf=confidence)[0]

                # Map YOLO class ids to ExDark class ids on first run
                if not yolo_class_map and len(results.boxes) > 0:
                    yolo_names = results.names
                    print("YOLO model class names:")
                    for yolo_id, yolo_name in yolo_names.items():
                        print(f"  {yolo_id}: {yolo_name}")
                        yolo_name_lower = yolo_name.lower()

                        # Check direct match
                        if yolo_name_lower in [name.lower() for name in class_names]:
                            for exdark_name, exdark_id in class_id_map.items():
                                if yolo_name_lower == exdark_name.lower():
                                    yolo_class_map[yolo_id] = exdark_id
                                    break
                        # Check name variations
                        elif yolo_name_lower in name_variations:
                            exdark_name = name_variations[yolo_name_lower]
                            yolo_class_map[yolo_id] = class_id_map[exdark_name]

                    print("Mapping from YOLO classes to ExDark classes:")
                    for yolo_id, exdark_id in yolo_class_map.items():
                        yolo_name = yolo_names[yolo_id]
                        exdark_name = next(
                            name for name, id in class_id_map.items() if id == exdark_id)
                        print(
                            f"  YOLO class {yolo_id} ({yolo_name}) → ExDark class {exdark_id} ({exdark_name})")

                    # Check for missing mappings
                    missing_classes = []
                    for exdark_name in class_names:
                        exdark_id = class_id_map[exdark_name]
                        if exdark_id not in yolo_class_map.values():
                            missing_classes.append(exdark_name)

                    if missing_classes:
                        print(
                            f"Warning: No mapping found for ExDark classes: {', '.join(missing_classes)}")
                        print("Manual mapping may be required for these classes.")

                # Convert YOLO detections to COCO format
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    box_width = x2 - x1
                    box_height = y2 - y1

                    yolo_class_id = int(box.cls[0].item())
                    confidence_score = float(box.conf[0].item())

                    # Map YOLO class to ExDark class if possible
                    if yolo_class_id in yolo_class_map:
                        category_id = yolo_class_map[yolo_class_id]
                    else:
                        # Try to map using class name
                        yolo_name = results.names[yolo_class_id].lower()
                        if yolo_name in name_variations:
                            exdark_name = name_variations[yolo_name]
                            category_id = class_id_map[exdark_name]
                            yolo_class_map[yolo_class_id] = category_id
                        else:
                            # Use new ID for classes not in ExDark
                            yolo_name = results.names[yolo_class_id]
                            new_id = max(class_id_map.values()) + \
                                1 if class_id_map else 1
                            class_id_map[yolo_name] = new_id
                            yolo_class_map[yolo_class_id] = new_id

                            # Add new category
                            coco_data["categories"].append({
                                "id": new_id,
                                "name": yolo_name,
                                "supercategory": "object"
                            })

                            category_id = new_id

                    annotation_id += 1
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x1, y1, box_width, box_height],
                        "area": box_width * box_height,
                        "score": confidence_score,
                        "iscrowd": 0
                    })

    # Count annotations per category
    category_counts = {}
    for ann in coco_data["annotations"]:
        cat_id = ann["category_id"]
        category_counts[cat_id] = category_counts.get(cat_id, 0) + 1

    print("\nAnnotations per category:")
    for cat in coco_data["categories"]:
        cat_id = cat["id"]
        cat_name = cat["name"]
        count = category_counts.get(cat_id, 0)
        print(f"  Category {cat_id} ({cat_name}): {count} annotations")

    # Use provided output path or generate one based on model name
    if output_path:
        pred_coco_path = output_path
    else:
        pred_coco_path = f"yolo_predictions_{Path(model_path).stem}.json"

    # Save predictions to JSON file
    with open(pred_coco_path, 'w') as f:
        json.dump(coco_data["annotations"], f, indent=2)

    print(f"YOLO predictions saved to {pred_coco_path}")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total predictions: {len(coco_data['annotations'])}")

    return pred_coco_path


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO on ExDark dataset and save detections in COCO format")
    parser.add_argument("--exdark_root", type=str, default=".",
                        help="Path to ExDark dataset root directory")
    parser.add_argument("--model_path", type=str, default="yolo11n.pt",
                        help="Path to YOLO model (.pt file)")
    parser.add_argument("--confidence", type=float, default=0.25,
                        help="Confidence threshold for detections")
    parser.add_argument("--gt_coco_path", type=str, default="ground_truth/ground_truth.json",
                        help="Path to ground truth COCO JSON file (optional)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename for YOLO predictions in COCO format")

    args = parser.parse_args()

    # Make sure the model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found.")
        return

    # Make sure the ground truth COCO file exists if provided
    if args.gt_coco_path and not os.path.exists(args.gt_coco_path):
        print(
            f"Warning: Ground truth COCO file '{args.gt_coco_path}' not found.")
        use_gt = input("Continue without ground truth? (y/n): ")
        if use_gt.lower() != 'y':
            return
        args.gt_coco_path = None

    # Run YOLO on ExDark dataset
    output_path = run_yolo_on_exdark(
        args.exdark_root,
        args.model_path,
        args.confidence,
        args.gt_coco_path,
        args.output
    )

    print(f"YOLO predictions saved to {output_path}")
    print("Now you can run the check_categories.py script to verify the results.")
    print("To evaluate the results, run:")
    print(
        f"python eval.py --gt_coco {args.gt_coco_path} --pred_coco {output_path} --by_category")


if __name__ == "__main__":
    main()

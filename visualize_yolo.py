#!/usr/bin/env python3
"""
Script to visualize YOLO detections on ExDark dataset images.
Detects objects using YOLO, labels both the YOLO detections and ground truth in different colors,
and saves the visualized images into a folder.
"""
import os
import json
import argparse
from pathlib import Path
import random
from tqdm import tqdm
from ultralytics import YOLO
import cv2
import numpy as np


def visualize_yolo_detections(exdark_root, model_path, image_index=0, confidence=0.25,
                              gt_coco_path=None, output_dir="visualized_detections"):
    """
    Run YOLO on specified ExDark images, visualize detections with labels, and save images

    Args:
        exdark_root: Path to ExDark dataset root
        model_path: Path to YOLO model (.pt file)
        image_index: Index of image to process in each class directory (0 = first image)
        confidence: Confidence threshold for detections
        gt_coco_path: Path to ground truth COCO JSON file (optional)
        output_dir: Directory to save visualized images
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

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate random colors for each class for visualization
    np.random.seed(42)
    colors = {name: np.random.randint(0, 255, size=3).tolist()
              for name in class_names}

    # Ground truth annotations (if provided)
    gt_annotations = {}
    if gt_coco_path and os.path.exists(gt_coco_path):
        print(f"Loading ground truth from {gt_coco_path}")
        with open(gt_coco_path, 'r') as f:
            gt_data = json.load(f)

        # Map image_id to filename
        image_id_to_file = {img["id"]: img["file_name"]
                            for img in gt_data["images"]}

        # Map category_id to name
        category_id_to_name = {cat["id"]: cat["name"]
                               for cat in gt_data["categories"]}

        # Group annotations by image_id
        for ann in gt_data["annotations"]:
            image_id = ann["image_id"]
            if image_id not in gt_annotations:
                gt_annotations[image_id] = []
            gt_annotations[image_id].append({
                "category_id": ann["category_id"],
                "category_name": category_id_to_name.get(ann["category_id"], "Unknown"),
                "bbox": ann["bbox"]  # [x, y, width, height]
            })

    # Process one image from each class directory
    image_dir = os.path.join(exdark_root, "ExDark")

    for class_name in class_names:
        class_img_dir = os.path.join(image_dir, class_name)

        if not os.path.exists(class_img_dir):
            print(
                f"Warning: Image directory for class {class_name} not found.")
            continue

        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(Path(class_img_dir).glob(ext)))

        if not image_files:
            print(f"No images found in {class_name} directory")
            continue

        # Make sure the image_index is valid
        if image_index >= len(image_files):
            print(
                f"Image index {image_index} exceeds available images ({len(image_files)}) in {class_name}. Using last image.")
            img_path = image_files[-1]
        else:
            img_path = image_files[image_index]

        print(f"Processing {img_path}")

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Failed to load image {img_path}. Skipping.")
            continue

        # Get original dimensions
        orig_height, orig_width = image.shape[:2]

        # Run YOLO inference
        results = model(image, conf=confidence)[0]

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

        # Draw YOLO detections (blue)
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            yolo_class_id = int(box.cls[0].item())
            confidence_score = float(box.conf[0].item())

            # Filter out classes that are not in ExDark dataset
            if yolo_class_id not in yolo_class_map:
                yolo_name = results.names[yolo_class_id].lower()
                if yolo_name in name_variations:
                    exdark_name = name_variations[yolo_name]
                    category_id = class_id_map[exdark_name]
                    yolo_class_map[yolo_class_id] = category_id
                else:
                    # Skip classes not in ExDark
                    continue

            # Get the ExDark class
            exdark_class_id = yolo_class_map[yolo_class_id]
            exdark_class_name = next(
                name for name, id in class_id_map.items() if id == exdark_class_id)

            # Draw bounding box and label (blue for detections)
            color = colors[exdark_class_name]
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Add text background
            text = f"{exdark_class_name}: {confidence_score:.2f}"
            text_size = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                image, (x1, y1-text_size[1]-5), (x1+text_size[0], y1), (255, 0, 0), -1)

            # Add text
            cv2.putText(image, text, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw ground truth boxes if available (green)
        if gt_coco_path:
            # Find image in ground truth by filename
            file_name = os.path.basename(img_path)
            image_id = None

            for img in gt_data["images"]:
                if img["file_name"] == file_name:
                    image_id = img["id"]
                    break

            if image_id is not None and image_id in gt_annotations:
                for ann in gt_annotations[image_id]:
                    # Extract coordinates [x, y, width, height] -> [x1, y1, x2, y2]
                    x, y, w, h = ann["bbox"]
                    x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)

                    # Draw bounding box and label (green for ground truth)
                    category_name = ann["category_name"]
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Add text background
                    text = f"GT: {category_name}"
                    text_size = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(
                        image, (x1, y1-text_size[1]-5), (x1+text_size[0], y1), (0, 255, 0), -1)

                    # Add text
                    cv2.putText(image, text, (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Add title with class name and model name
        model_name = Path(model_path).stem
        title = f"Class: {class_name} - Model: {model_name}"
        cv2.putText(image, title, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Add legend
        cv2.putText(image, "Blue: YOLO detections", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(image, "Green: Ground Truth", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save image
        output_file = os.path.join(
            output_dir, f"{class_name}_{image_index}_{model_name}.jpg")
        cv2.imwrite(output_file, image)
        print(f"Saved visualization to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize YOLO detections on ExDark dataset images")
    parser.add_argument("--exdark_root", type=str, default=".",
                        help="Path to ExDark dataset root directory")
    parser.add_argument("--model_path", type=str, default="yolov8n.pt",
                        help="Path to YOLO model (.pt file)")
    parser.add_argument("--image_index", type=int, default=0,
                        help="Index of image to use from each class directory (0 = first image)")
    parser.add_argument("--confidence", type=float, default=0.25,
                        help="Confidence threshold for detections")
    parser.add_argument("--gt_coco_path", type=str, default="exdark_coco.json",
                        help="Path to ground truth COCO JSON file (optional)")
    parser.add_argument("--output_dir", type=str, default="visualized_detections",
                        help="Directory to save visualized images")

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

    # Visualize YOLO detections on ExDark dataset
    visualize_yolo_detections(
        args.exdark_root,
        args.model_path,
        args.image_index,
        args.confidence,
        args.gt_coco_path,
        args.output_dir
    )

    print(f"Visualizations saved to {args.output_dir} directory")


if __name__ == "__main__":
    main()

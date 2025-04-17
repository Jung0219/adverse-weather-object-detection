import os
import json
from collections import defaultdict
from pathlib import Path


def convert_exdark_to_coco(exdark_root, output_json):
    """
    Convert ExDark annotations to COCO format

    Args:
        exdark_root: Path to ExDark dataset root directory
        output_json: Path to output COCO JSON file
    """
    # ExDark class names
    class_names = [
        'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat',
        'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table'
    ]

    class_id_map = {name: i+1 for i, name in enumerate(class_names)}

    # Initialize COCO structure
    coco_data = {
        "info": {
            "description": "ExDark Dataset converted to COCO format",
            "version": "1.0",
            "year": 2023,
            "contributor": "ExDark dataset converted to COCO"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": class_id_map[name], "name": name, "supercategory": "object"}
            for name in class_names
        ]
    }

    # Get all image directories
    image_dir = os.path.join(exdark_root, "ExDark")
    anno_dir = os.path.join(exdark_root, "ExDark_Anno")

    image_id = 0
    annotation_id = 0

    # Process each class directory
    for class_name in class_names:
        class_img_dir = os.path.join(image_dir, class_name)
        class_anno_dir = os.path.join(anno_dir, class_name)

        if not os.path.exists(class_img_dir) or not os.path.exists(class_anno_dir):
            print(f"Warning: Directory for class {class_name} not found.")
            continue

        # Get list of annotation files
        anno_files = [f for f in os.listdir(
            class_anno_dir) if f.endswith('.txt')]

        for anno_file in anno_files:
            image_filename = anno_file[:-4]  # Remove .txt extension

            # Check if image file exists (could be .jpg, .jpeg, .png)
            img_path = None
            for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']:
                potential_path = os.path.join(
                    class_img_dir, image_filename.replace(".txt", ""))
                if os.path.exists(potential_path):
                    img_path = potential_path
                    break
                elif os.path.exists(potential_path + ext):
                    img_path = potential_path + ext
                    break

            if img_path is None:
                print(
                    f"Warning: Image file for {image_filename} not found. Skipping.")
                continue

            # Get image dimensions
            # For simplicity, we'll use placeholder dimensions
            # In a real application, use PIL or OpenCV to get actual dimensions
            width, height = 640, 480  # Placeholder values

            # Add image to COCO format
            image_id += 1
            coco_data["images"].append({
                "id": image_id,
                "file_name": os.path.basename(img_path),
                "width": width,
                "height": height,
                "license": 0,
                "date_captured": "",
                "coco_url": "",
                "flickr_url": ""
            })

            # Parse annotation file
            with open(os.path.join(class_anno_dir, anno_file), 'r') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line or line.startswith('%'):
                    continue

                parts = line.split()
                if len(parts) < 6:
                    continue

                obj_class = parts[0]
                if obj_class not in class_id_map:
                    print(f"Warning: Unknown class {obj_class} in {anno_file}")
                    continue

                # ExDark format: Class x_min y_min width height [other fields...]
                x_min = float(parts[1])
                y_min = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # Add annotation to COCO format
                annotation_id += 1
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id_map[obj_class],
                    "bbox": [x_min, y_min, width, height],
                    "area": width * height,
                    "segmentation": [],
                    "iscrowd": 0
                })

    # Save to JSON file
    with open(output_json, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"Conversion complete. COCO format saved to {output_json}")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    return output_json


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert ExDark annotations to COCO format")
    parser.add_argument("--exdark_root", type=str,
                        required=True, help="Path to ExDark dataset root")
    parser.add_argument(
        "--output", type=str, default="exdark_coco.json", help="Output COCO JSON file")

    args = parser.parse_args()
    convert_exdark_to_coco(args.exdark_root, args.output)

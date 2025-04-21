import os
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_coco(gt_json, pred_json, output_file=None):
    """
    Evaluate predictions against ground truth using COCO metrics

    Args:
        gt_json: Path to ground truth COCO JSON file
        pred_json: Path to predictions COCO JSON file or just annotations array
        output_file: Optional custom output filename for results (default: evaluation_results_{pred_json_stem}.json)
    """
    print(f"Evaluating predictions against ground truth...")

    # Load ground truth
    coco_gt = COCO(gt_json)

    # Handle both full COCO format and just annotations array
    with open(pred_json, 'r') as f:
        pred_data = json.load(f)

    # Check if predictions are in full COCO format or just annotations
    if isinstance(pred_data, dict) and 'annotations' in pred_data:
        print("Detected full COCO format in predictions file. Extracting annotations...")
        # Extract just the annotations for loadRes
        annotations = pred_data['annotations']

        # Save to a temporary file
        temp_file = f"temp_annotations_{Path(pred_json).stem}.json"
        with open(temp_file, 'w') as f:
            json.dump(annotations, f)

        # Load the processed predictions
        coco_dt = coco_gt.loadRes(temp_file)
        print(f"Loaded {len(annotations)} annotations for evaluation")
    else:
        # Assume it's already in the right format (just annotations)
        print("Using annotations directly from predictions file...")
        coco_dt = coco_gt.loadRes(pred_json)
        print(f"Loaded {len(pred_data)} annotations for evaluation")

    # Initialize evaluator
    cocoEval = COCOeval(coco_gt, coco_dt, 'bbox')

    # Run evaluation
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # Save evaluation results
    eval_results = {
        "AP": float(cocoEval.stats[0]),
        "AP50": float(cocoEval.stats[1]),
        "AP75": float(cocoEval.stats[2]),
        "AP_small": float(cocoEval.stats[3]),
        "AP_medium": float(cocoEval.stats[4]),
        "AP_large": float(cocoEval.stats[5]),
        "AR_max1": float(cocoEval.stats[6]),
        "AR_max10": float(cocoEval.stats[7]),
        "AR_max100": float(cocoEval.stats[8]),
        "AR_small": float(cocoEval.stats[9]),
        "AR_medium": float(cocoEval.stats[10]),
        "AR_large": float(cocoEval.stats[11])
    }

    # Use custom output filename if provided, otherwise use default
    if output_file:
        results_json = output_file
    else:
        results_json = f"evaluation_results_{Path(pred_json).stem}.json"

    with open(results_json, 'w') as f:
        json.dump(eval_results, f, indent=2)

    print(f"Evaluation results saved to {results_json}")
    return results_json


def evaluate_by_category(gt_json, pred_json, output_file=None):
    """
    Evaluate predictions by category and generate per-category metrics

    Args:
        gt_json: Path to ground truth COCO JSON file
        pred_json: Path to predictions COCO JSON file or just annotations array
        output_file: Optional custom output filename for category results
    """
    print(f"Evaluating predictions by category...")

    # Load ground truth
    coco_gt = COCO(gt_json)

    # Handle both full COCO format and just annotations array
    with open(pred_json, 'r') as f:
        pred_data = json.load(f)

    # Check if predictions are in full COCO format or just annotations
    if isinstance(pred_data, dict) and 'annotations' in pred_data:
        print("Detected full COCO format in predictions file. Extracting annotations...")
        # Extract just the annotations for loadRes
        annotations = pred_data['annotations']

        # Save to a temporary file
        temp_file = f"temp_annotations_{Path(pred_json).stem}.json"
        with open(temp_file, 'w') as f:
            json.dump(annotations, f)

        # Load the processed predictions
        coco_dt = coco_gt.loadRes(temp_file)
        print(f"Loaded {len(annotations)} annotations for evaluation")
    else:
        # Assume it's already in the right format (just annotations)
        print("Using annotations directly from predictions file...")
        coco_dt = coco_gt.loadRes(pred_json)
        print(f"Loaded {len(pred_data)} annotations for evaluation")

    # Get categories
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    cat_names = [cat['name'] for cat in cats]
    print(f"Found {len(cat_names)} categories: {', '.join(cat_names)}")

    # Evaluate per category
    cat_metrics = {}
    for cat_id in coco_gt.getCatIds():
        cat_name = next(cat['name'] for cat in cats if cat['id'] == cat_id)
        print(f"Evaluating category {cat_id}: {cat_name}")

        # Check if there are any ground truth annotations for this category
        gt_ann_ids = coco_gt.getAnnIds(catIds=[cat_id])
        if len(gt_ann_ids) == 0:
            print(
                f"  Warning: No ground truth annotations for category {cat_name}")
            cat_metrics[cat_name] = {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}
            continue

        # Check if there are any prediction annotations for this category
        dt_ann_ids = coco_dt.getAnnIds(catIds=[cat_id])
        if len(dt_ann_ids) == 0:
            print(f"  Warning: No predictions for category {cat_name}")
            cat_metrics[cat_name] = {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}
            continue

        print(
            f"  Found {len(gt_ann_ids)} ground truth and {len(dt_ann_ids)} predictions")

        # Initialize evaluator for this category
        try:
            cocoEval = COCOeval(coco_gt, coco_dt, 'bbox')
            cocoEval.params.catIds = [cat_id]

            # Run evaluation
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            # Verify stats are populated
            if len(cocoEval.stats) >= 3:
                cat_metrics[cat_name] = {
                    "AP": float(cocoEval.stats[0]),
                    "AP50": float(cocoEval.stats[1]),
                    "AP75": float(cocoEval.stats[2]),
                    "AP_small": float(cocoEval.stats[3]),
                    "AP_medium": float(cocoEval.stats[4]),
                    "AP_large": float(cocoEval.stats[5]),
                    "AR_max1": float(cocoEval.stats[6]),
                    "AR_max10": float(cocoEval.stats[7]),
                    "AR_max100": float(cocoEval.stats[8]),
                    "AR_small": float(cocoEval.stats[9]),
                    "AR_medium": float(cocoEval.stats[10]),
                    "AR_large": float(cocoEval.stats[11])
                }
                print(
                    f"  Results: AP={cat_metrics[cat_name]['AP']:.4f}, AP50={cat_metrics[cat_name]['AP50']:.4f}")
            else:
                print(f"  Warning: Insufficient stats for category {cat_name}")
                cat_metrics[cat_name] = {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}
        except Exception as e:
            print(f"  Error evaluating category {cat_name}: {e}")
            cat_metrics[cat_name] = {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}

    # Determine output filename for category results
    if output_file:
        # If a custom overall output is provided, derive the category output name
        base_name = Path(output_file).stem
        cat_results_json = f"category_evaluation_{base_name}.json"
        chart_filename = f"category_performance_{base_name}.png"
    else:
        cat_results_json = f"category_evaluation_{Path(pred_json).stem}.json"
        chart_filename = f"category_performance_{Path(pred_json).stem}.png"

    with open(cat_results_json, 'w') as f:
        json.dump(cat_metrics, f, indent=2)

    print(f"Category evaluation results saved to {cat_results_json}")

    # Generate bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    categories = list(cat_metrics.keys())
    ap_values = [cat_metrics[cat]["AP"] for cat in categories]
    ap50_values = [cat_metrics[cat]["AP50"] for cat in categories]

    x = np.arange(len(categories))
    width = 0.35

    ax.bar(x - width/2, ap_values, width, label='AP')
    ax.bar(x + width/2, ap50_values, width, label='AP50')

    ax.set_ylabel('Score')
    ax.set_title(f'Performance by Category - {Path(pred_json).stem}')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(chart_filename, dpi=300)
    plt.close()

    print(f"Category performance chart saved to {chart_filename}")
    return cat_results_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO predictions against ground truth annotations")
    parser.add_argument("--gt_coco", type=str, required=True,
                        help="Path to ground truth COCO JSON file")
    parser.add_argument("--pred_coco", type=str, required=True,
                        help="Path to predictions COCO JSON file (full format or just annotations)")
    parser.add_argument("--by_category", action="store_true",
                        help="Evaluate performance by category")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom output filename for evaluation results")

    args = parser.parse_args()

    # Run overall evaluation
    eval_results = evaluate_coco(args.gt_coco, args.pred_coco, args.output)

    # Run category evaluation if requested
    if args.by_category:
        cat_results = evaluate_by_category(
            args.gt_coco, args.pred_coco, args.output)

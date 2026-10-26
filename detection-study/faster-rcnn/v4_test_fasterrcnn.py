from pathlib import Path
import csv
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from fasterrcnn_dataset import YoloTxtDetectionDataset, collate_fn


ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "runs" / "detector_analysis_all_trained_fasterrcnn_v4" / "best.pt"

YAML_DIR = ROOT.parent / "yamls-detector-analysis"

RESULTS_DIR = ROOT / "runs" / "detector_analysis_all_fasterrcnn_tests_v4"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DOMAINS = [
    "complex",
    "simple",
    "textured",
    "blue",
    "green",
    "natural",
    "bright",
    "dark",
    "med_ill",
    "crowded",
    "mod_layout",
    "sparse",
    "rotated",
    "tilted",
    "upright",
    "front",
    "nadir",
    "oblique",
    "large",
    "med_scale",
    "small",
    "high_vis",
    "low_vis",
    "mod_vis",
]


def box_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter = max(0, xB - xA) * max(0, yB - yA)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def evaluate_domain(model, yaml_path, domain, device, conf_thres=0.25, iou_thres=0.5):
    dataset = YoloTxtDetectionDataset(yaml_path, split="val")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    map_metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
    )

    total_gt = 0
    total_pred = 0
    tp = 0
    fp = 0
    fn = 0

    pred_rows = []

    model.eval()

    with torch.no_grad():
        for images, targets in tqdm(loader, desc=domain):
            image = images[0].to(device)
            target = targets[0]

            output = model([image])[0]

            pred_boxes = output["boxes"].cpu()
            pred_labels = output["labels"].cpu()
            pred_scores = output["scores"].cpu()

            keep = pred_scores >= conf_thres
            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]
            pred_scores = pred_scores[keep]

            gt_boxes = target["boxes"].cpu()
            gt_labels = target["labels"].cpu()

            map_metric.update(
                [
                    {
                        "boxes": pred_boxes,
                        "scores": pred_scores,
                        "labels": pred_labels,
                    }
                ],
                [
                    {
                        "boxes": gt_boxes,
                        "labels": gt_labels,
                    }
                ],
            )

            total_gt += len(gt_boxes)
            total_pred += len(pred_boxes)

            matched_gt = set()

            for pb, pl, ps in zip(pred_boxes, pred_labels, pred_scores):
                best_iou = 0
                best_gt_idx = -1

                for gi, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                    if gi in matched_gt:
                        continue

                    if int(pl) != int(gl):
                        continue

                    iou = box_iou(pb.tolist(), gb.tolist())

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gi

                if best_iou >= iou_thres:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1

                pred_rows.append(
                    {
                        "domain": domain,
                        "class_id": int(pl) - 1,
                        "score": float(ps),
                        "x1": float(pb[0]),
                        "y1": float(pb[1]),
                        "x2": float(pb[2]),
                        "y2": float(pb[3]),
                    }
                )

            fn += len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    map_results = map_metric.compute()

    return {
        "domain": domain,
        "images": len(dataset),
        "gt_objects": total_gt,
        "predictions": total_pred,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mAP50": float(map_results["map_50"]),
        "mAP50-95": float(map_results["map"]),
    }, pred_rows


def main():
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from: {MODEL_PATH}")

    model = fasterrcnn_resnet50_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 5)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)

    summary_rows = []
    all_predictions = []

    for domain in DOMAINS:
        yaml_path = YAML_DIR / f"all_{domain}_data_test.yaml"

        if not yaml_path.exists():
            print(f"Skipping missing YAML: {yaml_path}")
            continue

        print(f"\nEvaluating domain: {domain}")
        print(f"Using YAML: {yaml_path}")

        metrics, preds = evaluate_domain(
            model=model,
            yaml_path=yaml_path,
            domain=domain,
            device=device,
            conf_thres=0.25,
            iou_thres=0.5,
        )

        summary_rows.append(metrics)
        all_predictions.extend(preds)

        print(
            f"{domain}: "
            f"mAP50={metrics['mAP50']:.4f}, "
            f"mAP50-95={metrics['mAP50-95']:.4f}, "
            f"P={metrics['precision']:.4f}, "
            f"R={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}"
        )

    if not summary_rows:
        raise RuntimeError("No domains were evaluated. Check YAML filenames.")

    summary_file = RESULTS_DIR / "fasterrcnn_domain_summary_v4.csv"
    pred_file = RESULTS_DIR / "fasterrcnn_predictions_v4.csv"

    with open(summary_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(pred_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["domain", "class_id", "score", "x1", "y1", "x2", "y2"],
        )
        writer.writeheader()
        writer.writerows(all_predictions)

    print(f"\nSaved summary to: {summary_file}")
    print(f"Saved predictions to: {pred_file}")


if __name__ == "__main__":
    main()
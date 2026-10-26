from ultralytics import YOLO
import os
from pathlib import Path
import csv


DOMAINS = ["mixed"]

# set the working directory
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

yaml_path = Path(__file__).resolve().parent.parent / "yamls-detector-analysis" / "all_mixed_data_test.yaml"

# model paths
saved_model = ROOT / 'runs/detect/detector_analysis_all_trained_yolo_v4/train/weights/best.pt'

# check if model files exist
assert saved_model.exists(), f"Model file not found: {saved_model}"

# load models
model_50 = YOLO(saved_model)

# CSV setup
csv_path = ROOT / 'summary_test_results_all/testing_on_all_trained_yolo_v4/mixed.csv'
csv_path.parent.mkdir(parents=True, exist_ok=True)

with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # header
    writer.writerow([
        "domain",
        "model_epochs",
        "mAP50",
        "mAP50-95",
        "precision",
        "recall"
    ])

    for domain in DOMAINS:
        project_name = f"{domain}_test"


        # --- 50 epochs ---
        metrics_50 = model_50.val(
            data=yaml_path,
            conf=0.25,
            save_json=True,
            visualize=True,
            project=f"{project_name}_"
        )

        writer.writerow([
            domain,
            50,
            metrics_50.box.map50,
            metrics_50.box.map,
            metrics_50.box.mp,
            metrics_50.box.mr
        ])

print(f"Results saved to: {csv_path}")
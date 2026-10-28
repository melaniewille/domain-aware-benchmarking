from ultralytics import RTDETR
import os
from pathlib import Path
import csv

DOMAINS = ["high_vis", "mod_vis", "low_vis"]

# set the working directory
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

# model paths
saved_model = ROOT / 'runs/detect/detector_analysis_all_trained_rtdetr_v5/train/weights/best.pt'

# check if model file exists and load
assert saved_model.exists(), f"Model file not found: {saved_model}"

model_50 = RTDETR(saved_model)

# CSV setup
csv_path = ROOT / 'summary_test_results_all/testing_on_all_trained_rtdetr_v5/visibility.csv'
csv_path.parent.mkdir(parents=True, exist_ok=True)

with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    
   
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
        yaml_path = Path(__file__).resolve().parent.parent / "yamls-detector-analysis" / f"all_{domain}_data_test.yaml"
        


        metrics_50 = model_50.val(
            data=yaml_path,
            conf=0.25,
            save_json=True,
            visualize=True,
            project=f"{project_name}"
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
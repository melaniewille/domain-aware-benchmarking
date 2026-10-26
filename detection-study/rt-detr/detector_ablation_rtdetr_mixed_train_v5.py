from ultralytics import RTDETR
from pathlib import Path
import os

os.environ["ULTRALYTICS_REQUIREMENTS"] = "0"

project_name = "detector_analysis_all_trained_rtdetr_v5"

# set the working directory
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

yaml_path = Path(__file__).resolve().parent.parent / "yamls-detector-analysis" / "v5_all_mixed_data_train.yaml"

# Load a model
model = RTDETR('rtdetr-l.pt')   # load a pretrained model (recommended for training)

# Train the model
model.train(data=yaml_path, epochs=50, imgsz=640, batch=16, project=project_name)  # train the model under standard parameters

# Best model saved per default in "current_directory/runs/detect/combi_mixed_train/train/weights/best.pt"

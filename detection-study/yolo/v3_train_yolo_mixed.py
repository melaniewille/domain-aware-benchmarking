from ultralytics import YOLO
from pathlib import Path
import os

os.environ["ULTRALYTICS_REQUIREMENTS"] = "0"

project_name = "detector_analysis_all_trained_yolo_v3"

# set the working directory
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

yaml_path = Path(__file__).resolve().parent.parent / "yamls-detector-analysis" / "v3_all_mixed_data_train.yaml"

# load model
model = YOLO('yolo26n.pt')   

# train model
model.train(data=yaml_path, epochs=50, imgsz=640, batch=16, project=project_name)  



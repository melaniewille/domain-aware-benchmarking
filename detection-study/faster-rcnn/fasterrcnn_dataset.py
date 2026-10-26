from pathlib import Path
import yaml
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torchvision.transforms import functional as F

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YoloTxtDetectionDataset(Dataset):
    def __init__(self, yaml_path, split="train"):
        self.yaml_path = Path(yaml_path).resolve()
        with open(self.yaml_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.root = (self.yaml_path.parent / self.cfg["path"]).resolve()

        split_file = Path(self.cfg[split])

        # If train/val starts with ../, resolve it relative to the YAML file location
        if str(split_file).startswith("../"):
            self.split_txt = (self.yaml_path.parent / split_file).resolve()
        else:
            self.split_txt = (self.root / split_file).resolve()

        self.image_paths = []
        for line in self.split_txt.read_text().splitlines():
            line = line.strip()
            if not line:
                continue

            line_path = Path(line)

            # Handle ../ image paths relative to the YAML file location
            if str(line_path).startswith("../"):
                p = (self.yaml_path.parent / line_path).resolve()
            else:
                p = (self.root / line_path).resolve()

            self.image_paths.append(p)

        self.num_classes = int(self.cfg["nc"]) + 1  # + background
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        label_path = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")

        boxes = []
        labels = []

        if label_path.exists():
            for line in label_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                cls, xc, yc, bw, bh = map(float, parts)
                cls = int(cls) + 1  # Faster R-CNN uses 0 as background

                x1 = (xc - bw / 2) * w
                y1 = (yc - bh / 2) * h
                x2 = (xc + bw / 2) * w
                y2 = (yc + bh / 2) * h

                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        img = F.to_tensor(img)
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))

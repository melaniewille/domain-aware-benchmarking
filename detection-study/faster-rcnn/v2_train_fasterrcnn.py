from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from fasterrcnn_dataset import YoloTxtDetectionDataset, collate_fn
from tqdm import tqdm

def main():
    project_name = "detector_analysis_all_trained_fasterrcnn_v2"

    ROOT = Path(__file__).resolve().parent
    yaml_path = ROOT.parent / "yamls-detector-analysis" / "v2_all_mixed_data_train.yaml"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = YoloTxtDetectionDataset(yaml_path, split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=3,
        collate_fn=collate_fn,
    )

    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, train_dataset.num_classes)

    model.to(device)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
    )

    num_epochs = 50
    save_dir = ROOT / "runs" / project_name
    save_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for images, targets in tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{num_epochs}",
                leave=False):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        print(f"Epoch {epoch+1}/{num_epochs}, loss={total_loss:.4f}")

        torch.save(model.state_dict(), save_dir / "last.pt")

        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), save_dir / "best.pt")
            print("Saved new best model")

    print(f"Saved model to {save_dir / 'best.pt'}")



if __name__ == "__main__":
    main()
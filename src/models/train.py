import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

import mlflow
import mlflow.pytorch

from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

from .model import SimpleCNN


def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        y_true.extend(y.numpy().tolist())
        y_pred.extend(pred)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return acc, cm


def save_confusion_matrix(cm, out_path: Path):
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_loss_curve(losses, out_path: Path):
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="data/processed/splits (contains train/val/test)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mlflow_uri", type=str, default="http://127.0.0.1:5000")
    p.add_argument("--experiment", type=str, default="cats-vs-dogs")
    p.add_argument("--out_dir", type=str, default="artifacts")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = ImageFolder(data_dir / "train", transform=get_transforms(train=True))
    val_ds = ImageFolder(data_dir / "val", transform=get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = SimpleCNN(num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    losses = []
    start = time.time()

    with mlflow.start_run():
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("lr", args.lr)
        mlflow.log_param("device", device)

        model.train()
        step_count = 0
        for epoch in range(args.epochs):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()

                losses.append(float(loss.item()))
                if step_count % 25 == 0:
                    mlflow.log_metric("train_loss", float(loss.item()), step=step_count)
                step_count += 1

        val_acc, cm = evaluate(model, val_loader, device)
        mlflow.log_metric("val_accuracy", float(val_acc))

        model_path = out_dir / "model.pt"
        torch.save(model.state_dict(), model_path)

        metrics_path = out_dir / "metrics.json"
        metrics_path.write_text(json.dumps({"val_accuracy": float(val_acc)}, indent=2))

        cm_path = out_dir / "confusion_matrix.png"
        save_confusion_matrix(cm, cm_path)

        loss_path = out_dir / "loss_curve.png"
        save_loss_curve(losses, loss_path)

        # log artifacts to MLflow
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(loss_path))

        # log model to MLflow (optional but good for marks)
        mlflow.pytorch.log_model(model, artifact_path="model")

    print(f"Training finished in {time.time() - start:.1f}s")
    print(f"Artifacts saved to: {out_dir.resolve()}")
    print(f"Val accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()

    # CI demo trigger comment
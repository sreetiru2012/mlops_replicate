from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

from src.models.model import SimpleCNN


class Predictor:
    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SimpleCNN(num_classes=2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.labels = {0: "cat", 1: "dog"}

    @torch.no_grad()
    def predict(self, image: Image.Image):
        x = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = float(probs[0][pred].item())
        return {
            "label": self.labels[pred],
            "confidence": confidence
        }
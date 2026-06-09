from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

MODEL_DIR = Path(__file__).resolve().parent / "models" / "cnn_classifier"
CHECKPOINT_FILE = "best_model.pt"
CLASSES_FILE = "classes.txt"
CLASSES_JSON_FILE = "classes.json"

IMAGE_SIZE = 128
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class CnnPrediction:
    label: str
    confidence: float
    top_k: tuple[tuple[str, float], ...]


def load_env(path: Path | None = None) -> None:
    env_path = path or Path.cwd() / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def model_files_exist(model_dir: Path = MODEL_DIR) -> bool:
    return (model_dir / CHECKPOINT_FILE).exists() and (model_dir / CLASSES_FILE).exists()


def read_classes(model_dir: Path = MODEL_DIR) -> list[str]:
    classes_json = model_dir / CLASSES_JSON_FILE
    if classes_json.exists():
        with classes_json.open(encoding="utf-8") as f:
            data = json.load(f)
        names = data.get("class_names")
        if isinstance(names, list) and all(isinstance(name, str) for name in names):
            return names

    classes_txt = model_dir / CLASSES_FILE
    with classes_txt.open(encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def download_model_from_mlflow(model_dir: Path = MODEL_DIR) -> Path:
    load_env()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    model_name = os.environ.get("MLFLOW_REGISTERED_MODEL_NAME")
    model_alias = os.environ.get("MLFLOW_MODEL_ALIAS")
    if not tracking_uri or not model_name or not model_alias:
        raise RuntimeError(
            "Defina MLFLOW_TRACKING_URI, MLFLOW_REGISTERED_MODEL_NAME e "
            "MLFLOW_MODEL_ALIAS no .env para baixar a CNN."
        )

    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    model_version = client.get_model_version_by_alias(model_name, model_alias)

    model_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "checkpoint/best_model.pt": CHECKPOINT_FILE,
        "classes/classes.txt": CLASSES_FILE,
        "classes/classes.json": CLASSES_JSON_FILE,
    }
    for artifact_path, output_name in artifacts.items():
        local_path = Path(
            client.download_artifacts(
                model_version.run_id,
                artifact_path,
                dst_path=str(model_dir),
            )
        )
        target_path = model_dir / output_name
        if local_path != target_path:
            local_path.replace(target_path)

    metadata = {
        "tracking_uri": tracking_uri,
        "registered_model_name": model_name,
        "model_alias": model_alias,
        "model_version": model_version.version,
        "run_id": model_version.run_id,
        "source": model_version.source,
    }
    with (model_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return model_dir


def ensure_model_files(model_dir: Path = MODEL_DIR) -> Path:
    if not model_files_exist(model_dir):
        return download_model_from_mlflow(model_dir)
    return model_dir


def _torch_modules() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "A CNN baixada é um checkpoint PyTorch. Instale a dependência `torch` "
            "antes de executar o relabeler."
        ) from exc
    return torch, nn


def _conv_block(nn: Any, in_channels: int, out_channels: int) -> Any:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


def build_model(num_classes: int) -> Any:
    _, nn = _torch_modules()

    class SmallTrafficSignCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                _conv_block(nn, 3, 32),
                _conv_block(nn, 32, 64),
                _conv_block(nn, 64, 128),
                _conv_block(nn, 128, 256),
                _conv_block(nn, 256, 384),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(384, num_classes),
            )

        def forward(self, x: Any) -> Any:
            x = self.features(x)
            return self.classifier(x)

    return SmallTrafficSignCNN()


class CnnClassifier:
    def __init__(self, model: Any, classes: list[str], device: str) -> None:
        self.model = model
        self.classes = classes
        self.device = device

    @classmethod
    def from_model_dir(cls, model_dir: Path = MODEL_DIR) -> "CnnClassifier":
        model_dir = ensure_model_files(model_dir)
        classes = read_classes(model_dir)
        torch, _ = _torch_modules()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = build_model(len(classes)).to(device)
        checkpoint = torch.load(model_dir / CHECKPOINT_FILE, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.eval()
        return cls(model=model, classes=classes, device=device)

    def predict(self, image: Image.Image, top_k: int = 3) -> CnnPrediction:
        torch, _ = _torch_modules()
        tensor = self._preprocess(image, torch).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu()

        k = min(top_k, len(self.classes))
        scores, indices = torch.topk(probabilities, k=k)
        top_predictions = tuple(
            (self.classes[int(index)], float(score))
            for score, index in zip(scores, indices, strict=True)
        )
        label, confidence = top_predictions[0]
        return CnnPrediction(label=label, confidence=confidence, top_k=top_predictions)

    def _preprocess(self, image: Image.Image, torch: Any) -> Any:
        resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        array = (array - np.array(NORMALIZATION_MEAN, dtype=np.float32)) / np.array(
            NORMALIZATION_STD,
            dtype=np.float32,
        )
        array = np.transpose(array, (2, 0, 1))
        return torch.from_numpy(array).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa a CNN do MLflow configurado no .env")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    model_dir = download_model_from_mlflow(args.model_dir)
    print(f"Modelo CNN salvo em: {model_dir}")
    print(f"Classes: {len(read_classes(model_dir))}")


if __name__ == "__main__":
    main()

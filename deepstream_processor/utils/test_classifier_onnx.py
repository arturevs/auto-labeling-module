#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "detection_models" / "classifier.onnx"
DEFAULT_LABELS = ROOT / "detection_models" / "labels_classifier.txt"
DEFAULT_IMAGES = [
    ROOT / "data" / "input" / "pare.png",
    ROOT / "data" / "input" / "seta.png",
]
EXPECTED_LABELS = {
    "pare.png": "R-1",
    "seta.png": "R-26",
}
DEEPSTREAM_OFFSETS = np.array([123.675, 116.28, 103.53], dtype=np.float32)
DEEPSTREAM_SCALE = np.float32(0.01735211)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the traffic sign classifier ONNX directly, outside DeepStream."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--preprocess",
        choices=("deepstream", "imagenet", "unit"),
        nargs="+",
        default=("deepstream", "imagenet", "unit"),
        help="deepstream matches config_infer_secondary_traffic_sign.txt.",
    )
    parser.add_argument("images", type=Path, nargs="*", default=DEFAULT_IMAGES)
    return parser.parse_args()


def read_labels(path: Path) -> list[str]:
    return [
        label.strip()
        for line in path.read_text().splitlines()
        for label in line.split(";")
        if label.strip()
    ]


def load_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def preprocess(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "deepstream":
        image = (image - DEEPSTREAM_OFFSETS) * DEEPSTREAM_SCALE
    elif mode == "imagenet":
        image = (image / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    elif mode == "unit":
        image = image / 255.0
    else:
        raise ValueError(f"Unknown preprocess mode: {mode}")
    return np.transpose(image, (2, 0, 1))


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def print_results(
    image_paths: list[Path],
    labels: list[str],
    logits: np.ndarray,
    mode: str,
    top_k: int,
) -> None:
    probabilities = softmax(logits)
    print(f"\npreprocess={mode}")
    for path, image_logits, image_probabilities in zip(image_paths, logits, probabilities):
        best_indexes = np.argsort(image_probabilities)[::-1][:top_k]
        expected = EXPECTED_LABELS.get(path.name)
        predicted = labels[int(best_indexes[0])]
        verdict = ""
        if expected is not None:
            verdict = f" expected={expected} verdict={'PASS' if predicted == expected else 'FAIL'}"
        print(f"{path.name}: predicted={predicted}{verdict}")
        for index in best_indexes:
            print(
                f"  {int(index):>2} {labels[int(index)]:<10}"
                f" probability={image_probabilities[index]:.6f}"
                f" logit={image_logits[index]:.6f}"
            )


def main() -> None:
    args = parse_args()
    labels = read_labels(args.labels)
    session = ort.InferenceSession(args.model.as_posix(), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    image_paths = [path.resolve() for path in args.images]
    images = [load_image(path) for path in image_paths]

    print(f"model={args.model.resolve()}")
    print(f"input={input_meta.name} shape={input_meta.shape}")
    print(f"output={output_meta.name} shape={output_meta.shape}")
    print(f"labels={len(labels)}")

    for mode in args.preprocess:
        batch = np.stack([preprocess(image, mode) for image in images]).astype(np.float32)
        logits = session.run([output_meta.name], {input_meta.name: batch})[0]
        print_results(image_paths, labels, logits, mode, args.top_k)


if __name__ == "__main__":
    main()

#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "Pillow",
# ]
# ///
"""
Plota bounding boxes nas imagens do dataset COCO gerado pelo DeepStream.
Uso: uv run plot_bbox.py [--dataset PATH] [--images PATH] [--output PATH]
"""

import argparse
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PALETTE = [
    (220, 50,  50),
    (50,  150, 220),
    (50,  200, 100),
    (230, 160,  30),
    (160,  50, 220),
    (30,  200, 200),
    (220, 100, 160),
    (120, 180,  50),
    (200,  80,  30),
    (80,  120, 220),
]


def load_font(size=18):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_bbox(draw, bbox, label, color, font):
    x, y, w, h = bbox
    x2, y2 = x + w, y + h
    draw.rectangle([x, y, x2, y2], outline=color, width=3)

    text_bbox = draw.textbbox((0, 0), label, font=font)
    tw = text_bbox[2] - text_bbox[0]
    th = text_bbox[3] - text_bbox[1]
    pad = 3
    label_y = max(y - th - pad * 2, 0)
    draw.rectangle([x, label_y, x + tw + pad * 2, label_y + th + pad * 2], fill=color)
    draw.text((x + pad, label_y + pad), label, fill=(255, 255, 255), font=font)


def plot_dataset(dataset_path, images_dir, output_dir):
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    categories = {c["id"]: c["name"] for c in dataset["categories"]}
    color_by_cat = {
        cat_id: PALETTE[i % len(PALETTE)]
        for i, cat_id in enumerate(categories)
    }

    anns_by_image = {}
    for ann in dataset["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    os.makedirs(output_dir, exist_ok=True)
    font = load_font(18)
    processed = 0
    skipped = 0

    for img_info in dataset["images"]:
        anns = anns_by_image.get(img_info["id"])
        if not anns:
            continue

        src_path = Path(images_dir) / img_info["file_name"]
        if not src_path.exists():
            print(f"  [AVISO] Imagem não encontrada: {src_path}")
            skipped += 1
            continue

        img = Image.open(src_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        for ann in anns:
            cat_name = categories.get(ann["category_id"], "?")
            score = ann.get("score", 0)
            color = color_by_cat.get(ann["category_id"], (180, 180, 180))
            draw_bbox(draw, ann["bbox"], f"{cat_name} {score:.2f}", color, font)

        # Adiciona legenda com source no canto inferior
        source = img_info.get("source", "")
        if source:
            draw.text((8, img.height - 22), source, fill=(220, 220, 220), font=font)

        out_path = Path(output_dir) / img_info["file_name"]
        img.save(out_path, quality=95)
        processed += 1

    print(f"Imagens plotadas : {processed}")
    print(f"Imagens ausentes : {skipped}")
    print(f"Salvo em         : {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Plota bboxes do dataset COCO nas imagens")
    parser.add_argument("--dataset", default="dataset_output/annotations.json")
    parser.add_argument("--images",  default="dataset_output/images")
    parser.add_argument("--output",  default="dataset_output/images_bbox")
    args = parser.parse_args()

    if not Path(args.dataset).exists():
        print(f"Erro: dataset não encontrado em '{args.dataset}'")
        raise SystemExit(1)

    print(f"Dataset : {args.dataset}")
    print(f"Imagens : {args.images}")
    print(f"Saída   : {args.output}")
    plot_dataset(args.dataset, args.images, args.output)


if __name__ == "__main__":
    main()

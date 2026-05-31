#!/usr/bin/env python3
"""
Gera um vídeo de validação com bounding boxes sobrepostas nas imagens do dataset.

Uso
---
python visualize.py output/           # lê output/annotations.coco.json e output/images/
python visualize.py output/ --fps 4
python visualize.py output/ --out validacao.mp4
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# Paleta de cores por categoria (BGR)
_PALETTE = [
    (0, 200, 255),
    (0, 255, 100),
    (255, 80,  0),
    (180,  0, 255),
    (255, 220,  0),
    (0, 100, 255),
]


def _color(cat_id: int) -> tuple[int, int, int]:
    return _PALETTE[cat_id % len(_PALETTE)]


def _draw(img: np.ndarray, anns: list[dict], catid2name: dict[int, str]) -> np.ndarray:
    """Desenha bounding boxes e labels sobre uma cópia do frame.

    Parâmetros
    ----------
    img : np.ndarray
        Frame BGR (formato OpenCV).
    anns : list[dict]
        Anotações COCO da imagem; cada item deve ter ``category_id``, ``bbox``
        (formato XYWH) e, opcionalmente, ``score``.
    catid2name : dict[int, str]
        Mapa de ID de categoria para nome legível.

    Retorna
    -------
    np.ndarray
        Cópia do frame com as bounding boxes desenhadas.
    """
    out = img.copy()
    h, w = out.shape[:2]
    font_scale = max(0.4, min(w, h) / 1000)
    thickness = max(1, round(min(w, h) / 400))

    for ann in anns:
        cat_id = ann["category_id"]
        x, y, bw, bh = ann["bbox"]  # COCO: x, y, w, h
        x1, y1, x2, y2 = int(x), int(y), int(x + bw), int(y + bh)

        color = _color(cat_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness + 1)

        name = catid2name.get(cat_id, str(cat_id))
        score = ann.get("score")
        label = f"{name} {score:.2f}" if score is not None else name

        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        ty = max(y1 - 4, th + baseline)
        cv2.rectangle(out, (x1, ty - th - baseline), (x1 + tw + 2, ty + 2), color, -1)
        cv2.putText(
            out, label, (x1 + 1, ty),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness
        )

    return out


def _encode_with_ffmpeg(frames_dir: Path, fps: float, out_path: Path) -> Path:
    """
    Usa o ffmpeg do sistema para encodar frames JPEG em vídeo.
    Testa codec+container em ordem de compatibilidade e retorna o caminho criado.
    """
    candidates = [
        # (codec, extensão, extra_args)
        ("mpeg4",    ".avi", []),
        ("libxvid",  ".avi", []),
        ("libvpx",   ".webm", ["-b:v", "2M"]),
        ("libx264",  ".mp4", ["-pix_fmt", "yuv420p"]),
        ("mpeg4",    ".mp4", ["-pix_fmt", "yuv420p"]),
    ]
    last_err = ""
    for codec, ext, extra in candidates:
        path = out_path.with_suffix(ext)
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-pattern_type", "glob",
            "-i", str(frames_dir / "*.jpg"),
            "-c:v", codec,
            *extra,
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Codec: {codec}  container: {ext}")
            return path
        last_err = result.stderr
    raise RuntimeError(f"Todos os codecs ffmpeg falharam. Último erro:\n{last_err[-800:]}")


def build_video(dataset_dir: Path, fps: float, out_path: Path) -> None:
    """Renderiza um vídeo de validação com as bounding boxes do dataset sobrepostas.

    Lê ``annotations.coco.json`` e as imagens em ``images/`` dentro de
    ``dataset_dir``. Tenta encodar com ffmpeg (preferido); se não estiver
    disponível, faz fallback para ``cv2.VideoWriter`` testando codecs em ordem.
    O caminho final do vídeo pode ter extensão diferente de ``out_path`` caso
    o codec selecionado exija outro container.

    Parâmetros
    ----------
    dataset_dir : Path
        Pasta do dataset com a estrutura ``images/`` + ``annotations.coco.json``.
    fps : float
        Taxa de frames do vídeo de saída.
    out_path : Path
        Caminho base de saída. A extensão pode ser ajustada conforme o codec
        escolhido (ex: ``.mp4`` → ``.avi`` se libx264 não estiver disponível).
    """
    ann_file = dataset_dir / "annotations.coco.json"
    images_dir = dataset_dir / "images"

    with ann_file.open(encoding="utf-8") as f:
        coco = json.load(f)

    catid2name = {cat["id"]: cat["name"] for cat in coco["categories"]}

    img_anns: dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    images = sorted(coco["images"], key=lambda x: x["id"])
    if not images:
        raise ValueError("Nenhuma imagem encontrada no JSON.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    use_ffmpeg = shutil.which("ffmpeg") is not None

    total = len(images)
    writer = None
    tmp_dir = None

    if use_ffmpeg:
        tmp_dir = Path(tempfile.mkdtemp(prefix="visualize_"))
        print(f"Renderizando {total} frames em {tmp_dir} …")
    else:
        # fallback: tenta cv2.VideoWriter
        first = cv2.imread(str(images_dir / images[0]["file_name"]))
        if first is None:
            raise FileNotFoundError(f"Imagem não encontrada: {images_dir / images[0]['file_name']}")
        h, w = first.shape[:2]
        for fourcc_str, suffix in [("avc1", ".mp4"), ("mp4v", ".mp4"), ("XVID", ".avi")]:
            candidate = out_path.with_suffix(suffix)
            w_ = cv2.VideoWriter(str(candidate), cv2.VideoWriter_fourcc(*fourcc_str), fps, (w, h))
            if w_.isOpened():
                writer = w_
                out_path = candidate
                print(f"Codec {fourcc_str}  →  {out_path}")
                break
            w_.release()
            candidate.unlink(missing_ok=True)  # remove stub criado antes de isOpened() retornar False
        if writer is None:
            raise RuntimeError("ffmpeg não encontrado e nenhum codec cv2 disponível.")

    try:
        for i, img_meta in enumerate(images, 1):
            path = images_dir / img_meta["file_name"]
            frame = cv2.imread(str(path))
            if frame is None:
                print(f"  aviso: imagem não encontrada, pulando — {path}")
                continue

            frame = _draw(frame, img_anns[img_meta["id"]], catid2name)

            counter = f"{i}/{total}  {img_meta['file_name']}"
            cv2.putText(frame, counter, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(frame, counter, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

            if use_ffmpeg:
                cv2.imwrite(str(tmp_dir / f"{i:08d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                writer.write(frame)

            if i % 20 == 0 or i == total:
                print(f"  {i}/{total} frames processados")

        if use_ffmpeg:
            print("Encodando vídeo com ffmpeg …")
            out_path = _encode_with_ffmpeg(tmp_dir, fps, out_path)
        else:
            writer.release()
    finally:
        if writer is not None:
            writer.release()
        if tmp_dir and tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"O arquivo de vídeo não foi criado em: {out_path}")

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nVídeo salvo em: {out_path}  ({total} frames, {fps} FPS, {size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualiza bounding boxes do dataset COCO em vídeo")
    parser.add_argument("dataset_dir", type=Path, help="Pasta com images/ e annotations.coco.json")
    parser.add_argument("--fps", type=float, default=2.0, help="FPS do vídeo de saída (default: 2)")
    parser.add_argument("--out", type=Path, default=None, help="Caminho do vídeo de saída")
    args = parser.parse_args()

    out = args.out or args.dataset_dir / "validacao.mp4"
    build_video(args.dataset_dir, args.fps, out)


if __name__ == "__main__":
    main()

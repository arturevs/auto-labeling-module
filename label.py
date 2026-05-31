#!/usr/bin/env python3
"""
Auto-rotulagem de vídeos ou pastas de imagens com RT-DETR (placa_v1).

Exemplos
--------
# Vídeo — 2 FPS, GPU
python label.py video.mp4 output/dataset/

# Vídeo — 1 FPS, desativar filtragem de duplicatas
python label.py video.mp4 output/dataset/ --fps 1 --no-dedup

# Pasta de imagens — CPU
python label.py imagens/ output/dataset/ --device cpu

# Especificar limiar de similaridade (menor = mais frames aceitos)
python label.py video.mp4 output/ --fps 3 --max-distance 5

# Pesos e config customizados
python label.py video.mp4 output/ --weights /caminho/modelo.pdparams --config /caminho/rtdetr.yml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from auto_labeling import AutoLabeler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-rotulagem com RT-DETR → dataset COCO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("input", type=Path, help="Vídeo ou pasta de imagens")
    parser.add_argument("output", type=Path, help="Pasta de saída do dataset COCO")

    model_group = parser.add_argument_group("modelo")
    model_group.add_argument(
        "--paddle-repo",
        type=Path,
        default=None,
        metavar="PATH",
        help="Raiz do repositório treinamento-paddle-rtdetr (default: ../treinamento-paddle-rtdetr)",
    )
    model_group.add_argument("--config", type=Path, default=None, help="Caminho para rtdetr.yml")
    model_group.add_argument("--weights", type=Path, default=None, help="Caminho para best_model.pdparams")
    model_group.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score mínimo para aceitar uma detecção (default: 0.5)",
    )
    model_group.add_argument(
        "--device",
        default="gpu",
        choices=["gpu", "cpu"],
        help="Dispositivo de inferência (default: gpu)",
    )

    video_group = parser.add_argument_group("vídeo")
    video_group.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames por segundo a inferir do vídeo (default: 1.0)",
    )
    video_group.add_argument(
        "--max-distance",
        type=int,
        default=10,
        metavar="N",
        help=(
            "Distância máxima de Hamming (dhash 64-bit) para considerar frames "
            "duplicados. Intervalo: 0–64. Menor = mais estrito. (default: 10)"
        ),
    )
    video_group.add_argument(
        "--no-dedup",
        action="store_true",
        help="Desativa a filtragem de duplicatas para vídeos",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    labeler = AutoLabeler(
        paddle_repo=args.paddle_repo,
        config_path=args.config,
        weights_path=args.weights,
        threshold=args.threshold,
        device=args.device,
        max_distance=None if args.no_dedup else args.max_distance,
    )

    labeler.label(input_path=args.input, output_dir=args.output, fps=args.fps)


if __name__ == "__main__":
    main()

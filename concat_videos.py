#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "imageio-ffmpeg",
# ]
# ///

import re
import subprocess
import sys
from pathlib import Path


def get_video_info(ffmpeg, path):
    result = subprocess.run(
        [ffmpeg, "-i", str(path)],
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
    )
    stderr = result.stderr.decode("utf-8", errors="replace")

    fps = None
    for pattern in [r"(\d+(?:\.\d+)?) fps", r"(\d+(?:\.\d+)?) tbr"]:
        m = re.search(pattern, stderr)
        if m:
            fps = float(m.group(1))
            break
    if fps is None:
        fps = 30.0

    # Match resolutions like 1920x1080, but avoid matching timestamps (00:00:00)
    m = re.search(r"\b(\d{2,5})x(\d{2,5})\b", stderr)
    width, height = (1280, 720) if not m else (int(m.group(1)), int(m.group(2)))

    return fps, width, height


def concat_videos(videos_dir="c_videos", output="output_concat.mp4", black_seconds=2):
    videos = sorted(Path(videos_dir).glob("*.mp4"))
    if not videos:
        print(f"Nenhum arquivo MP4 encontrado em '{videos_dir}'")
        sys.exit(1)

    from imageio_ffmpeg import get_ffmpeg_exe
    ffmpeg = get_ffmpeg_exe()

    print(f"Encontrados {len(videos)} vídeo(s):")
    for v in videos:
        print(f"  {v.name}")

    fps, width, height = get_video_info(ffmpeg, videos[0])
    print(f"\nReferência: {width}x{height} @ {fps} fps")

    n = len(videos)
    n_black = n - 1

    if n == 1:
        subprocess.run([ffmpeg, "-y", "-i", str(videos[0]), "-c", "copy", output], check=True)
        print(f"\nApenas 1 vídeo — copiado para {output}")
        return

    cmd = [ffmpeg, "-y"]
    for v in videos:
        cmd += ["-i", str(v)]

    filter_parts = []

    # Normaliza todos os vídeos para mesma resolução/fps/SAR
    for i in range(n):
        filter_parts.append(
            f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}[v{i}]"
        )

    # Gera fonte de tela preta
    filter_parts.append(
        f"color=black:s={width}x{height}:r={fps}:d={black_seconds}[bsrc]"
    )

    # Split da fonte preta em N-1 segmentos
    if n_black == 1:
        black_labels = ["bsrc"]
    else:
        black_labels = [f"b{i}" for i in range(n_black)]
        splits = "".join(f"[{lbl}]" for lbl in black_labels)
        filter_parts.append(f"[bsrc]split={n_black}{splits}")

    # Monta sequência para concat: v0, b0, v1, b1, ..., v(n-1)
    concat_inputs = ""
    for i in range(n):
        concat_inputs += f"[v{i}]"
        if i < n_black:
            concat_inputs += f"[{black_labels[i]}]"

    total_segments = n + n_black
    filter_parts.append(f"{concat_inputs}concat=n={total_segments}:v=1:a=0[out]")

    filter_complex = "; ".join(filter_parts)

    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-r", str(fps),
        output,
    ]

    print(f"\nConcatenando {n} vídeos com {black_seconds}s de tela preta entre eles...")
    subprocess.run(cmd, check=True)
    print(f"\nConcluído! Arquivo gerado: {output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Concatena vídeos MP4 de uma pasta com tela preta entre eles"
    )
    parser.add_argument(
        "--dir", default="c_videos", help="Pasta com os MP4s (padrão: c_videos)"
    )
    parser.add_argument(
        "--output", default="output_concat.mp4", help="Arquivo de saída (padrão: output_concat.mp4)"
    )
    parser.add_argument(
        "--black", type=float, default=2.0, help="Duração da tela preta em segundos (padrão: 2)"
    )
    args = parser.parse_args()

    concat_videos(args.dir, args.output, args.black)

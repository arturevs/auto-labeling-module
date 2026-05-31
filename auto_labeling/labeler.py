from __future__ import annotations

import sys
from pathlib import Path

import cv2
from PIL import Image

from .coco_builder import COCOBuilder
from .dedup import FrameDeduplicator

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _setup_paddle_path(paddle_repo: Path) -> None:
    for p in [str(paddle_repo / "tools"), str(paddle_repo)]:
        if p not in sys.path:
            sys.path.insert(0, p)


def _load_predictor(
    paddle_repo: Path,
    config_path: Path | None,
    weights_path: Path | None,
    threshold: float,
    device: str,
):
    _setup_paddle_path(paddle_repo)
    from predictor import RTDETRPredictor  # noqa: PLC0415

    kwargs: dict = {"threshold": threshold, "device": device}
    if config_path is not None:
        kwargs["config_path"] = str(config_path)
    if weights_path is not None:
        kwargs["weights_path"] = str(weights_path)
    return RTDETRPredictor(**kwargs)


def _coco_categories(predictor) -> list[dict]:
    return [
        {"id": cat_id, "name": name, "supercategory": ""}
        for cat_id, name in sorted(predictor._catid2name.items())
    ]


class AutoLabeler:
    """
    Recebe um vídeo ou pasta de imagens, realiza inferência com RT-DETR (placa_v1)
    e gera um dataset no formato COCO.

    Parâmetros
    ----------
    paddle_repo : Path | None
        Raiz do repositório treinamento-paddle-rtdetr.
        Default: ../treinamento-paddle-rtdetr relativo a este módulo.
    config_path : Path | None
        Caminho para o rtdetr.yml. None usa o default do predictor.
    weights_path : Path | None
        Caminho para best_model.pdparams. None usa o default do predictor.
    threshold : float
        Score mínimo para aceitar uma detecção.
    device : str
        'gpu' ou 'cpu'.
    max_distance : int | None
        Limiar de Hamming para filtrar duplicatas em vídeo.
        None desativa a filtragem.
    """

    def __init__(
        self,
        paddle_repo: Path | None = None,
        config_path: Path | None = None,
        weights_path: Path | None = None,
        threshold: float = 0.5,
        device: str = "gpu",
        max_distance: int | None = 10,
    ) -> None:
        _module_root = Path(__file__).resolve().parent.parent

        if paddle_repo is None:
            paddle_repo = _module_root.parent / "treinamento-paddle-rtdetr"

        paddle_repo = Path(paddle_repo).resolve()
        if not paddle_repo.exists():
            raise FileNotFoundError(
                f"Repositório Paddle não encontrado: {paddle_repo}\n"
                "Use --paddle-repo para especificar o caminho."
            )

        if weights_path is None:
            candidate = _module_root / "best_model.pdparams"
            if candidate.exists():
                weights_path = candidate

        print("Carregando modelo...")
        self._predictor = _load_predictor(paddle_repo, config_path, weights_path, threshold, device)
        self._categories = _coco_categories(self._predictor)
        self._max_distance = max_distance
        print(f"Modelo carregado. Categorias: {[c['name'] for c in self._categories]}")

    def label(self, input_path: Path, output_dir: Path, fps: float = 1.0) -> None:
        """
        Rotula automaticamente um vídeo ou pasta de imagens.

        Parâmetros
        ----------
        input_path : Path
            Vídeo (ex: .mp4, .avi) ou pasta com imagens.
        output_dir : Path
            Pasta de saída. Será criada se não existir.
            Estrutura gerada:
                output_dir/
                ├── images/
                │   └── *.jpg
                └── annotations.coco.json
        fps : float
            Para vídeos: quantos frames por segundo inferir. Ignorado para pastas.
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)

        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        builder = COCOBuilder(self._categories)

        if input_path.is_file():
            skipped_dedup = self._label_video(input_path, images_dir, builder, fps)
            extra = f", duplicatas ignoradas: {skipped_dedup}"
        elif input_path.is_dir():
            self._label_image_folder(input_path, images_dir, builder)
            extra = ""
        else:
            raise ValueError(f"Input não encontrado: {input_path}")

        builder.save(output_dir / "annotations.coco.json")
        print(f"\nConcluído — imagens: {builder.num_images}, anotações: {builder.num_annotations}{extra}")
        print(f"Dataset salvo em: {output_dir}")

    # ------------------------------------------------------------------
    # Vídeo
    # ------------------------------------------------------------------

    def _label_video(
        self, video_path: Path, images_dir: Path, builder: COCOBuilder, fps: float
    ) -> int:
        """Extrai frames do vídeo, filtra duplicatas e executa inferência em cada frame aceito.

        Parâmetros
        ----------
        video_path : Path
            Arquivo de vídeo de entrada.
        images_dir : Path
            Pasta de destino onde os frames aceitos são salvos como JPEG.
        builder : COCOBuilder
            Instância que acumula as anotações geradas.
        fps : float
            Taxa de extração desejada. Frames intermediários são pulados
            para aproximar essa taxa à FPS real do vídeo.

        Retorna
        -------
        int
            Número de frames descartados pela filtragem de duplicatas.
        """
        dedup = FrameDeduplicator(max_distance=self._max_distance) if self._max_distance is not None else None

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Não foi possível abrir o vídeo: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_interval = max(1, round(video_fps / fps))

        print(
            f"Vídeo: {video_fps:.1f} FPS, ~{total_frames} frames → "
            f"extraindo a cada {frame_interval} frames (alvo: {fps} FPS)"
        )
        if dedup is not None:
            print(f"Filtragem de duplicatas ativa (max_distance={self._max_distance})")

        frame_idx = 0
        accepted = 0
        skipped_dedup = 0

        try:
            while True:
                ret, bgr = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval != 0:
                    frame_idx += 1
                    continue

                pil_image = Image.fromarray(bgr[..., ::-1])  # BGR → RGB

                if dedup is not None and dedup.is_duplicate(pil_image):
                    skipped_dedup += 1
                    frame_idx += 1
                    continue

                accepted += 1
                file_name = f"{accepted:08d}.jpg"
                img_path = images_dir / file_name
                pil_image.save(img_path, quality=95)

                h, w = bgr.shape[:2]
                img_id = builder.add_image(file_name, w, h)

                detections = self._predictor.predict(str(img_path))
                for det in detections:
                    cat_id = self._predictor._clsid2catid[det["class_id"]]
                    builder.add_detection(img_id, cat_id, det["bbox"], det["score"])

                frame_idx += 1

                if accepted % 50 == 0:
                    print(f"  aceitos={accepted}  duplicatas_ignoradas={skipped_dedup}")
        finally:
            cap.release()

        return skipped_dedup

    # ------------------------------------------------------------------
    # Pasta de imagens
    # ------------------------------------------------------------------

    def _label_image_folder(self, folder: Path, images_dir: Path, builder: COCOBuilder) -> None:
        """Processa todos os arquivos de imagem de uma pasta e executa inferência em cada um.

        Imagens não-JPEG são convertidas para JPEG (qualidade 95) antes da inferência,
        garantindo formato consistente com o caminho de vídeo e compatibilidade com
        o predictor.

        Parâmetros
        ----------
        folder : Path
            Pasta de origem. Extensões suportadas: ``.jpg``, ``.jpeg``, ``.png``,
            ``.bmp``, ``.webp``, ``.tif``, ``.tiff``.
        images_dir : Path
            Pasta de destino onde as imagens (convertidas) são salvas.
        builder : COCOBuilder
            Instância que acumula as anotações geradas.
        """
        image_files = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in _IMAGE_EXTENSIONS
        )

        if not image_files:
            raise ValueError(f"Nenhuma imagem encontrada em: {folder}")

        print(f"Pasta: {len(image_files)} imagens")

        for i, img_path in enumerate(image_files, 1):
            # Sempre salva como JPEG — garante formato consistente para o predictor
            dest = images_dir / (img_path.stem + ".jpg")
            with Image.open(img_path) as pil_image:
                w, h = pil_image.size
                if img_path.resolve() != dest.resolve():
                    pil_image.convert("RGB").save(dest, quality=95)

            img_id = builder.add_image(dest.name, w, h)

            detections = self._predictor.predict(str(dest))
            for det in detections:
                cat_id = self._predictor._clsid2catid[det["class_id"]]
                builder.add_detection(img_id, cat_id, det["bbox"], det["score"])

            if i % 50 == 0 or i == len(image_files):
                print(f"  {i}/{len(image_files)} imagens processadas")

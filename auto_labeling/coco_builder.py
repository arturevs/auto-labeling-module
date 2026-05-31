from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class COCOBuilder:
    """Constrói um dataset no formato COCO incrementalmente.

    Uso típico::

        builder = COCOBuilder(categories)
        for frame in frames:
            img_id = builder.add_image(frame.name, frame.width, frame.height)
            for det in detections:
                builder.add_detection(img_id, det.category_id, det.bbox_xyxy, det.score)
        builder.save(Path("annotations.coco.json"))

    Parâmetros
    ----------
    categories : list[dict]
        Lista de dicionários ``{"id": int, "name": str, "supercategory": str}``
        no formato COCO. Normalmente obtida de ``_coco_categories(predictor)``.
    """

    def __init__(self, categories: list[dict]) -> None:
        self._categories = categories
        self._images: list[dict] = []
        self._annotations: list[dict] = []
        self._next_img_id = 1
        self._next_ann_id = 1

    def add_image(self, file_name: str, width: int, height: int) -> int:
        """Registra uma imagem no dataset e retorna seu ID.

        Parâmetros
        ----------
        file_name : str
            Nome do arquivo relativo à pasta ``images/`` (ex: ``00000001.jpg``).
        width : int
            Largura em pixels.
        height : int
            Altura em pixels.

        Retorna
        -------
        int
            ID atribuído à imagem (usado em :meth:`add_detection`).
        """
        img_id = self._next_img_id
        self._images.append({
            "id": img_id,
            "file_name": file_name,
            "width": width,
            "height": height,
        })
        self._next_img_id += 1
        return img_id

    def add_detection(
        self,
        image_id: int,
        category_id: int,
        bbox_xyxy: list[float],
        score: float | None = None,
    ) -> None:
        """Adiciona uma detecção ao dataset.

        Converte o bounding box de XYXY (saída do predictor) para XYWH
        (formato COCO) e armazena a anotação.

        Parâmetros
        ----------
        image_id : int
            ID retornado por :meth:`add_image`.
        category_id : int
            ID de categoria conforme a lista ``categories`` passada ao construtor.
        bbox_xyxy : list[float]
            Bounding box no formato ``[x1, y1, x2, y2]`` em pixels.
        score : float | None
            Confiança da detecção (0–1). Incluído no JSON quando fornecido;
            omitido para datasets de ground-truth puro.
        """
        x1, y1, x2, y2 = bbox_xyxy
        w, h = x2 - x1, y2 - y1
        ann: dict = {
            "id": self._next_ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
            "area": round(w * h, 2),
            "iscrowd": 0,
            "segmentation": [],
        }
        if score is not None:
            ann["score"] = round(score, 4)
        self._annotations.append(ann)
        self._next_ann_id += 1

    @property
    def num_images(self) -> int:
        """Número de imagens registradas até o momento."""
        return len(self._images)

    @property
    def num_annotations(self) -> int:
        """Número de anotações registradas até o momento."""
        return len(self._annotations)

    def build(self) -> dict:
        """Retorna o dataset completo como dicionário no formato COCO.

        O dicionário contém as chaves padrão: ``info``, ``licenses``,
        ``categories``, ``images`` e ``annotations``.
        """
        return {
            "info": {
                "description": "Auto-labeled dataset (RT-DETR placa_v1)",
                "date_created": datetime.now().isoformat(),
            },
            "licenses": [],
            "categories": self._categories,
            "images": self._images,
            "annotations": self._annotations,
        }

    def save(self, path: Path) -> None:
        """Serializa o dataset para um arquivo JSON.

        Cria os diretórios pai se não existirem.

        Parâmetros
        ----------
        path : Path
            Destino do arquivo (ex: ``output/annotations.coco.json``).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.build(), f, ensure_ascii=False, indent=2)

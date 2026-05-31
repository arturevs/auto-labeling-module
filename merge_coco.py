#!/usr/bin/env python3
"""
Fusão de datasets no formato COCO.

Combina múltiplos datasets gerados pelo pipeline de rotulagem automática
(ou revisados pelo Relabeler) em um único ``annotations.coco.json``.

Cada entrada pode ser:
  - Diretório contendo ``annotations.coco.json`` e ``images/``
  - Caminho direto para um arquivo ``.json`` no formato COCO

Conflitos de ID (imagens, anotações) são resolvidos por remapeamento global.
Conflitos de ``file_name`` são resolvidos prefixando com o nome da fonte
(ex: ``ds0_00000001.jpg``). Categorias são fundidas pelo nome — IDs iguais
em datasets diferentes não implicam a mesma categoria.

Exemplos
--------
# Fundir dois diretórios de dataset
python merge_coco.py output/sessao1/ output/sessao2/ --output merged/

# Fundir JSONs diretamente, sem copiar imagens
python merge_coco.py sessao1.json sessao2.json sessao3.json --output merged/ --no-copy-images

# Especificar prefixos personalizados para cada fonte
python merge_coco.py output/a/ output/b/ --output merged/ --prefixes cam1 cam2
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


# ── Resolução de entradas ──────────────────────────────────────────────────────

def _resolve_input(path: Path) -> tuple[Path, Path | None]:
    """Retorna (json_path, images_dir_or_None) para uma entrada.

    Aceita diretório (procura ``annotations.coco.json`` dentro) ou JSON direto.
    """
    if path.is_dir():
        json_path = path / "annotations.coco.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"Nenhum 'annotations.coco.json' encontrado em: {path}"
            )
        images_dir = path / "images"
        return json_path, images_dir if images_dir.is_dir() else None
    if path.suffix.lower() == ".json":
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
        images_dir = path.parent / "images"
        return path, images_dir if images_dir.is_dir() else None
    raise ValueError(f"Entrada deve ser um diretório ou arquivo .json: {path}")


# ── Fusão ──────────────────────────────────────────────────────────────────────

def merge(
    inputs: list[Path],
    output_dir: Path,
    copy_images: bool = True,
    prefixes: list[str] | None = None,
) -> None:
    """Funde múltiplos datasets COCO em ``output_dir``.

    Parâmetros
    ----------
    inputs : list[Path]
        Caminhos para os datasets de entrada (diretórios ou JSONs).
    output_dir : Path
        Pasta de saída. Criada se não existir.
        Estrutura gerada::

            output_dir/
            ├── images/          ← todas as imagens (se copy_images=True)
            └── annotations.coco.json

    copy_images : bool
        Se True, copia as imagens para ``output_dir/images/``.
    prefixes : list[str] | None
        Prefixos personalizados para cada fonte (mesmo comprimento que ``inputs``).
        Se None, usa ``ds0``, ``ds1``, … automaticamente.
    """
    if prefixes is not None and len(prefixes) != len(inputs):
        raise ValueError("--prefixes deve ter o mesmo número de entradas que os inputs")

    output_dir = output_dir.resolve()
    images_out = output_dir / "images"
    if copy_images:
        images_out.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Mapa global: nome_categoria → id_global
    cat_name_to_id: dict[str, int] = {}
    next_cat_id = 1

    merged_images: list[dict] = []
    merged_annotations: list[dict] = []
    next_img_id = 1
    next_ann_id = 1

    for idx, input_path in enumerate(inputs):
        prefix = prefixes[idx] if prefixes else f"ds{idx}"
        json_path, images_src = _resolve_input(input_path)

        with json_path.open(encoding="utf-8") as f:
            coco = json.load(f)

        # ── Categorias: fundir por nome ────────────────────────────────────
        # old_cat_id (neste dataset) → new_cat_id (global)
        cat_id_remap: dict[int, int] = {}
        for cat in coco.get("categories", []):
            name = cat["name"]
            if name not in cat_name_to_id:
                cat_name_to_id[name] = next_cat_id
                next_cat_id += 1
            cat_id_remap[cat["id"]] = cat_name_to_id[name]

        # ── Imagens: remap de IDs + prefixar file_name ────────────────────
        # old_img_id → new_img_id
        img_id_remap: dict[int, int] = {}
        for img in coco.get("images", []):
            new_id = next_img_id
            img_id_remap[img["id"]] = new_id
            next_img_id += 1

            new_file_name = f"{prefix}_{img['file_name']}"
            merged_images.append({
                **img,
                "id": new_id,
                "file_name": new_file_name,
            })

            if copy_images and images_src is not None:
                src = images_src / img["file_name"]
                dst = images_out / new_file_name
                if src.exists():
                    shutil.copy2(src, dst)
                else:
                    print(f"  [aviso] imagem não encontrada, pulando: {src}")

        # ── Anotações: remap de IDs ────────────────────────────────────────
        for ann in coco.get("annotations", []):
            new_ann: dict = {
                **ann,
                "id": next_ann_id,
                "image_id": img_id_remap[ann["image_id"]],
                "category_id": cat_id_remap[ann["category_id"]],
            }
            merged_annotations.append(new_ann)
            next_ann_id += 1

        n_imgs = len(coco.get("images", []))
        n_anns = len(coco.get("annotations", []))
        print(f"  [{prefix}] {json_path} — {n_imgs} imagens, {n_anns} anotações")

    # ── Montar categorias na ordem dos IDs globais ─────────────────────────
    merged_categories = [
        {"id": gid, "name": name, "supercategory": ""}
        for name, gid in sorted(cat_name_to_id.items(), key=lambda x: x[1])
    ]

    merged_coco = {
        "info": {"description": "Dataset fundido pelo merge_coco.py"},
        "licenses": [],
        "categories": merged_categories,
        "images": merged_images,
        "annotations": merged_annotations,
    }

    out_json = output_dir / "annotations.coco.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(merged_coco, f, ensure_ascii=False, indent=2)

    print(
        f"\nFundido — imagens: {len(merged_images)}, "
        f"anotações: {len(merged_annotations)}, "
        f"categorias: {[c['name'] for c in merged_categories]}"
    )
    print(f"Salvo em: {out_json}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Funde múltiplos datasets COCO em um único arquivo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        metavar="ENTRADA",
        help="Diretórios de dataset ou arquivos .json no formato COCO.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        metavar="SAÍDA",
        help="Pasta de saída do dataset fundido.",
    )
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Não copiar imagens para a pasta de saída (apenas fundir os JSONs).",
    )
    parser.add_argument(
        "--prefixes",
        nargs="+",
        metavar="PREFIXO",
        help=(
            "Prefixo para os file_names de cada fonte (mesmo número que as entradas). "
            "Padrão: ds0, ds1, …"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"Fundindo {len(args.inputs)} dataset(s)…")
    merge(
        inputs=args.inputs,
        output_dir=args.output,
        copy_images=not args.no_copy_images,
        prefixes=args.prefixes,
    )


if __name__ == "__main__":
    main()

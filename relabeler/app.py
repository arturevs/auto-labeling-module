#!/usr/bin/env python3
"""
Relabeler — revisão manual de inferências COCO.

Uso:
    cd auto-labeling-module
    streamlit run relabeler/app.py
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

LABELS_DIR = Path(__file__).parent / "labels"


# ── Dados ─────────────────────────────────────────────────────────────────────

@st.cache_data
def load_labels() -> list[tuple[str, str]]:
    """Carrega as labels disponíveis em ``LABELS_DIR``.

    Retorna lista de ``(nome, caminho_absoluto_str)`` ordenada pelo nome,
    onde cada nome é o stem do arquivo PNG (ex: ``carro.png`` → ``"carro"``).
    Retorna ``[]`` se o diretório não existir.
    """
    if not LABELS_DIR.is_dir():
        return []
    return sorted(
        [(p.stem, str(p)) for p in LABELS_DIR.glob("*.png")],
        key=lambda x: x[0],
    )


@st.cache_data
def load_coco(path: str) -> dict:
    """Carrega e armazena em cache o arquivo COCO de entrada.

    O cache é mantido enquanto ``path`` não mudar. Como o arquivo de entrada
    nunca é modificado pelo Relabeler, o cache permanece válido por toda a sessão.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_output(path: Path) -> dict | None:
    """Lê o arquivo de saída relabelado do disco, sem cache.

    Retorna o dicionário COCO ou ``None`` se o arquivo ainda não existir.
    Sempre lê do disco para refletir confirmações feitas na sessão atual.
    """
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def save_output(data: dict, path: Path) -> None:
    """Persiste o dicionário COCO de saída em disco.

    Cria os diretórios pai se necessário. Chamada a cada confirmação para
    garantir que o progresso não se perca em caso de encerramento inesperado.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def make_empty_output(labels: list[tuple[str, str]], source_coco: str) -> dict:
    """Cria um dicionário COCO vazio pronto para receber anotações confirmadas.

    As categorias são derivadas das labels disponíveis em ``LABELS_DIR``,
    com IDs sequenciais a partir de 1. O campo ``source_coco`` no ``info``
    registra o arquivo de entrada, impedindo que sessões distintas escrevam
    no mesmo arquivo de saída.

    Parâmetros
    ----------
    labels : list[tuple[str, str]]
        Lista ``(nome, caminho)`` retornada por :func:`load_labels`.
    source_coco : str
        Caminho do arquivo COCO de entrada (armazenado para validação).
    """
    return {
        "info": {
            "description": "Relabeled dataset",
            "date_created": datetime.now().isoformat(),
            "source_coco": source_coco,
        },
        "licenses": [],
        "categories": [
            {"id": i + 1, "name": name, "supercategory": ""}
            for i, (name, _) in enumerate(labels)
        ],
        "images": [],
        "annotations": [],
    }


def crop_bbox(img: Image.Image, bbox: list[float], padding: float = 0.15) -> Image.Image:
    """Recorta a região de uma bounding box com margem proporcional.

    Parâmetros
    ----------
    img : PIL.Image.Image
        Imagem original.
    bbox : list[float]
        Bounding box no formato COCO ``[x, y, largura, altura]`` em pixels.
    padding : float
        Fração da largura/altura a adicionar como margem em cada lado.
        Padrão 0.15 = 15 % de cada dimensão.

    Retorna
    -------
    PIL.Image.Image
        Crop da região, limitado às bordas da imagem.
    """
    x, y, w, h = bbox
    pad_x = w * padding
    pad_y = h * padding
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img.width,  x + w + pad_x)
    y2 = min(img.height, y + h + pad_y)
    return img.crop((x1, y1, x2, y2))


def export_dataset(out_path: Path, images_dir: Path) -> None:
    """Exporta o dataset final para ``<out_path.parent>/dataset/``.

    Cria a estrutura ``dataset/images/`` + ``dataset/annotations.coco.json``
    copiando **apenas** as imagens cujas anotações foram confirmadas.
    Chamada como ``on_click`` do botão de download — executa antes de o
    navegador receber o arquivo.

    Parâmetros
    ----------
    out_path : Path
        Arquivo ``_relabeled.json`` gerado pelo Relabeler.
    images_dir : Path
        Pasta de origem das imagens (``output/images/``).
    """
    output = load_output(out_path)
    if output is None:
        return

    dataset_dir = out_path.parent / "dataset"
    ds_images_dir = dataset_dir / "images"
    ds_images_dir.mkdir(parents=True, exist_ok=True)

    confirmed_names = {img["file_name"] for img in output["images"]}
    for name in confirmed_names:
        src = images_dir / name
        if src.exists():
            shutil.copy2(src, ds_images_dir / name)

    with (dataset_dir / "annotations.coco.json").open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


# ── App ───────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Relabeler", layout="wide")

    labels = load_labels()
    if not labels:
        st.error(
            f"Nenhuma label encontrada em `{LABELS_DIR}`. "
            "Crie o diretório e adicione imagens .png com o nome de cada classe."
        )
        st.stop()

    label2catid = {name: i + 1 for i, (name, _) in enumerate(labels)}

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Entrada")
        coco_input = st.text_input(
            "Arquivo COCO",
            value="output/annotations.coco.json",
            help="Caminho relativo ao diretório onde você executa o streamlit",
        )
        images_input = st.text_input("Pasta de imagens", value="output/images")

        coco_path   = Path(coco_input)
        images_dir  = Path(images_input)
        out_path    = coco_path.parent / (coco_path.stem + "_relabeled.json")

        st.divider()
        st.caption(f"Saída: `{out_path}`")
        st.caption(f"Labels disponíveis: **{len(labels)}**")

        if out_path.exists():
            dataset_dir = out_path.parent / "dataset"
            st.download_button(
                "⬇ Baixar & Exportar dataset",
                data=out_path.read_bytes(),
                file_name=out_path.name,
                mime="application/json",
                on_click=export_dataset,
                args=(out_path, images_dir),
            )
            if dataset_dir.exists():
                st.caption(f"Dataset em: `{dataset_dir}`")

    # ── Validação dos caminhos ────────────────────────────────────────────────
    if not coco_path.exists():
        st.warning(f"Arquivo COCO não encontrado: `{coco_path}`")
        st.stop()
    if not images_dir.exists():
        st.warning(f"Pasta de imagens não encontrada: `{images_dir}`")
        st.stop()

    # ── Carrega COCO de entrada ───────────────────────────────────────────────
    coco      = load_coco(str(coco_path))
    imgid2meta = {img["id"]: img for img in coco["images"]}
    all_anns  = coco["annotations"]

    # ── Resume: IDs já confirmados estão no arquivo de saída ─────────────────
    output = load_output(out_path)
    if output is not None:
        stored_source = output.get("info", {}).get("source_coco")
        if stored_source is not None and stored_source != str(coco_path):
            st.warning(
                f"O arquivo `{out_path.name}` foi gerado de outro COCO "
                f"(`{stored_source}`). Escolha um nome de saída diferente "
                "ou remova o arquivo existente."
            )
            st.stop()

    confirmed_ids: set[int] = (
        {ann["id"] for ann in output["annotations"]} if output else set()
    )

    # IDs pulados ficam em session_state (memória da sessão)
    if "skipped_ids" not in st.session_state:
        st.session_state.skipped_ids: set[int] = set()
    if "selected_label" not in st.session_state:
        st.session_state.selected_label: str | None = None
    # Descarta seleção obsoleta caso o diretório de labels tenha mudado
    if st.session_state.selected_label is not None and st.session_state.selected_label not in label2catid:
        st.session_state.selected_label = None

    processed_ids = confirmed_ids | st.session_state.skipped_ids
    pending       = [a for a in all_anns if a["id"] not in processed_ids]

    # ── Barra de progresso ────────────────────────────────────────────────────
    total      = len(all_anns)
    done_count = len(processed_ids)
    st.title("Relabeler")
    st.progress(done_count / total if total else 0)
    st.caption(
        f"**{done_count}/{total}** processadas  •  "
        f"**{len(confirmed_ids)}** confirmadas  •  "
        f"**{len(st.session_state.skipped_ids)}** puladas  •  "
        f"**{len(pending)}** pendentes"
    )

    if not pending:
        st.success("Todas as anotações foram processadas!")
        st.stop()

    # ── Anotação atual ────────────────────────────────────────────────────────
    ann       = pending[0]
    img_meta  = imgid2meta[ann["image_id"]]
    img_path  = images_dir / img_meta["file_name"]
    score     = ann.get("score")
    selected  = st.session_state.selected_label

    st.divider()

    # ── Linha superior: crop + info + botões ─────────────────────────────────
    col_crop, col_info = st.columns([3, 2], gap="large")

    with col_crop:
        if img_path.exists():
            img   = Image.open(img_path)
            crop  = crop_bbox(img, ann["bbox"])
            st.image(crop, use_container_width=True)
        else:
            st.error(f"Imagem não encontrada: `{img_path}`")

    with col_info:
        st.markdown(f"**Imagem:** `{img_meta['file_name']}`")
        st.markdown(f"**Anotação ID:** `{ann['id']}`")
        bbox_r = [round(v) for v in ann["bbox"]]
        st.markdown(f"**Bbox (x y w h):** `{bbox_r}`")

        if score is not None:
            color = "green" if score >= 0.7 else "orange" if score >= 0.5 else "red"
            st.markdown(f"**Score:** :{color}[{score:.1%}]")

        st.divider()

        if selected:
            st.success(f"Selecionado: **{selected}**")
            label_img_path = dict(labels)[selected]
            st.image(label_img_path, width=100)
        else:
            st.info("Selecione uma classe no grid abaixo")

        st.divider()

        btn_confirm, btn_skip = st.columns(2)
        with btn_confirm:
            if st.button(
                "✔ Confirmar",
                disabled=(selected is None),
                type="primary",
                use_container_width=True,
            ):
                out_data = output if output is not None else make_empty_output(labels, str(coco_path))

                existing_img_ids = {img["id"] for img in out_data["images"]}
                if img_meta["id"] not in existing_img_ids:
                    out_data["images"].append(dict(img_meta))

                out_data["annotations"].append({
                    **ann,
                    "category_id": label2catid[selected],
                })
                save_output(out_data, out_path)

                st.session_state.selected_label = None
                st.rerun()

        with btn_skip:
            if st.button("⏭ Pular", use_container_width=True):
                st.session_state.skipped_ids.add(ann["id"])
                st.session_state.selected_label = None
                st.rerun()

    # ── Grid de labels 5 colunas ──────────────────────────────────────────────
    st.divider()
    st.subheader("Escolha a classe")

    grid = st.columns(5, gap="small")
    for i, (name, img_label_path) in enumerate(labels):
        with grid[i % 5]:
            is_sel = (selected == name)
            with st.container(border=is_sel):
                st.image(img_label_path, use_container_width=True)
                if st.button(
                    f"{'✔ ' if is_sel else ''}{name}",
                    key=f"lbl_{name}",
                    type="primary" if is_sel else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.selected_label = name
                    st.rerun()


if __name__ == "__main__":
    main()

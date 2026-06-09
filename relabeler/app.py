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
import streamlit.components.v1 as components
from PIL import Image

from cnn_classifier import CnnClassifier, MODEL_DIR, read_classes

LABELS_DIR = Path(__file__).parent / "labels"
HOTKEY_LABELS = {
    "w": "R-1",
    "a": "R-19",
    "s": "R-6a",
    "d": "R-6b",
}

# JavaScript injetado via st.components.v1.html() que escuta no documento pai
# (mesmo origem: Streamlit serve tudo em localhost) e clica nos botões nativos.
# Dessa forma não depende do protocolo setComponentValue nem de foco no iframe.
KEYBOARD_HTML = """<!doctype html>
<html><head><meta charset="utf-8"></head><body><script>
(function () {
  // Remove listener anterior se este script rodar novamente
  if (window.parent._rlkb) {
    try { window.parent.document.removeEventListener("keydown", window.parent._rlkb, true); } catch(e) {}
  }

  var MAP = {
    "w":          "W · R-1",
    "a":          "A · R-19",
    "s":          "S · R-6a",
    "d":          "D · R-6b",
    "arrowleft":  "Anterior",
    "arrowright": "Próxima",
    "enter":      "Salvar"
  };

  function clickBtn(text) {
    var btns = window.parent.document.querySelectorAll("button");
    for (var i = 0; i < btns.length; i++) {
      var b = btns[i];
      if (!b.disabled && b.textContent.includes(text)) { b.click(); return; }
    }
  }

  function onKey(e) {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    var tag = (e.target.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea" || e.target.isContentEditable) return;
    var label = MAP[e.key.toLowerCase()];
    if (!label) return;
    e.preventDefault();
    clickBtn(label);
  }

  try {
    window.parent.document.addEventListener("keydown", onKey, true);
    window.parent._rlkb = onKey;
    console.log("[relabeler] atalhos de teclado ativos ✓");
  } catch (err) {
    console.error("[relabeler] teclado falhou:", err.message);
  }
})();
</script></body></html>"""


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


def first_unclassified_index(all_anns: list[dict], confirmed_ids: set[int]) -> int:
    """Retorna o índice da primeira anotação ainda sem classificação."""
    for i, ann in enumerate(all_anns):
        if ann["id"] not in confirmed_ids:
            return i
    return 0


def next_unclassified_index(
    all_anns: list[dict],
    confirmed_ids: set[int],
    start_index: int,
) -> int:
    """Avança até a próxima anotação sem classificação, se houver."""
    if not all_anns:
        return 0

    bounded_start = min(max(start_index, 0), len(all_anns) - 1)
    for i in range(bounded_start, len(all_anns)):
        if all_anns[i]["id"] not in confirmed_ids:
            return i
    return bounded_start




def render_hotkey_reference(
    labels_by_name: dict[str, str],
    selected: str | None,
) -> None:
    """Mostra as quatro placas mais comuns com a tecla associada."""
    available_shortcuts = [
        (key.upper(), label, labels_by_name[label])
        for key, label in HOTKEY_LABELS.items()
        if label in labels_by_name
    ]
    if not available_shortcuts:
        return

    st.markdown("**Atalhos rápidos**")
    grid = st.columns(2, gap="small")
    for i, (key, label, img_label_path) in enumerate(available_shortcuts):
        with grid[i % 2]:
            is_sel = selected == label
            with st.container(border=is_sel):
                st.image(img_label_path, use_container_width=True)
                if st.button(
                    f"{key} · {label}",
                    key=f"hotkey_btn_{label}",
                    type="primary" if is_sel else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.selected_label = label
                    st.rerun()


def get_confirmed_annotations(output: dict | None, all_ann_ids: set[int]) -> tuple[dict[int, dict], int]:
    """Indexa anotações confirmadas e conta registros fora do COCO atual."""
    if output is None:
        return {}, 0

    confirmed: dict[int, dict] = {}
    orphan_count = 0
    for ann in output.get("annotations", []):
        ann_id = ann.get("id")
        if ann_id not in all_ann_ids:
            orphan_count += 1
            continue
        confirmed[ann_id] = ann
    return confirmed, orphan_count


def category_name_by_id(coco_like: dict | None) -> dict[int, str]:
    """Cria mapa category_id -> nome para um COCO de entrada ou saída."""
    if coco_like is None:
        return {}
    return {
        cat["id"]: cat["name"]
        for cat in coco_like.get("categories", [])
        if "id" in cat and "name" in cat
    }


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


@st.cache_resource(show_spinner="Carregando modelo CNN...")
def load_cnn_classifier() -> CnnClassifier:
    return CnnClassifier.from_model_dir(MODEL_DIR)


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
    labels_by_name = dict(labels)
    labels_set = set(labels_by_name)

    cnn_classifier: CnnClassifier | None = None
    cnn_error: str | None = None
    cnn_classes: set[str] = set()
    try:
        cnn_classes = set(read_classes(MODEL_DIR)) if (MODEL_DIR / "classes.txt").exists() else set()
        cnn_classifier = load_cnn_classifier()
        cnn_classes = set(cnn_classifier.classes)
    except Exception as exc:
        cnn_error = str(exc)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Entrada")
        coco_input = st.text_input(
            "Arquivo COCO",
            value="datasets/mapillary_traffic_sign_r_type_v2/train/_annotations.coco_relabeled.json",
            help="Caminho relativo ao diretório onde você executa o streamlit",
        )
        images_input = st.text_input("Pasta de imagens", value="datasets/mapillary_traffic_sign_r_type_v2/train")

        coco_path   = Path(coco_input)
        images_dir  = Path(images_input)
        if coco_path.stem.endswith("_relabeled"):
            out_path = coco_path
        else:
            out_path = coco_path.parent / (coco_path.stem + "_relabeled.json")

        st.divider()
        st.markdown("**Ir para anotação**")
        jump_col1, jump_col2 = st.columns([3, 1])
        with jump_col1:
            jump_to = st.number_input(
                "jump",
                min_value=1,
                step=1,
                value=st.session_state.get("current_ann_index", 0) + 1,
                label_visibility="collapsed",
                key="jump_input",
            )
        with jump_col2:
            if st.button("Ir", key="jump_btn", use_container_width=True):
                st.session_state.current_ann_index = int(jump_to) - 1
                st.session_state.selected_ann_id = None
                st.rerun()

        st.divider()
        st.caption(f"Saída: `{out_path}`")
        st.caption(f"Labels disponíveis: **{len(labels)}**")

        st.divider()
        st.markdown("**CNN**")
        if cnn_classifier is not None:
            covered = len(cnn_classes & labels_set)
            st.success(f"Modelo carregado: **{len(cnn_classes)}** classes")
            st.caption(f"Cobertura no grid: **{covered}/{len(labels_set)}** labels")
            missing_from_cnn = sorted(labels_set - cnn_classes)
            if missing_from_cnn:
                st.caption(
                    "Fora da CNN: "
                    + ", ".join(missing_from_cnn[:8])
                    + ("..." if len(missing_from_cnn) > 8 else "")
                )
        else:
            st.warning("CNN indisponível")
            if cnn_error:
                st.caption(cnn_error)

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
    selected_coco = load_coco(str(coco_path))
    review_coco_path = coco_path
    coco = selected_coco
    if coco_path.stem.endswith("_relabeled"):
        source_coco = selected_coco.get("info", {}).get("source_coco")
        if source_coco:
            source_path = Path(source_coco)
            if source_path.exists():
                review_coco_path = source_path
                coco = load_coco(str(review_coco_path))

    imgid2meta = {img["id"]: img for img in coco["images"]}
    all_anns  = coco["annotations"]
    all_ann_ids = {ann["id"] for ann in all_anns}

    # ── Resume: IDs já confirmados estão no arquivo de saída ─────────────────
    output = load_output(out_path)
    if output is not None:
        stored_source = output.get("info", {}).get("source_coco")
        if out_path != coco_path and stored_source is not None and stored_source != str(review_coco_path):
            st.warning(
                f"O arquivo `{out_path.name}` foi gerado de outro COCO "
                f"(`{stored_source}`). Escolha um nome de saída diferente "
                "ou remova o arquivo existente."
            )
            st.stop()

    confirmed_anns_by_id, orphan_output_count = get_confirmed_annotations(output, all_ann_ids)
    confirmed_ids = set(confirmed_anns_by_id)
    output_catid2name = category_name_by_id(output)
    label_catid2name = {catid: name for name, catid in label2catid.items()}

    # IDs pulados ficam em session_state (memória da sessão)
    if "skipped_ids" not in st.session_state:
        st.session_state.skipped_ids: set[int] = set()
    if "selected_label" not in st.session_state:
        st.session_state.selected_label: str | None = None
    if "current_ann_index" not in st.session_state:
        st.session_state.current_ann_index = 0
    if "selected_ann_id" not in st.session_state:
        st.session_state.selected_ann_id: int | None = None
    dataset_state_key = f"{review_coco_path}|{out_path}"
    if st.session_state.get("dataset_state_key") != dataset_state_key:
        st.session_state.dataset_state_key = dataset_state_key
        st.session_state.current_ann_index = first_unclassified_index(all_anns, confirmed_ids)
        st.session_state.selected_ann_id = None
        st.session_state.selected_label = None
        st.session_state.skipped_ids = set()

    components.html(KEYBOARD_HTML, height=0)
    st.session_state.current_ann_index = min(
        max(st.session_state.current_ann_index, 0),
        max(len(all_anns) - 1, 0),
    )
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
    if orphan_output_count:
        st.caption(
            f"`{out_path.name}` contém {orphan_output_count} anotação(ões) "
            "que não aparecem no COCO atual."
        )

    if not all_anns:
        st.success("Nenhuma anotação encontrada no COCO.")
        st.stop()

    # ── Anotação atual ────────────────────────────────────────────────────────
    ann       = all_anns[st.session_state.current_ann_index]
    img_meta  = imgid2meta[ann["image_id"]]
    img_path  = images_dir / img_meta["file_name"]
    score     = ann.get("score")
    confirmed_ann = confirmed_anns_by_id.get(ann["id"])
    confirmed_label = None
    if confirmed_ann is not None:
        confirmed_label = output_catid2name.get(
            confirmed_ann.get("category_id"),
            label_catid2name.get(confirmed_ann.get("category_id")),
        )
    if st.session_state.selected_ann_id != ann["id"]:
        st.session_state.selected_label = confirmed_label if confirmed_label in label2catid else None
        st.session_state.selected_ann_id = ann["id"]
    selected  = st.session_state.selected_label
    image_ann_ids = [
        image_ann["id"]
        for image_ann in all_anns
        if image_ann["image_id"] == ann["image_id"]
    ]
    image_confirmed_count = sum(1 for ann_id in image_ann_ids if ann_id in confirmed_ids)

    st.divider()

    # ── Linha superior: crop + atalhos + info + botões ───────────────────────
    col_crop, col_hotkeys, col_info = st.columns([3, 2, 2], gap="large")

    crop = None
    with col_crop:
        if img_path.exists():
            img   = Image.open(img_path)
            crop  = crop_bbox(img, ann["bbox"])
            st.image(crop, use_container_width=True)
        else:
            st.error(f"Imagem não encontrada: `{img_path}`")

    with col_hotkeys:
        render_hotkey_reference(labels_by_name, selected)

    with col_info:
        is_confirmed = confirmed_ann is not None
        is_skipped = ann["id"] in st.session_state.skipped_ids

        st.markdown(f"**Anotação:** `{st.session_state.current_ann_index + 1}/{total}`")
        if is_confirmed:
            st.success(f"Classificada: **{confirmed_label or 'categoria desconhecida'}**")
        elif is_skipped:
            st.warning("Pulada nesta sessão")
        else:
            st.info("Ainda não classificada")
        st.caption(
            f"Imagem: **{image_confirmed_count}/{len(image_ann_ids)}** "
            "anotação(ões) classificadas"
        )

        st.markdown(f"**Imagem:** `{img_meta['file_name']}`")
        st.markdown(f"**Anotação ID:** `{ann['id']}`")
        bbox_r = [round(v) for v in ann["bbox"]]
        st.markdown(f"**Bbox (x y w h):** `{bbox_r}`")

        if score is not None:
            color = "green" if score >= 0.7 else "orange" if score >= 0.5 else "red"
            st.markdown(f"**Score:** :{color}[{score:.1%}]")

        if crop is not None and cnn_classifier is not None:
            st.divider()
            st.markdown("**Previsão CNN**")
            try:
                cnn_prediction = cnn_classifier.predict(crop)
                predicted_label = cnn_prediction.label
                if predicted_label in labels_by_name:
                    st.image(labels_by_name[predicted_label], width=100)
                    st.success(f"Prevista: **{predicted_label}** ({cnn_prediction.confidence:.1%})")
                    if st.button(
                        "Usar previsão CNN",
                        disabled=(selected == predicted_label),
                        use_container_width=True,
                    ):
                        st.session_state.selected_label = predicted_label
                        st.rerun()
                else:
                    st.warning(f"CNN previu `{predicted_label}`, mas essa label não existe no grid.")

                with st.expander("Top 3 CNN"):
                    for label, confidence in cnn_prediction.top_k:
                        st.caption(f"{label}: {confidence:.1%}")
            except Exception as exc:
                st.warning("Não foi possível executar a CNN neste crop.")
                st.caption(str(exc))

        st.divider()

        nav_prev, nav_next = st.columns(2)
        with nav_prev:
            if st.button(
                "← Anterior",
                disabled=(st.session_state.current_ann_index == 0),
                use_container_width=True,
            ):
                st.session_state.current_ann_index -= 1
                st.session_state.selected_ann_id = None
                st.rerun()
        with nav_next:
            if st.button(
                "Próxima →",
                disabled=(st.session_state.current_ann_index >= total - 1),
                use_container_width=True,
            ):
                st.session_state.current_ann_index += 1
                st.session_state.selected_ann_id = None
                st.rerun()

        st.divider()

        if selected:
            st.success(f"Selecionado: **{selected}**")
            label_img_path = labels_by_name[selected]
            st.image(label_img_path, width=100)
        else:
            st.info("Selecione uma classe no grid abaixo")

        st.divider()

        btn_confirm, btn_skip = st.columns(2)
        with btn_confirm:
            if st.button(
                "✔ Salvar classificação",
                disabled=(selected is None),
                type="primary",
                use_container_width=True,
            ):
                out_data = output if output is not None else make_empty_output(labels, str(review_coco_path))

                existing_img_ids = {img["id"] for img in out_data["images"]}
                if img_meta["id"] not in existing_img_ids:
                    out_data["images"].append(dict(img_meta))

                out_data["annotations"] = [
                    saved_ann
                    for saved_ann in out_data["annotations"]
                    if saved_ann.get("id") != ann["id"]
                ]
                out_data["annotations"].append({
                    **ann,
                    "category_id": label2catid[selected],
                })
                save_output(out_data, out_path)

                st.session_state.skipped_ids.discard(ann["id"])
                st.session_state.selected_label = None
                st.session_state.selected_ann_id = None
                st.session_state.current_ann_index = next_unclassified_index(
                    all_anns,
                    confirmed_ids | {ann["id"]},
                    st.session_state.current_ann_index + 1,
                )
                st.rerun()

        with btn_skip:
            if st.button("⏭ Pular", use_container_width=True):
                st.session_state.skipped_ids.add(ann["id"])
                st.session_state.selected_label = None
                st.session_state.selected_ann_id = None
                st.session_state.current_ann_index = next_unclassified_index(
                    all_anns,
                    processed_ids | {ann["id"]},
                    st.session_state.current_ann_index + 1,
                )
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

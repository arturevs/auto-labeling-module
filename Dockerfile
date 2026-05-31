# Build context: civi-treinamentos/
#   docker build -f auto-labeling-module/Dockerfile -t auto-labeler .
#
# Uso:
#   docker run --rm --gpus all \
#     -v /caminho/entrada:/input \
#     -v /caminho/saida:/output \
#     auto-labeler /input/video.mp4 /output/dataset/

FROM paddlepaddle/paddle:3.3.1-gpu-cuda12.6-cudnn9.5

# libgl1 / libglib2.0 exigidos pelo OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ── Dependências Python do predictor ────────────────────────────────────────
# (paddle já vem na imagem base; ppdet é carregado via sys.path, não pip)
COPY treinamento-paddle-rtdetr/requirements_predict.txt /tmp/
RUN pip install --quiet -r /tmp/requirements_predict.txt \
    requests \
    matplotlib \
    scikit-learn \
    Cython \
    imgaug>=0.4.0 \
    pandas \
    mlflow

# ── treinamento-paddle-rtdetr (só o necessário para inferência) ─────────────
COPY treinamento-paddle-rtdetr/ppdet    /workspace/treinamento-paddle-rtdetr/ppdet
COPY treinamento-paddle-rtdetr/tools    /workspace/treinamento-paddle-rtdetr/tools
COPY treinamento-paddle-rtdetr/configs  /workspace/treinamento-paddle-rtdetr/configs

# ── auto-labeling-module ─────────────────────────────────────────────────────
COPY auto-labeling-module/auto_labeling        /workspace/auto-labeling-module/auto_labeling
COPY auto-labeling-module/label.py             /workspace/auto-labeling-module/label.py
# stub de categorias — ppdet exige o JSON para mapear classes durante inferência
COPY auto-labeling-module/datasets             /workspace/auto-labeling-module/datasets

# ── Pesos do modelo ──────────────────────────────────────────────────────────
COPY auto-labeling-module/best_model.pdparams /workspace/auto-labeling-module/best_model.pdparams

WORKDIR /workspace/auto-labeling-module
ENTRYPOINT ["python", "label.py"]

"""
Script standalone para baixar best_model.pdparams do MLflow.

Uso direto:
    python -m auto_labeling.mlflow_utils
    python -m auto_labeling.mlflow_utils --output /caminho/best_model.pdparams
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

TRACKING_URI = "https://mlflow-internal.inovacao.interjato.com.br"
MODEL_NAME = "rtdetr_placa"
MODEL_VERSION = "3"

_DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "best_model.pdparams"


def download(output: Path = _DEFAULT_OUTPUT) -> None:
    """Baixa ``best_model.pdparams`` do MLflow e salva em ``output``.

    Conecta ao servidor MLflow interno, localiza a versão registrada do modelo
    ``rtdetr_placa`` e copia o arquivo de pesos para o destino indicado.
    O certificado TLS do servidor interno é aceito sem verificação
    (``MLFLOW_TRACKING_INSECURE_TLS=true``).

    Parâmetros
    ----------
    output : Path
        Caminho de destino para o arquivo ``.pdparams``.
        Padrão: ``<raiz do módulo>/best_model.pdparams``.
    """
    import mlflow

    os.environ.setdefault("MLFLOW_TRACKING_INSECURE_TLS", "true")
    mlflow.set_tracking_uri(TRACKING_URI)

    client = mlflow.MlflowClient()
    mv = client.get_model_version(MODEL_NAME, MODEL_VERSION)

    # source: runs:/<run_id>/rtdetr_model/best_model.pdparams
    artifact_path = "/".join(mv.source.split("/")[2:])

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        local = client.download_artifacts(mv.run_id, artifact_path, dst_path=tmp)
        shutil.copy2(local, output)

    print(f"Modelo salvo em: {output} ({output.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baixa best_model.pdparams do MLflow")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args()
    download(args.output)

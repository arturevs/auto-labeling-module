import os
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

# Carrega as variáveis do .env
load_dotenv()

def get_artifacts():
    # Recupera variáveis do ambiente
    uri = os.getenv("MLFLOW_TRACKING_URI")
    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
    run_id = os.getenv("MLFLOW_RUN_ID")
    # Configura a conexão
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    # Lógica para definir qual Run usar
    if not run_id:
        print(f"Buscando a última Run do experimento: {exp_name}...")
        experiment = client.get_experiment_by_name(exp_name)
        if not experiment:
            print(f"Erro: Experimento '{exp_name}' não encontrado.")
            return
        # Busca a run mais recente (pela data de início) que terminou com sucesso
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="status = 'FINISHED'",
            max_results=1,
            order_by=["attributes.start_time DESC"]
        )
        if not runs:
            print("Nenhuma Run finalizada encontrada.")
            return
        run_id = runs[0].info.run_id
    print(f"Conectado em: {uri}")
    print(f"Baixando artefatos da Run: {run_id}")
    # Define pasta de destino local
    local_dir = f"./v2-deepstream/models"
    
    try:
        # O path="" indica que queremos baixar TUDO daquela run
        path_baixado = client.download_artifacts(run_id=run_id, path="", dst_path=local_dir)
        print(f"\n✅ Sucesso! Artefatos disponíveis em: {os.path.abspath(path_baixado)}")
    except Exception as e:
        print(f"❌ Erro ao baixar artefatos: {e}")

if __name__ == "__main__":
    get_artifacts()
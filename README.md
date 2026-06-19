# CIVI - Embarcado

Sistema de detecção e geolocalização de placas de sinalização em vídeos de dashcam utilizando NVIDIA DeepStream + RT-DETR + classificador CNN.

```
deep-stream/
├── deepstream_processor/          # pipeline DeepStream: detecção, tracking e geração de dataset
│   ├── deepstream_app_dataset.py  # script principal — processa pasta de vídeos
│   ├── detection_models/          # modelos TensorRT (detector + classificador)
│   ├── data/input/                # vídeos MP4 de entrada
│   └── Dockerfile.deepstream
├── edge-backend/                  # persiste imagens e metadados, envia para o backend
├── consumer-api/                  # API provisória para acompanhamento
├── dataset_output/                # saída gerada (criada automaticamente)
├── docker-compose.yml             # pipeline em tempo real
├── docker-compose.dataset.yml     # pipeline de geração de dataset
└── plot_bbox.py                   # visualização de bboxes pós-inferência
```

---

## Compatibilidade DeepStream e GPU

> **Atenção:** A versão do DeepStream é vinculada ao driver NVIDIA e à arquitetura da GPU. Não existe uma versão universal — cada ambiente exige uma combinação específica.

| GPU (arquitetura) | Driver mínimo | DeepStream recomendado |
|---|---|---|
| Ampere (RTX 30xx, A100) | 525+ | 6.3 |
| Ada Lovelace (RTX 40xx) | 535+ | 6.3 / 7.0 |
| Orin / Jetson AGX | JetPack 5.x | 6.3 |
| Turing (RTX 20xx, T4) | 515+ | 6.2 / 6.3 |

**O usuário é responsável por verificar a compatibilidade antes de executar.** Consulte a [matriz de compatibilidade oficial](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html) e ajuste a imagem base no `Dockerfile.deepstream` se necessário (ex: `nvcr.io/nvidia/deepstream:7.0-triton-multiarch`).

Para verificar o driver instalado no host:
```sh
nvidia-smi
```

Se o container falhar com `unknown or invalid runtime name: nvidia`, instale o `nvidia-container-toolkit` e reinicie o Docker:
```sh
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## Pipeline de Geração de Dataset

Processa uma pasta de vídeos MP4 e gera, para cada vídeo, uma pasta de output independente com dataset COCO e vídeo rotulado.

### Estrutura de saída

```
dataset_output/
└── nome_do_video/
    ├── images/          # frames dos objetos detectados (um frame por objeto rastreado)
    ├── annotated.mp4    # vídeo completo com bboxes desenhados
    └── annotations.json # dataset COCO deste vídeo
```

### Execução

Coloque os vídeos `.mp4` em `deepstream_processor/data/input/` e rode:

```sh
docker compose -f docker-compose.dataset.yml up --build
```

Para apontar para outra pasta de entrada:

```sh
INPUT_DIR=data/minha_pasta docker compose -f docker-compose.dataset.yml up
```

### Visualização de bboxes (pós-inferência)

Após a geração do dataset, visualize as anotações sem precisar rodar o DeepStream novamente:

```sh
# Requer `uv` instalado (https://docs.astral.sh/uv/)
uv run plot_bbox.py --dataset dataset_output/nome_do_video/annotations.json \
                    --images  dataset_output/nome_do_video/images \
                    --output  dataset_output/nome_do_video/images_bbox
```

---

## Conversão de Modelo para ONNX

### RT-DETR (detector)

```sh
# Dentro do container ou com as dependências instaladas
python3 utils/export_rtdetr.py
```

### SmallTrafficSignCNN (classificador)

```sh
python3 utils/convert_classifier_v2_to_onnx.py
```

O engine TensorRT (`.engine`) é compilado automaticamente pelo DeepStream na primeira execução. Após isso, reutilize o arquivo `.engine` para evitar recompilação.

---

## Sincronização entre máquinas

Para transferir o projeto (excluindo arquivos pesados e gerados):

```sh
rsync -avz --progress \
  --exclude='*.engine' \
  --exclude='__pycache__' \
  --exclude='dataset_output/' \
  --exclude='data/output/' \
  --exclude='.git/' \
  ./ usuario@host:/caminho/destino/deep-stream
```

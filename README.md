# auto-labeling-module

Pipeline de rotulagem automática e revisão manual para detecção de placas veiculares com RT-DETR.

## Visão geral

```
Vídeo ou pasta
de imagens          ──►  RT-DETR (placa_v1)  ──►  Dataset COCO
                              inferência              images/ + annotations.coco.json
                                   │
                                   ▼
                         Relabeler (Streamlit)
                         revisão anotação a anotação
                                   │
                                   ▼
                         dataset/
                         ├── images/          ← apenas imagens confirmadas
                         └── annotations.coco.json
```

## Estrutura do repositório

```
civi-treinamentos/
├── auto-labeling-module/          ← este repositório
│   ├── auto_labeling/             # pacote Python principal
│   │   ├── labeler.py             #   AutoLabeler — orquestra inferência + COCO
│   │   ├── coco_builder.py        #   constrói o JSON no formato COCO
│   │   ├── dedup.py               #   filtragem de frames duplicados (dhash)
│   │   └── mlflow_utils.py        #   download de pesos a partir do MLflow
│   ├── relabeler/                 # app Streamlit de revisão manual
│   │   ├── app.py
│   │   └── labels/                #   imagens .png nomeadas por classe
│   ├── label.py                   # entrypoint CLI
│   ├── visualize.py               # gera vídeo de validação com bboxes
│   ├── best_model.pdparams        # pesos do modelo (ver §5 para download)
│   ├── rtdetr.yml                 # configuração do modelo
│   └── Dockerfile
└── treinamento-paddle-rtdetr/     # repositório irmão (predictor + ppdet)
```

## Pré-requisitos

- Docker
- UV (facilita bastante)
- NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Os dois repositórios lado a lado conforme a estrutura acima

---

## 1. Download dos pesos via MLflow

Para (re)baixar o `best_model.pdparams` do MLflow interno:

```bash
uv run -m auto_labeling.mlflow_utils # caso o comando não rode, pode ser que precise instalar a versão do Python com o uv
# uv python install 3.11
# destino customizado
uv run -m auto_labeling.mlflow_utils --output /caminho/best_model.pdparams
```

---

## 2. Build da imagem

Execute **uma vez** a partir de do diretório anterior a esse:

```bash
docker build -f <nome_da_pasta>/Dockerfile -t auto-labeler .
```

> O `best_model.pdparams` é copiado para dentro da imagem durante o build —
> não é necessário montá-lo em runtime.

---

## 3. Rotulagem automática

### 3.1 A partir de um vídeo

Extrai frames em intervalos regulares, filtra duplicatas e gera o dataset COCO.

```bash
docker run --rm --gpus all \
  -v $(pwd)/video.mp4:/input/video.mp4:ro \
  -v $(pwd)/output:/output \
  auto-labeler /input/video.mp4 /output/
```

### 3.2 A partir de uma pasta de imagens

Processa cada imagem da pasta (JPEG, PNG, BMP, TIFF, WebP).
Formatos não-JPEG são convertidos para JPEG antes da inferência.

```bash
docker run --rm --gpus all \
  -v $(pwd)/imagens:/input/imagens:ro \
  -v $(pwd)/output:/output \
  auto-labeler /input/imagens/ /output/
```

> `--fps` e `--no-dedup` não têm efeito no modo pasta — todas as imagens são processadas.

### 3.3 Saída gerada

```
output/
├── images/
│   ├── 00000001.jpg
│   ├── 00000002.jpg
│   └── ...
└── annotations.coco.json
```

### 3.4 Opções

| Argumento | Padrão | Descrição |
|---|---|---|
| `--fps N` | `1.0` | Frames por segundo extraídos do vídeo |
| `--threshold N` | `0.5` | Score mínimo para aceitar uma detecção (0–1) |
| `--device gpu\|cpu` | `gpu` | Dispositivo de inferência |
| `--max-distance N` | `10` | Limiar de Hamming para filtrar frames duplicados (0–64, menor = mais estrito) |
| `--no-dedup` | — | Desativa a filtragem de duplicatas para vídeos |
| `--weights PATH` | `best_model.pdparams` na raiz do módulo | Pesos do modelo |
| `--config PATH` | default do predictor | Arquivo de configuração `.yml` |
| `--paddle-repo PATH` | `../treinamento-paddle-rtdetr` | Raiz do repositório irmão |

Exemplo com opções customizadas:

```bash
docker run --rm --gpus all \
  -v $(pwd)/video.mp4:/input/video.mp4:ro \
  -v $(pwd)/output:/output \
  auto-labeler /input/video.mp4 /output/ \
  --fps 2 --threshold 0.4 --max-distance 5
```

---

## 4. Revisão manual — Relabeler

Interface Streamlit para revisar as detecções anotação a anotação, corrigir classes
e exportar o dataset final corrigido.

### Pré-requisito

Adicione as imagens de referência de cada classe em `relabeler/labels/`,
nomeadas com o nome exato da classe:

```
relabeler/labels/
├── carro.png
├── moto.png
└── ...
```

### Sugestão automática com CNN

O Relabeler pode carregar uma CNN de classificação registrada no MLflow para
sugerir a classe de cada crop. Configure o `.env` com:

```bash
MLFLOW_TRACKING_URI=xxxxxxxxx
MLFLOW_REGISTERED_MODEL_NAME=traffic_sign_cnn_classifier
MLFLOW_MODEL_ALIAS=traffic_sign_cnn_v1
```

Baixe o modelo do MLflow antes de abrir o Relabeler:

```bash
# a partir de auto-labeling-module/
uv run python relabeler/cnn_classifier.py
```

O comando salva os artefatos em:

```
relabeler/models/cnn_classifier/
├── best_model.pt
├── classes.txt
└── classes.json
```

O `classes.txt` define as classes cobertas pela CNN. Ela não contém todas as
labels disponíveis em `relabeler/labels/`, então a sugestão aparece apenas para
as classes que o modelo conhece; a escolha manual continua disponível para todas.

### Execução

```bash
# a partir de auto-labeling-module/
streamlit run relabeler/app.py
```

### Fluxo de uso

1. **Sidebar** — informe o caminho do `annotations.coco.json` gerado na rotulagem
   e da pasta `images/`
2. Para cada anotação pendente, o crop da bbox é exibido junto ao score,
   metadados e a sugestão da CNN quando disponível
3. Escolha a classe correta no grid de labels (ou **⏭ Pular** para deixar para depois)
4. **✔ Confirmar** — grava imediatamente no arquivo `_relabeled.json`; o progresso
   é persistido a cada clique, podendo fechar e retomar a qualquer momento
5. **⬇ Baixar & Exportar dataset** (sidebar) — duas ações em um clique:
   - Baixa o JSON relabelado no navegador
   - Cria `output/dataset/images/` e `output/dataset/annotations.coco.json`
     contendo **apenas as imagens confirmadas**

---

## 5. Validação visual

Gera um vídeo com as bounding boxes sobrepostas para conferir as detecções.
Usa `ffmpeg` se disponível; caso contrário, faz fallback para `cv2.VideoWriter`.

```bash
# gera output/validacao.mp4 a 2 FPS
python visualize.py output/

# com opções
python visualize.py output/ --fps 4 --out minha_validacao.avi
```

| Argumento | Padrão | Descrição |
|---|---|---|
| `--fps N` | `2.0` | FPS do vídeo de saída |
| `--out ARQUIVO` | `<dataset>/validacao.mp4` | Caminho do vídeo de saída |


---

## 6. Desenvolvimento local (sem Docker)

```bash
# instala dependências
pip install -r requirements.txt
# ou com uv
uv sync

# rotulagem a partir de vídeo
python label.py video.mp4 output/

# rotulagem a partir de pasta
python label.py imagens/ output/

# revisão manual
streamlit run relabeler/app.py

# validação visual
python visualize.py output/
```

> Requer PaddlePaddle com suporte a GPU instalado e o repositório
> `treinamento-paddle-rtdetr` como irmão deste diretório.

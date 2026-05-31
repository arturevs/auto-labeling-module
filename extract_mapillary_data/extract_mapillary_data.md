# Baixar imagens do Mapillary e filtrar por deteccao

Este fluxo baixa imagens aleatorias do Mapillary nas capitais do Brasil e depois cria uma segunda pasta contendo somente as imagens em que o modelo detectou algum objeto.

## 1. Preparar o token do Mapillary

O script usa a API do Mapillary. Defina o token no ambiente ou em um arquivo `.env`.

```bash
MAPILLARY_ACCESS_TOKEN=seu_token_aqui
```

Tambem e possivel passar o token diretamente com `--access-token`, mas o `.env` evita expor o token no historico do terminal.

## 2. Baixar imagens


```bash
uv run python utils/extract_mapillary_data.py 1000 \
  --output-dir datasets/mapillary_capitais_random_1000 \
  --thumb-size 2048 \
  --workers 12 \
  --download-workers 12 \
  --max-cells 510 \
  --limit-per-cell 100 \
  --candidate-multiplier 1.3 \
  --min-distance-m 10 \
  --front-angle-tolerance 80 \
  --fallback
```

Apesar do nome do arquivo ainda conter `natal`, o script agora percorre todas as capitais brasileiras configuradas em `CITY_COVERAGES`.

Principais opcoes:

- `1000`: quantidade final de imagens a baixar.
- `--output-dir`: pasta onde as imagens e o `metadata.json` serao salvos.
- `--thumb-size`: tamanho da imagem baixada. Use `2048` para boa qualidade ou `original` para a maior imagem disponivel.
- `--max-cells`: quantidade maxima de micro-bboxes consultados por capital.
- `--min-distance-m`: distancia minima desejada entre imagens selecionadas.
- `--front-angle-tolerance`: tolerancia para priorizar cameras frontais de veiculos.
- `--fallback`: completa com imagens `perspective` nao panoramicas se nao houver frontais suficientes.

A saida inclui:

```text
datasets/mapillary_capitais_random_1000/
  mapillary_sao_paulo_0001_....jpg
  mapillary_recife_0002_....jpg
  metadata.json
```

O `metadata.json` registra coordenadas, cidade de origem, bbox consultado, compass angle e erro frontal.

## 3. Ajustes comuns

Para baixar menos imagens em um teste rapido:

```bash
uv run python utils/extract_mapillary_data.py 100 \
  --output-dir datasets/mapillary_capitais_random_100 \
  --thumb-size 1024 \
  --workers 8 \
  --download-workers 8 \
  --max-cells 60 \
  --limit-per-cell 100 \
  --candidate-multiplier 1.5 \
  --min-distance-m 10 \
  --front-angle-tolerance 75 \
  --fallback
```

Para ser mais rigoroso na filtragem do modelo, aumente o threshold:

```bash
--threshold 0.7
```

Para manter mais imagens candidatas, reduza o threshold:

```bash
--threshold 0.3
```

# 4. Filtrar Dataset

Navegue para ```treinamento-paddle-rtdetr``` e use:

```bash
uv run python tools/filter_detected_images.py \
  -c configs/custom_configs/placa_v1/rtdetr/rtdetr.yml \
  -o weights=output/placa_v1/best_model.pdparams \
  --infer_dir ../auto-labeling-module/datasets/mapillary_capitais_random_1000 \
  --output_dir ../auto-labeling-module/datasets/mapillary_capitais_com_deteccao \
  --threshold 0.5
```

## Observacoes

- A API do Mapillary pode retornar quantidades diferentes ao longo do tempo.
- Se poucas imagens forem baixadas, aumente `--max-cells` e `--pages-per-cell`, ou reduza `--min-distance-m`.
- O filtro de deteccao copia imagens originais; ele nao desenha boxes.
- O arquivo `metadata.json` da etapa de download continua sendo a melhor fonte para rastrear cidade, coordenadas e origem da imagem.
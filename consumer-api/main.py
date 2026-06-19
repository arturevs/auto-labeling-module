import os
import json
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
import logging

# Configuração de log para vermos o que chega
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="CIVI Cloud Consumer API")

# Pasta onde vamos salvar as imagens localmente na "nuvem"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/api/events")
async def receive_event(
    # Ficarão no 'form-data', então usamos Form para a string JSON e File para a imagem
    event: str = Form(...),
    image: UploadFile = File(None)
):
    logging.info("--- Novo Evento Recebido ---")
    
    # 1. Desserializa o JSON enviado pelo CloudPublisher
    try:
        event_data = json.loads(event)
        logging.info(f"Dados do evento: {json.dumps(event_data, indent=2)}")
    except json.JSONDecodeError:
        logging.error("O campo 'event' não contém um JSON válido.")
        return JSONResponse(status_code=400, content={"message": "Invalid JSON mapping."})

    # 2. Tratamento da Imagem (se houver)
    saved_path = None
    if image:
        # A imagem chega com o nome que o CloudPublisher mandou no multipart: f"{image_id}.jpg"
        file_path = os.path.join(UPLOAD_DIR, image.filename)
        
        # Salva o arquivo no disco do servidor
        try:
            with open(file_path, "wb") as buffer:
                content = await image.read()
                buffer.write(content)
            saved_path = file_path
            logging.info(f"Imagem recebida e salva em: {saved_path}")
        except Exception as e:
            logging.error(f"Erro ao salvar a imagem: {e}")
            return JSONResponse(status_code=500, content={"message": "Erro ao salvar a imagem."})
    else:
        logging.info("Nenhuma imagem atrelada a este evento.")

    # Retorna HTTP 200 de sucesso (que é o que o response.ok do Edge espera para marcar "published=True" no edge)
    return {
        "status": "success",
        "message": "Evento processado com sucesso.",
        "event_id": event_data.get("event_id"),
        "image_saved": bool(saved_path)
    }

if __name__ == "__main__":
    import uvicorn
    # Roda o servidor na porta 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
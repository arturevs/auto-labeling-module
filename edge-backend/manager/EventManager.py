import logging
import os
import json
import time
from parser.event_parser import EventParser
from filters.detection_filter import DetectionFilter
from persistence.orm_models import DetectionEventModel
from persistence.database import SessionLocal
import shutil
from settings import Settings


#TODO: Futuramente essa classe deve ser quebrada. por enquanto ela tem muitas responsabliidades: observa, chama parser, chama filtro, salva no banco, move imagem e limpa a pasta temporaria.
class EventManager:
    
    def __init__(self, incoming_path: str, detection_filter: DetectionFilter = None):
        self.incoming_path = incoming_path
        self.min_file_age = 3  
        self.detection_filter = detection_filter or DetectionFilter()         

    def scan(self):
        """Escaneia a pasta de entrada e processa os arquivos JSON encontrados."""
        try:
            files = [
                file for file in os.listdir(self.incoming_path)
                if file.endswith(".json")
            ]
        except FileNotFoundError:
            logging.error(f"Directory {self.incoming_path} not found.")
            return []
        except OSError as e:
            logging.error(f"Error accessing directory {self.incoming_path}: {e}")
            return []
            
        if not files:
            # logging.info("No new files to process.")
            return []
            
        for file_name in files:
            full_path = os.path.join(self.incoming_path, file_name)
                
            if not self._is_file_ready(full_path):
                # logging.info(f"File {file_name} is not ready for processing.")
                continue

            self._consume_file(full_path)
         
    def _consume_file(self, file_path: str):
        """Lê o arquivo, parseia o evento e aplica o filtro de detecção."""
       
        try:
            with open(file_path, 'r') as f:
                raw_data = json.load(f)
        except json.JSONDecodeError:
            logging.error(f"Error parsing JSON from file {file_path}")
            return
        except OSError as e:
            logging.error(f"Unexpected error occurred while reading {file_path}: {e}")
            return
        
        try:
            event = EventParser.parse(raw_data)
        except Exception as e:
            logging.error(f"Error parsing event from file {file_path}: {e}")
            return

    
        # logging.info(f"Consumed event: {event}")
        
        if not self.detection_filter.should_consume_event(event):
            logging.warning(f"Event {event.event_id} failed the filter.")
            self._cleanup(file_path)
            
            return
        
        logging.info(f"Event {event.event_id} passed the filter and will be consumed.")
        
        persisted_image_url = self._save_image(file_path, event.event_id)
        
        if persisted_image_url:
            event.image_url = persisted_image_url
        
        self._save_metadata(event)
        
        self._cleanup(file_path)
    
    
    def _save_metadata(self, event):
        """Salva os metadados do evento no banco de dados usando SQLAlchemy."""
        db = SessionLocal()
        
        try:
            db_event = DetectionEventModel(
                event_id=event.event_id,
                camera_id=event.camera_id,
                timestamp=event.timestamp,
                event_type=event.event_type,
                confidence=event.confidence,
                bbox=event.bbox,
                time_coord=event.time_coord,
                image_url=event.image_url
            )
            db.add(db_event)
            db.commit()
            logging.info(f"Event {event.event_id} metadata saved to database.") 
        except Exception as e:
            db.rollback()
            logging.error(f"Error saving event {event.event_id} to database: {e}")
        finally:
            db.close()       
    
    def _save_image(self, file_path: str, eventId: str) -> str:
        """move imagem da pasta de incoming para o volume de imagens"""
        destination_dir = Settings.IMAGE_STORAGE_PATH
        os.makedirs(destination_dir, exist_ok=True)
        
        source_image_path = self._get_image_paths(file_path)
        if not os.path.exists(source_image_path):
            logging.warning(f"Image file {source_image_path} not found for event {file_path}.")
            return
        
        file_name = f"{eventId}.jpg" 
        destination_path = os.path.join(destination_dir, file_name)
        
        try:
            shutil.copy2(source_image_path, destination_path)
            logging.info(f"Image {file_name} successfully persisted to {destination_path}")
            
            return destination_path
        except Exception as e:
            logging.error(f"Error persisting image {file_name}: {e}")
            
            return None    
    
    def _delete_file(self, file_path: str):
        """Remove o arquivo especificado, logando erros se ocorrerem."""
        try:
            os.remove(file_path)
            # logging.info(f"Deleted file {file_path}")
        except OSError as e:
            logging.error(f"Error deleting file {file_path}: {e}")
    
    
    def _cleanup(self, json_path: str):
        """Remove o arquivo JSON e a imagem associada após a persistência."""
        self._delete_file(json_path)
        self._delete_file(self._get_image_paths(json_path))
        logging.info(f"Deleted temp files")
        
    
    def _get_image_paths(self, json_path: str) -> str:
        base_name = os.path.splitext(json_path)[0]
        return f"{base_name}.jpg"
        
    
    def _is_file_ready(self, file_path: str) -> bool:
        """
        Verifica se o arquivo já tem idade mínima suficiente
        para evitar leitura durante escrita.
        """

        file_modified_time = os.path.getmtime(file_path)
        file_age = time.time() - file_modified_time

        return file_age > self.min_file_age


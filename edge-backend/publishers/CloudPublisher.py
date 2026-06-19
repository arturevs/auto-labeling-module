import logging
import os
import requests
import json
from persistence.database import SessionLocal
from persistence.orm_models import DetectionEventModel

class CloudPublisher:
    
    def __init__(self, civi_api_url: str):
        self.civi_api_url = civi_api_url
        logging.basicConfig(level=logging.INFO)
    
    def publish_pending_events(self):
        """Busca eventos pendentes no banco e tenta publicá-los na nuvem."""
        db = SessionLocal()
        try:
            # Busca eventos que ainda não foram publicados
            pending_events = db.query(DetectionEventModel).filter(DetectionEventModel.published == False).all()
            
            if not pending_events:
                return
                
            logging.info(f"Found {len(pending_events)} pending events to publish.")
            
            for event in pending_events:
                success = self._send_to_cloud(event)
                
                if success:
                    # Marca como publicado e salva no banco
                    event.published = True
                    db.commit()
                    logging.info(f"Event {event.event_id} successfully published and marked.")
                else:
                    logging.warning(f"Failed to publish event {event.event_id}. Will retry later.")
                    
        except Exception as e:
            logging.error(f"Error during cloud publishing routine: {e}")
        finally:
            db.close()
    
    def _send_to_cloud(self, event: DetectionEventModel) -> bool:
        
        image_id = None
        if event.image_url:
            image_filename = os.path.basename(event.image_url) 
            image_id = os.path.splitext(image_filename)[0]
            
        
        json_payload = {
            "event_id": event.event_id,
            "camera_id": event.camera_id,
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type,
            "confidence": event.confidence,
            "bbox": event.bbox,
            "time_coord": event.time_coord,
            "image_id": image_id
        }
        
        data = {
            "event": json.dumps(json_payload)
        }
        files = {}
        
        try:
            if event.image_url:
                try:
                    with open(event.image_url, "rb") as file_handle:
                        image_data = file_handle.read()
                        files = {"image": (f"{image_id}.jpg", image_data, "image/jpeg")}
                except FileNotFoundError:
                    logging.error(f"Image not found locally at {event.image_url}. Sending without it.")
            
            response = requests.post(self.civi_api_url, data=data, files=files, timeout=10)
            
            return response.ok
        
        except requests.RequestException as e:
            logging.error(f"Network error while sending event {event.event_id} to cloud: {e}")
            return False
        
       
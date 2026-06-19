from models.detection_event import DetectionEvent
from datetime import datetime

class EventParser:
    
    @staticmethod
    def parse(raw_data: dict) -> DetectionEvent:
        # Implementar a lógica de parsing dos dados brutos do DeepStream para criar um objeto DetectionEvent
           
        event_id = raw_data.get("event_id", None)
        camera_id = raw_data.get("camera_id")
        timestamp_str = raw_data.get("timestamp") or None
        timestamp = datetime.now()
        if timestamp_str:
            timestamp = datetime.fromisoformat(timestamp_str)
        event_type = raw_data.get("class", "") #por enquanto o json ta class
        confidence = raw_data.get("confidence")
        bbox = raw_data.get("bbox", [0.0, 0.0, 0.0, 0.0])
        time_coord = raw_data.get("time_coord", 0.0) #por enquanto é time_coord
        
        return DetectionEvent(
            event_id=event_id,
            camera_id=camera_id,
            timestamp=timestamp,
            event_type=event_type,
            confidence=confidence,
            bbox=bbox,
            time_coord=time_coord
        )

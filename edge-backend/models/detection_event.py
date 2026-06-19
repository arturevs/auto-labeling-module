
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass #usando dataclass para facilitar a criação de objetos e a manipulação dos dados
class DetectionEvent:
    #Campos que idealizo, podem ser outros. Teremos que lidar com o parse p/ dados vindo do deepstream
    #coloquei optional para oq acho que nao estara vindo na primeira versao
    
    event_id: Optional[str]
    camera_id:  Optional[str]
    timestamp: Optional[datetime]
    
    event_type: Optional[str]
    confidence: Optional[float]
    
    bbox: Optional[list[float]]  # [x_min, y_min, x_max, y_max]
    #coordinates: list[float]  # [latitude, longitude]
    time_coord:Optional[float] #tempo em segundos do evento, é o que usaremos na primeira versao
    
    image_url: Optional[str] = None
    
    #extra_info: dict  # Qualquer informação adicional relevante para o evento

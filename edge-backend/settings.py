import os

class Settings:
    
    INCOMING_PATH =os.getenv("INCOMING_PATH", "/incoming")
    IMAGE_STORAGE_PATH = os.getenv("IMAGE_STORAGE_PATH", "/data/images")
    
    SCAN_INTERVAL = 5.0 #segundos
    MIN_FILE_AGE = 5.0 #segundos, idade mínima do arquivo para ser processado
    
    CONFIDENCE_THRESHOLD = 0.5
    ALLOWED_EVENT_TYPES = []  
    
    DATABASE_URL = "postgresql://civi_edge:civi_edge@db:5432/civi_edge_db"
    
    CIVI_API_URL = os.getenv("CIVI_API_URL", "http://host.docker.internal:8000/api/events")

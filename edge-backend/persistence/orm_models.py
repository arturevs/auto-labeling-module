from datetime import datetime

from sqlalchemy import JSON, Column, String, Integer, Float, DateTime,Boolean
from persistence.database import Base

class DetectionEventModel(Base):
    __tablename__ = 'detection_events'
    
    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(String, unique=True, index=True,nullable=True)
    camera_id = Column(String, nullable=True)
    timestamp = Column(DateTime, nullable=True, default=datetime.now)
    
    event_type = Column(String,nullable=True)
    confidence = Column(Float, nullable=True)
    
    bbox = Column(JSON, nullable=True) 
    
    time_coord = Column(Float, nullable=True)  # Tempo em segundos do evento
    
    image_url = Column(String, nullable=True)
    
    published = Column(Boolean, default=False, index=True,nullable=True)

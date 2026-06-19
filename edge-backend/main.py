import time
from settings import Settings
import logging

from manager.EventManager import EventManager
from persistence.database import engine
from persistence.orm_models import Base
from publishers.CloudPublisher import CloudPublisher

def setup_logging():
 
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def init_db():
    logging.info("Initializing database...")
    
    try:
        Base.metadata.create_all(bind=engine)
        logging.info("Database initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing database: {e}")
        raise
    
def main():
    setup_logging()
    init_db()
    
    event_manager = EventManager(
        incoming_path=Settings.INCOMING_PATH,
        detection_filter=None
    )
    
    # cloud_publisher = CloudPublisher(civi_api_url=Settings.CIVI_API_URL)
    
    logging.info("Starting event manager...")
    
    while True:
        try:
            event_manager.scan()
            
            # cloud_publisher.publish_pending_events()
            
            time.sleep(Settings.SCAN_INTERVAL)
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            time.sleep(5)  # Espera um pouco antes de tentar novamente

if __name__ == "__main__":
    main()
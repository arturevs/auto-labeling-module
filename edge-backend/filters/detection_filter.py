class DetectionFilter:
    def __init__(self, confidence_threshold: float = 0.5, allowed_event_types: list[str] = None):
        self.confidence_threshold = confidence_threshold
        self.allowed_event_types = allowed_event_types if allowed_event_types is not None else []

    def should_consume_event(self, event)-> bool:
        
        if self.allowed_event_types:
            if event.event_type not in self.allowed_event_types:
                print(f"Event {event.event_id} filtered out by type: {event.event_type}")
                return False
        return True
from abc import ABC

class BaseAdapter(ABC):
    """Base apter class for all adapters"""
    def __init__(self, config: dict):
        self.config = config
        
    def to_device(self, device: str):
        if hasattr(self.original, 'to'):
            self.original.to(device)
        return self
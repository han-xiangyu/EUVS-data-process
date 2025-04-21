from typing import Union
from adapters.groundedSAM_adapter import GroundedSAMAdapter

# from adapters.foundationPose_adapter import FoundationPoseAdapter

class ModelFactory:
    @classmethod
    def create(cls, 
              model_type: str, 
              config: dict = None):
        """Factory method to create a model adapter"""
        config = config or {}
    
            
        if model_type == 'sam':
            return GroundedSAMAdapter(config)
        
        # elif model_type == 'foundationPose':
        #     return FoundationPoseAdapter(config)
        
        raise ValueError(f"Unsupported model type: {model_type}")
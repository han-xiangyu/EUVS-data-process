

class postprocessor():
    @staticmethod
    def to_dino_format(raw_features):
        """Transform raw features to the format that the rest of the system expects"""
        return raw_features
        
    @staticmethod
    def to_groundedSAM_format(masks):
        """Transform raw features to the format that the rest of the system expects"""
        return masks
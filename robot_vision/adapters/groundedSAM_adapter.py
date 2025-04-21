import numpy as np
import cv2
from pathlib import Path
import torch


from adapters.base_adapter import BaseAdapter
from pipelines.image_preprocessor import img_preprocessor
from pipelines.post_processor import postprocessor
from models.GroundedSAM2.sam2.build_sam import build_sam2
from models.GroundedSAM2.sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class GroundedSAMAdapter(BaseAdapter):
    """Adapter for Grounded SAM models"""
    def __init__(self, config: dict):
        super().__init__(config)
        self.GROUNDING_MODEL = config['grounding_dino']['model_id']
        self.TEXT_PROMPT = config['text_prompt']
        self.SAM2_CHECKPOINT = config['sam2']['checkpoint']
        self.SAM2_MODEL_CONFIG = config['sam2']['model_config']
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # Create output directory
        # self.OUTPUT_DIR = Path(config['output_dir'])
        # self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Environment settings
        # Use bfloat16
        torch.autocast(device_type=self.DEVICE, dtype=torch.bfloat16).__enter__()

        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            # Turn on tfloat32 for Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Build SAM2 image predictor
        self.sam2_checkpoint = self.SAM2_CHECKPOINT
        self.model_cfg = self.SAM2_MODEL_CONFIG
        self.sam2_model = build_sam2(self.model_cfg, self.sam2_checkpoint, device=self.DEVICE)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        # Build grounding dino from huggingface
        self.model_id = self.GROUNDING_MODEL
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.DEVICE)
        
    
    def segment(self, image):
        """Segment an image"""
        # Load the image
        image_np = np.array(image)
        self.sam2_predictor.set_image(image_np)
        inputs = self.processor(images=image, text=self.TEXT_PROMPT, return_tensors="pt").to(self.DEVICE)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )

        # Get the box prompt for SAM 2
        input_boxes = results[0]["boxes"].cpu().numpy()
        class_names = results[0]["labels"]

        if len(input_boxes) == 0:
            # No objects detected, output an all-zero mask
            mask = np.zeros((image.height, image.width), dtype=np.uint8)
            return mask, [], []

        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        # Convert the shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        # Combine all masks into one
        combined_mask = np.zeros((image.height, image.width), dtype=np.uint8)
        for mask in masks:
            mask_resized = cv2.resize(mask.astype(np.uint8), (image.width, image.height))
            combined_mask = np.logical_or(combined_mask, mask_resized).astype(np.uint8)

        # Set mask pixels to 1
        combined_mask[combined_mask > 0] = 1
        final_mask = combined_mask

        
        return final_mask, input_boxes, class_names
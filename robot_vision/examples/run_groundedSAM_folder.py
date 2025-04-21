from core.model_factory import ModelFactory
from pathlib import Path
import yaml
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def main():

    # Path to the YAML file
    yaml_file = "./config/groundedSAM_folder.yaml"

    # Read the YAML file
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)

    # 读取配置
    image_folder = Path(config['image_folder'])
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建分割器
    segmenter = ModelFactory.create("sam", config)

    # 支持的图片后缀
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    for img_path in sorted(image_folder.iterdir()):
        if img_path.suffix.lower() not in exts:
            continue
        
        # Load the image
        image = Image.open(img_path).convert("RGB")
        
        segmenter = ModelFactory.create("sam", config)
        mask, input_boxes, class_names = segmenter.segment(image, img_path)
        # # Convert bounding box format if needed 
        # # input_boxes should be in format [[x1, y1, x2, y2], ...] 

        # # Create a copy of the image for drawing boxes
        # image_with_boxes = np.array(image.copy())


        # # Create a figure with two subplots
        # fig, ax = plt.subplots(1, 2, figsize=(30, 10))
        # # Make bounding box thicker (now 6 pixels wide)
        # for i, box in enumerate(input_boxes):
        #     x1, y1, x2, y2 = map(int, box)
        #     color = np.random.randint(0, 255, size=3).tolist()
            
        #     # Draw thicker rectangle (6 pixels)
        #     thickness = 6
        #     image_with_boxes[y1:y1+thickness, x1:x2] = color  # Top
        #     image_with_boxes[y2-thickness:y2, x1:x2] = color  # Bottom
        #     image_with_boxes[y1:y2, x1:x1+thickness] = color  # Left
        #     image_with_boxes[y1:y2, x2-thickness:x2] = color  # Right
            
        #     # Get class name
        #     class_name = class_names[i] if i < len(class_names) else f"Object {i+1}"
            
        #     # Add class names to the existing subplot
        #     ax[0].text(x1, y1-10, class_name, bbox=dict(facecolor=np.array(color)/255, alpha=0.8),
        #             fontsize=12, color='white')
        
        # # Display the original image
        # ax[0].imshow(image_with_boxes)
        # ax[0].set_title('Original Image')
        # ax[0].axis('off')

        # # Display the mask
        # ax[1].imshow(mask, cmap='gray')
        # ax[1].set_title('Segmentation Mask')
        # ax[1].axis('off')

        # plt.tight_layout()
        # plt.show()

if __name__ == "__main__":
    main()
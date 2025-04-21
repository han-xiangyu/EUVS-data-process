import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_image(image_path):
    """
    Load an image from the specified path.
    """
    return Image.open(image_path)

def load_mask(mask_path):
    """
    Load the npy mask file from the specified path.
    """
    return np.load(mask_path)

def visualize_mask(image, mask, title="Mask Verification"):
    """
    Visualize the mask overlaid on the image.
    """
    plt.figure(figsize=(10, 5))

    # Display the original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')

    # Display the mask
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Mask (0 for dynamic objects, 1 for static)")
    plt.axis('off')

    plt.suptitle(title)
    plt.show()

def check_dynamic_objects(mask):
    """
    Check if unwanted dynamic objects are correctly labeled as 0 in the mask.
    """
    unique_values = np.unique(mask)
    print(f"Unique values in the mask: {unique_values}")
    
    if 0 in unique_values and 1 in unique_values:
        print("Mask contains both 0 (dynamic objects) and 1 (static objects).")
    elif 0 in unique_values:
        print("Mask only contains 0 (all dynamic objects).")
    elif 1 in unique_values:
        print("Mask only contains 1 (all static objects).")
    else:
        print("Mask contains unexpected values:", unique_values)


# Example usage
image_path = 'location_06_10days/downsampled_multitraversal/frony_only_level_1/images/trav_0_channel_1_img_002.jpg'  # Replace with your image path
mask_path = 'location_06_10days/downsampled_multitraversal/frony_only_level_1/seg_mask/trav_0_channel_1_img_002.npy'  # Replace with your npy mask path

# Load the image and mask
image = load_image(image_path)
mask = load_mask(mask_path)

# Visualize the image and mask
visualize_mask(image, mask)

# Check if unwanted dynamic objects are marked as 0
check_dynamic_objects(mask)

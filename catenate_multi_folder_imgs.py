
    # base_folders = [
    #     "/home/xiangyu/Common/loc_15_case_2_T_cross/models/3DGS/test/ours_30000/gt", "/home/xiangyu/Common/loc_15_case_2_T_cross/models/3DGS/test/ours_30000/renders", "/home/xiangyu/Common/loc_15_case_2_T_cross/models/3DGS/test/ours_30000/feat_rgb_denoised_dinov2_base_c64_w110_h180",
    #     "/home/xiangyu/Common/loc_15_case_2_T_cross/models/GS_pro/test/ours_30000/renders", "/home/xiangyu/Common/loc_15_case_2_T_cross/models/GS_pro/test/ours_30000/feat_rgb_denoised_dinov2_base_c64_w110_h180", "/home/xiangyu/Common/loc_15_case_2_T_cross/models/VEGS/test/ours_30000/renders", "/home/xiangyu/Common/loc_15_case_2_T_cross/models/VEGS/test/ours_30000/feat_rgb_denoised_dinov2_base_c64_w110_h180"
    # ]
import os
import cv2
import numpy as np

def concatenate_images(image_list, rows, cols):
    assert len(image_list) == rows * cols, f"Expected {rows * cols} images, but got {len(image_list)}"
    row_images = []
    
    for i in range(rows):
        row_images.append(np.concatenate(image_list[i * cols:(i + 1) * cols], axis=1))
    
    return np.concatenate(row_images, axis=0)

def load_images_from_folders(base_folders, image_index, num_images_per_camera):
    images = []
    min_height, min_width = float('inf'), float('inf')
    
    # Load images and determine the minimum dimensions
    for folder in base_folders:
        folder_images = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))])
        left_image_path = folder_images[image_index + num_images_per_camera]  # Left camera images
        front_image_path = folder_images[image_index]  # Front camera images
        right_image_path = folder_images[image_index + 2 * num_images_per_camera]  # Right camera images
        
        for image_path in [left_image_path, front_image_path, right_image_path]:
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Image not found: {image_path}")
            images.append(image)
            height, width = image.shape[:2]
            min_height = min(min_height, height)
            min_width = min(min_width, width)
    
    # Resize all images to the minimum dimensions
    resized_images = [cv2.resize(image, (min_width, min_height)) for image in images]
    return resized_images

def main():

    base_folders = [
        "/home/xiangyu/Common/loc_15_case_2_T_cross/models/3DGS/test/ours_30000/gt", "/home/xiangyu/Common/loc_15_case_2_T_cross/models/3DGS/test/ours_30000/renders", "/home/xiangyu/Common/loc_15_case_2_T_cross/models/3DGS/test/ours_30000/feat_rgb_denoised_dinov2_base_c64_w110_h180",
        "/home/xiangyu/Common/loc_15_case_2_T_cross/models/GS_pro/test/ours_30000/renders", "/home/xiangyu/Common/loc_15_case_2_T_cross/models/GS_pro/test/ours_30000/feat_rgb_denoised_dinov2_base_c64_w110_h180", "/home/xiangyu/Common/loc_15_case_2_T_cross/models/VEGS/test/ours_30000/renders", "/home/xiangyu/Common/loc_15_case_2_T_cross/models/VEGS/test/ours_30000/feat_rgb_denoised_dinov2_base_c64_w110_h180"
    ]
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    # Assuming the number of images is the same in all folders
    num_images_per_camera = len(os.listdir(base_folders[0])) // 3

    for image_index in range(num_images_per_camera):
        try:
            images = load_images_from_folders(base_folders, image_index, num_images_per_camera)
            concatenated_image = concatenate_images(images, rows=7, cols=3)
            output_path = os.path.join(output_folder, f"concatenated_{image_index:04d}.jpg")
            cv2.imwrite(output_path, concatenated_image)
            print(f"Saved: {output_path}")
        except FileNotFoundError as e:
            print(e)

if __name__ == "__main__":
    main()





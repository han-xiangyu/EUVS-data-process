import cv2
import os
import numpy as np

# Define the paths to the folders
# folders = [
#     '/home/xiangyu/Ultra/MARS_multitraversal/location_62/downsampled_multitraversal/lane_change/models_aligned/3dgs/lane_change/test/ours_30000/gt',
#     '/home/xiangyu/Ultra/MARS_multitraversal/location_62/downsampled_multitraversal/lane_change/models_aligned/3dgm/lane_change/test/ours_30000/gt',
#     '/home/xiangyu/Ultra/MARS_multitraversal/location_62/downsampled_multitraversal/lane_change/models_aligned/gaussian_pro/lane_change/test/ours_30000/gt',
#     '/home/xiangyu/Ultra/MARS_multitraversal/location_62/downsampled_multitraversal/lane_change/models_aligned/3dgs/lane_change/test/ours_30000/renders',
#     '/home/xiangyu/Ultra/MARS_multitraversal/location_62/downsampled_multitraversal/lane_change/models_aligned/3dgm/lane_change/test/ours_30000/renders',
#     '/home/xiangyu/Ultra/MARS_multitraversal/location_62/downsampled_multitraversal/lane_change/models_aligned/gaussian_pro/lane_change/test/ours_30000/renders',
# ]
# folders = [
#     'location_62/downsampled_multitraversal/cross/models/3DGM/test/ours_30000/gt',
#     'location_62/downsampled_multitraversal/cross/models/3DGM/test/ours_30000/renders',
#     'location_62/downsampled_multitraversal/cross/models/3DGM/test/ours_30000/depth_map',

# ]
# num_images = 173 # Number of images to process
# frame_rate = 5  # Frames per second

folders = [
    'location_62/downsampled_multitraversal/cross/models/3DGM/train/ours_30000/gt',
    'location_62/downsampled_multitraversal/cross/models/3DGM/train/ours_30000/renders',
    'location_62/downsampled_multitraversal/cross/models/3DGM/train/ours_30000/depth_map',

]
num_images = 1029 # Number of images to process
frame_rate = 10  # Frames per second

# Function to get the image paths in sorted orderls

def get_sorted_image_paths(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    files.sort()  # Sort files to ensure correct order
    return files[:num_images]  # Return only the required number of images

# Read the first image to determine the size
sample_img_path = get_sorted_image_paths(folders[0])[0]
sample_img = cv2.imread(sample_img_path)
frame_height, frame_width, _ = sample_img.shape

# Calculate the video frame size for a 3x2 grid
video_frame_width = frame_width * 3
video_frame_height = frame_height * 1

# Video output setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
video = cv2.VideoWriter('location_62/downsampled_multitraversal/cross/models/3DGM/view_change_train.mp4', fourcc, frame_rate, (video_frame_width, video_frame_height))

# Read and process the images
for i in range(num_images):
    grid = []  # List to hold rows of images

    for row in range(1):  # Two rows
        row_images = []

        for col in range(3):  # Three columns
            folder_index = row * 1+ col
            image_paths = get_sorted_image_paths(folders[folder_index])
            
            if i < len(image_paths):
                img = cv2.imread(image_paths[i])
            else:
                img = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)  # Black image if index out of range

            row_images.append(img)

        # Concatenate images in this row horizontally
        row_images = np.hstack(row_images)
        grid.append(row_images)

    # Concatenate all rows vertically to form the grid
    video_frame = np.vstack(grid)
    video.write(video_frame)

# Release the video writer
video.release()
print("Video has been created successfully.")

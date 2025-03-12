import os


def count_images_in_input_subfolders(folder_paths):
    image_count_dict = {}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    total_image_count = 0

    for folder_path in folder_paths:
        input_folder = os.path.join(folder_path, 'input')
        if os.path.isdir(input_folder):
            image_count = sum(1 for file in os.listdir(input_folder) 
                              if os.path.isfile(os.path.join(input_folder, file)) 
                              and os.path.splitext(file)[1].lower() in image_extensions)
            image_count_dict[folder_path] = image_count
            total_image_count += image_count
        else:
            image_count_dict[folder_path] = 0  # No 'input' folder found

    return image_count_dict, total_image_count

# Example usage:
folder_paths = [
    # "/home/xiangyu/Common/loc_06_case_1",
    # "/home/xiangyu/Common/loc_07_case_1_T_cross",
    # "/home/xiangyu/Common/loc_11_case_1",
    # "/home/xiangyu/Common/loc_15_case_1",
    # "/home/xiangyu/Common/loc_15_case_2_T_cross",
    # "/home/xiangyu/Common/loc_24_case_1_T_cross",
    # "/home/xiangyu/Common/loc_37_case_1",
    # "/home/xiangyu/Common/loc_37_case_3_T_cross",
    # "/home/xiangyu/Common/loc_41_case_1_two_turns",
    # "/home/xiangyu/Common/loc_41_case_2_straight_and_right_turn",
    # "/home/xiangyu/Common/loc_41_case_3_straight_and_left_turn",
    # "/home/xiangyu/Common/loc_41_case_4_T_cross",
    # "/home/xiangyu/Common/loc_41_case_5_T_cross",
    # "/home/xiangyu/Common/loc_43_case_1_T_cross",
    # "/home/xiangyu/Common/loc_62_case_1_middle_lane_change",
    # "/home/xiangyu/Common/loc_62_case_2_right_lane_change",
    # "/home/xiangyu/Common/loc_62_case_3_cross" 
    "/home/xiangyu/Common/multi-agent/processed_data/scene_0",
    "/home/xiangyu/Common/multi-agent/processed_data/scene_1",
    "/home/xiangyu/Common/multi-agent/processed_data/scene_2",
    "/home/xiangyu/Common/multi-agent/processed_data/scene_3",
    "/home/xiangyu/Common/multi-agent/processed_data/scene_5",
    "/home/xiangyu/Common/multi-agent/processed_data/scene_6",
    "/home/xiangyu/Common/multi-agent/processed_data/scene_7_T_cross",
    "/home/xiangyu/Common/multi-agent/processed_data/scene_8",
    "/home/xiangyu/Common/multi-agent/processed_data/scene_9",
    "/home/xiangyu/Common/multi-agent/processed_data/scene_10",
    "/home/xiangyu/Common/multi-agent/processed_data/scene_11",
    "/home/xiangyu/Common/multi-agent/processed_data/scene_12",
    "/home/xiangyu/Common/multi-agent/processed_data/scene_13",
    "/home/xiangyu/Common/multi-agent/processed_data/scene_14",
    "/home/xiangyu/Common/multi-agent/processed_data/scene_15",
    "/home/xiangyu/Common/multi-agent/processed_data/scene_16"

]

image_counts, total_image_count = count_images_in_input_subfolders(folder_paths)
for folder, count in image_counts.items():
    print(f"{folder}: {count} images")

print(f"Total images: {total_image_count}")

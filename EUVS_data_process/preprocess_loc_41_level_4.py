
from nuscenes.nuscenes import NuScenes
import cv2
from utils import get_all_sample_tokens
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import json

def undistort_image(camera_matrix, distortion_coefficients, img):
    camera_matrix = np.array(camera_matrix)
    distortion_coefficients = np.array(distortion_coefficients)
    R = np.eye(3)
    img_size = (img.shape[1], img.shape[0])
    distortion_coefficients = np.array(distortion_coefficients, dtype=np.float32).reshape(-1, 1)
    mapx, mapy = cv2.fisheye.initUndistortRectifyMap(camera_matrix, distortion_coefficients, R, camera_matrix, img_size, cv2.CV_32FC1)
    resultImg = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    return resultImg

def write_list_to_txt(file_path, content_list):
    try:
        # 对 list 进行排序
        sorted_list = sorted(content_list)
        
        # 打开文件并逐行写入
        with open(file_path, 'a', encoding='utf-8') as file:
            for item in sorted_list:
                file.write(str(item) + '\n')  # 写入元素，并换行
        
        print(f"List successfully written to {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def process_data(nusc, output_dir, trainset_idxes, testset_idxes,  train_sensors, test_sensors, train_sample_ratio, test_sample_ratio):
    merged_traversal_idxes = trainset_idxes + testset_idxes
    num_of_scenes = len(nusc.scene)
    print(num_of_scenes)

    channel_to_idx = {
        'CAM_FRONT_CENTER': 1,
        'CAM_FRONT_LEFT' : 2,
        'CAM_FRONT_RIGHT': 3,
        'CAM_BACK_CENTER': 4,
        'CAM_SIDE_LEFT': 5,
        'CAM_SIDE_RIGHT': 6
    }
    sample_token_to_cam_pose_in_cam_frame = {}
    sample_token_to_geo_in_cam_frame = {}
    test_set_img_names = []
    train_set_img_names = []
    for traversal_idx in merged_traversal_idxes:
        my_scene = nusc.scene[traversal_idx]

        all_sample_tokens = get_all_sample_tokens(nusc,my_scene['token'])
        
        if traversal_idx in testset_idxes:
            sensors = test_sensors
            TESTSET = True
            sample_ratio = test_sample_ratio
        else:
            sensors = train_sensors
            TESTSET = False
            sample_ratio = train_sample_ratio




        image_output_dir = os.path.join(output_dir, 'input')
        os.makedirs(image_output_dir, exist_ok=True)
        lidar_output_dir = os.path.join(output_dir, 'lidar')
        os.makedirs(lidar_output_dir, exist_ok=True)
        test_txt_path = os.path.join(output_dir,'test_set.txt')
        train_txt_path = os.path.join(output_dir,'train_set.txt')

        # 定义每个 traversal 的视频输出路径
        video_output_dir = os.path.join(output_dir, 'videos')
        os.makedirs(video_output_dir, exist_ok=True)
        video_output_path = os.path.join(video_output_dir, f'traversal_{traversal_idx}_video.mp4')

        # 初始化视频写入对象（在后续代码中确定图像尺寸后初始化）
        video_writer = None
        fps = 10


        num_files = len(all_sample_tokens)
        print("The number of samples in traversal ",traversal_idx," is: ",num_files)
        skip_indices = np.linspace(0, num_files - 1, int(num_files * (1 - sample_ratio)), dtype=int)
        for idx, sample_token in enumerate(all_sample_tokens):
            images_list = []  # 存储三个 camera 图像
            if idx not in skip_indices: ########### Sample data in certain ratio ##############
                img_idx = str(idx+1).zfill(3)
                sample_record = nusc.get("sample", sample_token)

                if traversal_idx == 13 and idx>107:
                    continue
                elif traversal_idx == 9 and idx<446:
                    continue
                elif traversal_idx == 45 and 56<idx<268:
                    continue
                elif traversal_idx == 24 and 30<idx<640:
                    continue
                elif traversal_idx == 38 and 35<idx<558:
                    continue
                elif traversal_idx == 48 and 30<idx<200:
                    continue
                elif traversal_idx == 47 and 35<idx<200:
                    continue
                # Get ego poses
                lidar_token = sample_record["data"]['LIDAR_FRONT_CENTER']
                sd_record_lid = nusc.get("sample_data", lidar_token)
                # 定义文件路径
                input_file, _, _ = nusc.get_sample_data(lidar_token)
                lidar_name = f"trav_{traversal_idx}_lidar_{img_idx}.pcd.bin"
                output_file = os.path.join(lidar_output_dir, lidar_name)  # 保存为新的文件
                # 打开原始文件进行读取，并将其写入新文件
                with open(input_file, 'rb') as f_in:
                    data = f_in.read()
                with open(output_file, 'wb') as f_out:
                    f_out.write(data)
                print(f"Lidar file {lidar_name} has saved in {output_file}")
                ego_record_lid = nusc.get("ego_pose", sd_record_lid["ego_pose_token"])
                ego_world_rotation = np.array(ego_record_lid["rotation"])
                ego_world_translation = np.array(ego_record_lid['translation'])
                ego_world_rotation_R = R.from_quat(ego_world_rotation,scalar_first=True)
                for sensor_channel in sensors:
                    channel_idx = str(channel_to_idx[sensor_channel]).zfill(1)
                    camera_token = sample_record['data'][sensor_channel]
                    camera_data = nusc.get('sample_data', camera_token)
                    image_path, boxes, _ = nusc.get_sample_data(camera_token)
                    camera_intrinsic = nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])['camera_intrinsic']
                    distortion_coefficients = nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])['distortion_coefficient']
                    image_name = f"trav_{traversal_idx}_channel_{channel_idx}_img_{img_idx}.jpg"
                    output_path = os.path.join(image_output_dir, image_name)
                    if TESTSET:
                        test_set_img_names.append(image_name)
                    else:
                        train_set_img_names.append(image_name)
                        
                    # if os.path.exists(image_path):
                    #     with open(image_path, 'rb') as src_file:
                    #         with open(output_path, 'wb') as dst_file:
                    #             dst_file.write(src_file.read())
                    if os.path.exists(image_path):
                        # 读取图像
                        with open(image_path, 'rb') as src_file:
                            img_array = np.asarray(bytearray(src_file.read()), dtype=np.uint8)
                            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                            # 对图像进行去畸变处理
                            undistorted_img = undistort_image(camera_intrinsic, distortion_coefficients, img)
                            # 将去畸变后的图像写入输出路径
                            cv2.imwrite(output_path, undistorted_img)
                            # 将图像添加到 images_list 列表中
                            images_list.append(undistorted_img)
                        print(f"Saved image to {output_path}")
                    else:
                        print(f"Image file {image_path} does not exist.")
                    # Output image-pose pairs
                    sample_data = nusc.get('sample_data', sample_record['data'][sensor_channel])
                    calibrated_sensor_token = sample_data['calibrated_sensor_token']
                    cs_record = nusc.get("calibrated_sensor", calibrated_sensor_token)
                    cam_ego_rotation = R.from_quat(np.array(cs_record['rotation']),scalar_first=True)
                    cam_ego_translation = np.array(cs_record['translation'])
                    # Convert world to camera
                    cam_translation_world = ego_world_rotation_R.apply(cam_ego_translation) + ego_world_translation
                    cam_rotation_world = ego_world_rotation_R * cam_ego_rotation
                    cam_rotation_cam = cam_rotation_world.inv()
                    translation_camera = -cam_rotation_cam.apply(cam_translation_world)
                    # Restore corresponding camera's poses
                    key = f"trav_{traversal_idx}_channel_{channel_idx}_img_{img_idx}"
                    sample_token_to_cam_pose_in_cam_frame[key] = np.append(cam_rotation_cam.as_quat(scalar_first=True), translation_camera)    # key: {sample_token}_{camera_channel};    value: qw, qx, qy, qz, x,y,z
                    sample_token_to_geo_in_cam_frame[key] = cam_translation_world    # key: {sample_token}_{camera_channel};    value: x,y,z

                    if len(images_list) == len(sensors):
                        # 检查所有图像的形状是否一致
                        if all(img.shape == images_list[0].shape for img in images_list):
                            if len(images_list) == 1:
                                # 如果只有一个图像，则不需要重新排序或拼接，直接使用该图像
                                concatenated_img = images_list[0]
                            else:
                                # 对于多个图像，重新排序并拼接
                                images_reordered = [images_list[1], images_list[0], images_list[2]]
                                concatenated_img = cv2.hconcat(images_reordered)

                            # 如果 video_writer 尚未初始化，则根据第一个图像的尺寸初始化它
                            if video_writer is None:
                                height, width, _ = concatenated_img.shape
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

                            # 将拼接或单独的图像写入视频
                            video_writer.write(concatenated_img)
                            print(f"Added frame {idx} from traversal {traversal_idx} to video.")
                    else:
                        print(f"Image shapes do not match or incomplete images for frame {idx}, skipping.")

            else:
                print(f'Skipped due to sample rate condition')
        # 在所有图像处理完成后，释放视频写入对象
        if video_writer is not None:
            video_writer.release()
            print(f"Video for traversal {traversal_idx} saved at {video_output_path}")

    # print training set and test set numbers
    print("The number of training set is: ", len(sample_token_to_cam_pose_in_cam_frame)-len(test_set_img_names))
    print("The number of test set is: ", len(test_set_img_names))
    # write information txts
    write_list_to_txt(test_txt_path,test_set_img_names)
    write_list_to_txt(train_txt_path,train_set_img_names)
    # Write all image-pose pairs
    pose_folder_path = os.path.join(output_dir,'poses')
    os.makedirs(pose_folder_path, exist_ok=True)
    img_pose_path = os.path.join(pose_folder_path,'images.txt')
    write_in_image_pose_pairs(img_pose_path, sample_token_to_cam_pose_in_cam_frame, TEST_FLAG=False)

    pose_folder_path = os.path.join(output_dir,'geo_registration')
    os.makedirs(pose_folder_path, exist_ok=True)
    img_pose_path = os.path.join(pose_folder_path,'geo_registration.txt')
    write_in_image_geo_pairs(img_pose_path, sample_token_to_geo_in_cam_frame)

    key_info = {
    "train_set_traversals": trainset_idxes,
    "train_set_camera": train_sensors,
    "train_set_downsaple_ratio": train_sample_ratio,
    "number_of_train_set": len(sample_token_to_cam_pose_in_cam_frame)-len(test_set_img_names),
    "test_set_traversals": testset_idxes,
    "test_set_camera": test_sensors,
    "test_set_downsaple_ratio": test_sample_ratio,
    "number_of_test_set": len(test_set_img_names),
    }
    key_config_path = os.path.join(output_dir,'key_configs.txt')
    with open(key_config_path, 'w') as file:
        file.write(json.dumps(key_info, ensure_ascii=False, indent=4))  # ensure_ascii=False 用于保留中文字符


def write_in_image_pose_pairs(imgs_geo_path, sample_token_to_geo, TEST_FLAG, starting_num=0):
    
    # 按键排序并存储
    sorted_dict = {key: sample_token_to_geo[key] for key in sorted(sample_token_to_geo.keys())}
    j = 0
    camera_id = 1
    # Open file with write mode
    with open(imgs_geo_path, 'w') as f:
        if TEST_FLAG:
            j = starting_num
        else:
            j = 0
        for key, values in sorted_dict.items():
            j += 1
            # values 里面包含四元数和位移向量，确保它们是浮点数并格式化为字符串
            formatted_values = ' '.join(f"{float(v):.12g}" for v in values)
            # 生成一行数据并写入文件
            line = f"{j} {formatted_values} {camera_id} {key}.jpg\n\n"
            f.write(line)

def write_in_image_geo_pairs(imgs_geo_path, sample_token_to_geo):
    # 按键排序并存储
    sorted_dict = {key: sample_token_to_geo[key] for key in sorted(sample_token_to_geo.keys())}
    j = 0
    # Open file with write mode
    with open(imgs_geo_path, 'w') as f:
        j = 0
        for key, values in sorted_dict.items():
            j += 1
            # values 里面包含四元数和位移向量，确保它们是浮点数并格式化为字符串
            formatted_values = ' '.join(f"{float(v):.12g}" for v in values)
            # 生成一行数据并写入文件
            # line = f"{j} {formatted_values} {camera_id} {key}.jpg\n\n"
            line = f"{key}.jpg {formatted_values} \n\n"
            f.write(line)

def main():
    # The "version" variable is the name of the folder holding all .json metadata tables.
    nusc = NuScenes(version='v1.1', dataroot=f'./location_41/41', verbose=True)

    # # case 1 middle lane change
    # output_dir = '/home/xiangyu/Ultra/MARS_multitraversal/processed_data/loc_41_case_1_two_turns'
    # trainset_idxes = [23,16,26] # front all, sample rate 1/3
    # testset_idxes = [13]  # front only, sample rate 1
    # train_sample_ratio = 1/3
    # test_sample_ratio = 1
    # train_sensors = [
    #         'CAM_FRONT_CENTER',
    #         'CAM_FRONT_LEFT',
    #         'CAM_FRONT_RIGHT',
    #         # 'CAM_BACK_CENTER',
    #         # 'CAM_SIDE_LEFT',
    #         # 'CAM_SIDE_RIGHT',
    #         # 'LIDAR_FRONT_CENTER',
    #         # 'IMU_TOP'
    #         ]
    # test_sensors = [
    #         'CAM_FRONT_CENTER',
    #         # 'CAM_FRONT_LEFT',
    #         # 'CAM_FRONT_RIGHT',
    #         # 'CAM_BACK_CENTER',
    #         # 'CAM_SIDE_LEFT',
    #         # 'CAM_SIDE_RIGHT',
    #         # 'LIDAR_FRONT_CENTER',
    #         # 'IMU_TOP'
    #         ]
    

    # # case 2 right lane change
    # output_dir = '/home/xiangyu/Ultra/MARS_multitraversal/processed_data/loc_41_case_2_straight_and_right_turn'
    # trainset_idxes = [9,11,27,29] # front all, sample rate 2/3
    # testset_idxes = [13] # front all, sample rate 1/2
    # train_sample_ratio = 2/3
    # test_sample_ratio = 1/2
    # train_sensors = [
    #         'CAM_FRONT_CENTER',
    #         'CAM_FRONT_LEFT',
    #         'CAM_FRONT_RIGHT',
    #         # 'CAM_BACK_CENTER',
    #         # 'CAM_SIDE_LEFT',
    #         # 'CAM_SIDE_RIGHT',
    #         # 'LIDAR_FRONT_CENTER',
    #         # 'IMU_TOP'
    #         ]
    # test_sensors = [
    #         'CAM_FRONT_CENTER',
    #         'CAM_FRONT_LEFT',
    #         'CAM_FRONT_RIGHT',
    #         # 'CAM_BACK_CENTER',
    #         # 'CAM_SIDE_LEFT',
    #         # 'CAM_SIDE_RIGHT',
    #         # 'LIDAR_FRONT_CENTER',
    #         # 'IMU_TOP'
    #         ]

    # # case 3 cross
    # output_dir = '/home/xiangyu/Ultra/MARS_multitraversal/processed_data/loc_41_case_3_straight_and_left_turn'
    # trainset_idxes = [9,11,27,29] # front all, sample rate 7/12
    # testset_idxes = [16] # front all, sample rate 1/3
    # train_sample_ratio = 7/12
    # test_sample_ratio = 1/3
    # train_sensors = [
    #         'CAM_FRONT_CENTER',
    #         'CAM_FRONT_LEFT',
    #         'CAM_FRONT_RIGHT',
    #         # 'CAM_BACK_CENTER',
    #         # 'CAM_SIDE_LEFT',
    #         # 'CAM_SIDE_RIGHT',
    #         # 'LIDAR_FRONT_CENTER',
    #         # 'IMU_TOP'
    #         ]
    # test_sensors = [
    #         'CAM_FRONT_CENTER',
    #         'CAM_FRONT_LEFT',
    #         'CAM_FRONT_RIGHT',
    #         # 'CAM_BACK_CENTER',
    #         # 'CAM_SIDE_LEFT',
    #         # 'CAM_SIDE_RIGHT',
    #         # 'LIDAR_FRONT_CENTER',
    #         # 'IMU_TOP'
    #         ]

    # # case 4 T cross
    # output_dir = '/home/xiangyu/Ultra/MARS_multitraversal/processed_data/loc_41_case_4_T_cross'
    # trainset_idxes = [38,48,40,47] # front all, sample rate 7/12
    # testset_idxes = [24] # front all, sample rate 1/3
    # train_sample_ratio = 4/9
    # test_sample_ratio = 1/3
    # train_sensors = [
    #         'CAM_FRONT_CENTER',
    #         'CAM_FRONT_LEFT',
    #         'CAM_FRONT_RIGHT',
    #         # 'CAM_BACK_CENTER',
    #         # 'CAM_SIDE_LEFT',
    #         # 'CAM_SIDE_RIGHT',
    #         # 'LIDAR_FRONT_CENTER',
    #         # 'IMU_TOP'
    #         ]
    # test_sensors = [
    #         'CAM_FRONT_CENTER',
    #         'CAM_FRONT_LEFT',
    #         'CAM_FRONT_RIGHT',
    #         # 'CAM_BACK_CENTER',
    #         # 'CAM_SIDE_LEFT',
    #         # 'CAM_SIDE_RIGHT',
    #         # 'LIDAR_FRONT_CENTER',
    #         # 'IMU_TOP'
    #         ]
    
    # case 5 T cross
    output_dir = '/home/xiangyu/Ultra/MARS_multitraversal/processed_data/loc_41_case_5_T_cross'
    trainset_idxes = [38,48,40,47] # front all, sample rate 7/12
    testset_idxes = [45] # front all, sample rate 1
    train_sample_ratio = 4/9
    test_sample_ratio = 1
    train_sensors = [
            'CAM_FRONT_CENTER',
            'CAM_FRONT_LEFT',
            'CAM_FRONT_RIGHT',
            # 'CAM_BACK_CENTER',
            # 'CAM_SIDE_LEFT',
            # 'CAM_SIDE_RIGHT',
            # 'LIDAR_FRONT_CENTER',
            # 'IMU_TOP'
            ]
    test_sensors = [
            'CAM_FRONT_CENTER',
            # 'CAM_FRONT_LEFT',
            # 'CAM_FRONT_RIGHT',
            # 'CAM_BACK_CENTER',
            # 'CAM_SIDE_LEFT',
            # 'CAM_SIDE_RIGHT',
            # 'LIDAR_FRONT_CENTER',
            # 'IMU_TOP'
            ]
    
    process_data(nusc,output_dir,trainset_idxes, testset_idxes, train_sensors, test_sensors, train_sample_ratio, test_sample_ratio)


if __name__=="__main__":
    main()





# def main():
#     # The "version" variable is the name of the folder holding all .json metadata tables.
#     nusc = NuScenes(version='v1.1', dataroot=f'./location_41/41', verbose=True)

#     output_dir = './location_41/downsampled_multitraversal/case_1_left_turn_and_right_turn'
#     # traversal_idxes = list(range(0, 51))

#     # case 1 right turn and left turn
#     traversal_idxes = [23,16,26] # front all, sample rate 1/3
#     testset_idxes = [13] # front only, sample rate 1 sence 13

#     # case 2 straight annd right turn
#     # traversal_idxes = [9,11,27,29] # front all, sample rate 2/3
#     # testset_idxes = [13] # front all, sample rate 1/2
    
#     # case 3 straight annd left turn
#     # traversal_idxes = [9,11,27,29] # front all, sample rate 7/12
#     # testset_idxes = [16] # front all, sample rate 1/3

#     save_images(nusc,output_dir,traversal_idxes, testset_idxes)


# if __name__=="__main__":
#     main()
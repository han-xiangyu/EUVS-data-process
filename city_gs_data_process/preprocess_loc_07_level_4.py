
from nuscenes.nuscenes import NuScenes
import cv2
from utils import get_all_sample_tokens
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
from nuscenes.utils.data_classes import LidarPointCloud
from tqdm import tqdm
import open3d as o3d

from robot_vision.core.model_factory import ModelFactory
from pathlib import Path
import yaml
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def get_dynamic_point_mask(points_world, R_cw, t_cw, K, mask_img):
    """
    计算哪些世界坐标下的点投影到 mask=1 区域。

    Args:
        points_world: (N,3) np.ndarray，世界系点云
        R_cw: (3,3) np.ndarray，世界->相机旋转矩阵
        t_cw: (3,)  np.ndarray，世界->相机平移向量
        K:     (3,3) np.ndarray，相机内参矩阵
        mask_img: (H,W) 二值掩码，动态物体区域=1
    Returns:
        dynamic_mask: (N,) bool 数组，True 表示该点投影到动态区域
    """
    # 1. 变换到相机系
    pts_cam = (R_cw @ points_world.T + t_cw[:, None]).T
    z = pts_cam[:, 2]
    valid = z > 0

    # 2. 投影到像素平面
    proj = (K @ pts_cam[valid].T).T
    u = proj[:,0] / proj[:,2]
    v = proj[:,1] / proj[:,2]
    u_i = np.round(u).astype(int)
    v_i = np.round(v).astype(int)

    H, W = mask_img.shape[:2]
    in_bounds = (u_i>=0)&(u_i<W)&(v_i>=0)&(v_i<H)

    dynamic = np.zeros(points_world.shape[0], dtype=bool)
    idx_valid = np.nonzero(valid)[0]
    good_idx = idx_valid[in_bounds]
    dynamic_vals = mask_img[v_i[in_bounds], u_i[in_bounds]] > 0
    dynamic[good_idx] = dynamic_vals

    return dynamic

def filter_lidar_points(points_world, cam_params_list, masks_list):
    """
    用多视角 mask 过滤掉动态点。

    Args:
        points_world: (N,3) np.ndarray，世界系 LiDAR 点
        cam_params_list: list of dict，每个 dict 包含
            {
              'R_cw': np.ndarray (3,3),
              't_cw': np.ndarray (3,),
              'K':    np.ndarray (3,3)
            }
        masks_list: list of np.ndarray (H,W)，对应每个视角的二值掩码
    Returns:
        points_static: (M,3) np.ndarray，过滤掉动态点后的点云
    """
    N = points_world.shape[0]
    all_dynamic = np.zeros(N, dtype=bool)

    for params, mask in zip(cam_params_list, masks_list):
        dyn = get_dynamic_point_mask(
            points_world,
            params['R_cw'],
            params['t_cw'],
            params['K'],
            mask
        )
        all_dynamic |= dyn

    return points_world[~all_dynamic]



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

    # Path to the YAML file
    yaml_file = "./robot_vision/config/groundedSAM.yaml"

    # Read the YAML file
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    segmenter = ModelFactory.create("sam", config)
    
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
    shared_camera_params = None
    sample_token_to_cam_pose_in_cam_frame = {}
    sample_token_to_geo_in_cam_frame = {}
    test_set_img_names = []
    train_set_img_names = []
    all_world_lidar_points = [] # Initialize list to collect all points
    lidar_voxel_size = 0.5  # Voxel size for downsampling
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



        image_output_dir = os.path.join(output_dir, 'images')
        os.makedirs(image_output_dir, exist_ok=True)
        test_txt_path = os.path.join(output_dir,'test_set.txt')
        train_txt_path = os.path.join(output_dir,'train_set.txt')

        mask_output_dir = os.path.join(output_dir, 'dynamic_masks')
        os.makedirs(mask_output_dir, exist_ok=True)

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
        for idx, sample_token in enumerate(tqdm(all_sample_tokens, desc="Samples", unit="sample")):
            images_list = []  # 存储三个 camera 图像
            if idx not in skip_indices: ########### Sample data in certain ratio ##############
                img_idx = str(idx+1).zfill(3)
                sample_record = nusc.get("sample", sample_token)

                if traversal_idx == 5 and 80<idx<435:
                    continue
                elif traversal_idx == 7 and 80<idx<430:
                    continue
                elif traversal_idx == 32 and 96<idx<300:
                    continue
                # Get ego poses
                lidar_token = sample_record["data"]['LIDAR_FRONT_CENTER']
                lidar_sample_data = nusc.get("sample_data", lidar_token)

                # Read lidar data and convert
                lidar_file_path, _, _ = nusc.get_sample_data(lidar_token)
                pc = LidarPointCloud.from_file(lidar_file_path)


                # (a) 传感器 -> 车辆 的变换信息
                cs_record = nusc.get('calibrated_sensor', lidar_sample_data['calibrated_sensor_token'])
                sensor_to_ego_trans = np.array(cs_record['translation'])
                sensor_to_ego_rot = np.array(cs_record['rotation'])
                sensor_to_ego_rot_R = R.from_quat(sensor_to_ego_rot, scalar_first=True)

                # (b) 车辆 -> 世界 的变换信息
                pose_record = nusc.get('ego_pose', lidar_sample_data['ego_pose_token'])
                ego_world_translation = np.array(pose_record['translation'])
                ego_world_rotation = np.array(pose_record["rotation"])
                ego_world_rotation_R = R.from_quat(ego_world_rotation, scalar_first=True)

                # --- 应用变换: 传感器 -> 车辆 ---
                # LidarPointCloud 对象有内置的变换方法
                # 注意：通常先旋转，再平移
                pc.rotate(sensor_to_ego_rot_R.as_matrix())
                pc.translate(sensor_to_ego_trans)

                # --- 应用变换: 车辆 -> 世界 ---
                pc.rotate(ego_world_rotation_R.as_matrix())
                pc.translate(ego_world_translation)


                # --- 提取世界坐标系下的点 ---
                points_all = pc.points[:3, :].T  # 只取前3行，表示点的坐标

                # <<< --- ADD DOWNSAMPLING STEP --- >>>
                points_downsampled = points_all # Initialize to original points
                if lidar_voxel_size > 0 and points_all.shape[0] > 0:
                    try:
                        # 1. Create Open3D PointCloud object
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(points_all)

                        # 2. Apply Voxel Downsampling
                        downsampled_pcd = pcd.voxel_down_sample(voxel_size=lidar_voxel_size)

                        # 3. Convert back to NumPy array
                        points_downsampled = np.asarray(downsampled_pcd.points)
                        print(f"Downsampled frame LiDAR from {points_all.shape[0]} to {points_downsampled.shape[0]} points (voxel size: {lidar_voxel_size}m)")

                    except Exception as e:
                        print(f"Error during downsampling: {e}. Using original points for this frame.")
                        points_downsampled = points_all
                elif points_all.shape[0] == 0:
                     print("Warning: No points found in original LiDAR data for this frame.")
                     points_downsampled = points_all # Append empty array if it was empty
                # <<< --- END OF DOWNSAMPLING STEP --- >>>
                
                list_of_cam_rtks = []
                list_of_masks = []

                
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
                        

                    if os.path.exists(image_path):
                        # 读取图像
                        with open(image_path, 'rb') as src_file:
                            img_array = np.asarray(bytearray(src_file.read()), dtype=np.uint8)
                            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                            # 对图像进行去畸变处理
                            undistorted_img = undistort_image(camera_intrinsic, distortion_coefficients, img)
                            pil_image = Image.fromarray(undistorted_img)
                            mask, _, _ = segmenter.segment(pil_image)

                            # <<< —— ADD: 保存 mask —— >>>
                            # mask 目前是一个 H×W 的二值图（0/1），先转成 uint8，255 表示前景
                            mask_static = 1 - mask
                            mask_uint8 = (mask_static.astype(np.uint8) * 255)
                            mask_path = os.path.join(mask_output_dir, image_name)
                            cv2.imwrite(mask_path, mask_uint8)
                            # print(f"Saved mask to {mask_path}")

                            # 将去畸变后的图像写入输出路径
                            cv2.imwrite(output_path, undistorted_img)
                            # 将图像添加到 images_list 列表中
                            images_list.append(undistorted_img)
                        # print(f"Saved image to {output_path}")
                    else:
                        print(f"Image file {image_path} does not exist.")

                    # Camera intrinsic parameters
                    if shared_camera_params is None:
                        camera_intrinsic = np.array(camera_intrinsic)
                        fx = camera_intrinsic[0, 0]
                        fy = camera_intrinsic[1, 1]
                        cx = camera_intrinsic[0, 2]
                        cy = camera_intrinsic[1, 2]
                        h, w = undistorted_img.shape[:2]
                        shared_camera_params = ("PINHOLE", w, h, [fx, fy, cx, cy])

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

                    # For visualization
                    R_cw = cam_rotation_world.as_matrix()
                    t_cw = cam_translation_world

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



                    # Used for mask lidar points
                    list_of_cam_rtks.append([
                        R_cw,
                        t_cw,
                        camera_intrinsic
                    ])
                    # Read mask image 
                    list_of_masks.append(np.array(mask))
                    # print("list_of_cam_rtks is :", list_of_cam_rtks)
                
                cam_params_list = []
                masks_list = []

                for (R_cw, t_cw, K), mask_img in zip(list_of_cam_rtks, list_of_masks):
                    cam_params_list.append({
                        'R_cw': R_cw,
                        't_cw': t_cw,
                        'K': K
                    })
                    masks_list.append(mask_img)
                # print("cam_params_list is:", cam_params_list)
                # 过滤
                points_static = filter_lidar_points(points_downsampled, cam_params_list, masks_list)
                all_world_lidar_points.append(points_static)

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
    sparse_folder_path = os.path.join(output_dir,'sparse')
    colmap_data_path = os.path.join(sparse_folder_path,'0')
    os.makedirs(colmap_data_path, exist_ok=True)
    img_pose_path = os.path.join(colmap_data_path,'images.txt')
    write_in_image_pose_pairs(img_pose_path, sample_token_to_cam_pose_in_cam_frame, TEST_FLAG=False)

    geo_folder_path = os.path.join(output_dir,'geo_registration')
    os.makedirs(geo_folder_path, exist_ok=True)
    img_pose_path = os.path.join(geo_folder_path,'geo_registration.txt')
    write_in_image_geo_pairs(img_pose_path, sample_token_to_geo_in_cam_frame)


    # Aggregate all collected LiDAR points
    if all_world_lidar_points: # Check if the list is not empty
        print("Aggregating LiDAR points...")
        aggregated_points = np.concatenate(all_world_lidar_points, axis=0)
        print(f"Total aggregated LiDAR points: {aggregated_points.shape[0]}")

        # Define output path for points3D.txt
        points3d_output_path = os.path.join(colmap_data_path, 'points3D.txt')

        # Write the aggregated points to points3D.txt
        write_points3D_txt(points3d_output_path, aggregated_points)
    else:
        print("No LiDAR points were collected to write to points3D.txt.")


    cameras_txt_path = os.path.join(colmap_data_path, 'cameras.txt')
    with open(cameras_txt_path, 'w') as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS = fx, fy, cx, cy\n")
        model, w, h, params = shared_camera_params
        params_str = " ".join(f"{p:.6f}" for p in params)
        f.write(f"1 {model} {w} {h} {params_str}\n")
    print(f"Wrote single-entry cameras.txt to {cameras_txt_path}")

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
            f.write(f"{j} {formatted_values} {camera_id} {key}.jpg\n")
            f.write("\n")

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

def write_points3D_txt(filepath, points_array, default_rgb=(128, 128, 128), default_error=1.0):
    """
    Writes a NumPy array of 3D points to COLMAP's points3D.txt format.

    Args:
        filepath (str): The path to the output points3D.txt file.
        points_array (np.ndarray): A NumPy array of shape (N, 3) containing X, Y, Z coordinates.
        default_rgb (tuple, optional): Default RGB color tuple (0-255). Defaults to (128, 128, 128).
        default_error (float, optional): Default error value. Defaults to 1.0.
    """
    print(f"Writing {points_array.shape[0]} points to {filepath}...")
    r, g, b = default_rgb
    error = default_error
    point3D_id_counter = 1 # COLMAP IDs usually start from 1

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # Write header comments required by COLMAP
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            f.write(f"# Number of points: {points_array.shape[0]}, mean track length: 0\n") # Placeholder track length

            # Iterate through points and write formatted lines
            for i in tqdm(range(points_array.shape[0]), desc="Writing points3D.txt"):
                x, y, z = points_array[i, 0], points_array[i, 1], points_array[i, 2]
                # Format: POINT3D_ID X Y Z R G B ERROR (Track is empty)
                line = f"{point3D_id_counter} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {error:.6f}\n"
                f.write(line)
                point3D_id_counter += 1
        print(f"Successfully wrote points3D.txt to {filepath}")
    except Exception as e:
        print(f"An error occurred while writing points3D.txt: {e}")



def main():
    # The "version" variable is the name of the folder holding all .json metadata tables.
    nusc = NuScenes(version='v1.1', dataroot=f'/home/neptune/Data/MARS/raw_data/loc7/7', verbose=True)

    # case 1 middle lane change
    output_dir = '/home/neptune/Data/MARS/city_gs_data/loc07_single_trav_mask_lidar'
    # trainset_idxes = [5,6,7,10,12,13] 
    trainset_idxes = [6] 
    testset_idxes = [] 
    train_sample_ratio = 1
    test_sample_ratio = 0
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
            # 'CAM_FRONT_CENTER',
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




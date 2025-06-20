
from nuscenes.nuscenes import NuScenes
import cv2
from city_gs_data_process.utils import get_all_sample_tokens
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

def process_data(output_dir, channel_to_idx, train_sensors, test_sensors, spatial_step_m, target_frames):
    # ------------- Set parameters -------------
    lidar_voxel_size = 0.5  # Voxel size for downsampling
    fps = 10

    # ------------- Create output directories -------------
    key_config_path = os.path.join(output_dir,'key_configs.txt')
    image_output_dir = os.path.join(output_dir, 'images')
    os.makedirs(image_output_dir, exist_ok=True)
    test_txt_path = os.path.join(output_dir,'test_set.txt')
    train_txt_path = os.path.join(output_dir,'train_set.txt')
    mask_output_dir = os.path.join(output_dir, 'dynamic_masks')
    os.makedirs(mask_output_dir, exist_ok=True)
    video_output_dir = os.path.join(output_dir, 'videos')
    os.makedirs(video_output_dir, exist_ok=True)
    lidar_on_image_debug_dir = os.path.join(output_dir, 'debug_lidar_on_image_vis')
    os.makedirs(video_output_dir, exist_ok=True)
    geo_folder_path = os.path.join(output_dir,'geo_registration')
    os.makedirs(geo_folder_path, exist_ok=True)
    img_geo_path = os.path.join(geo_folder_path,'geo_registration.txt')
    # Create sparse folder for COLMAP
    sparse_folder_path = os.path.join(output_dir,'sparse')
    os.makedirs(sparse_folder_path, exist_ok=True)
    colmap_data_path = os.path.join(sparse_folder_path,'0')
    os.makedirs(colmap_data_path, exist_ok=True)
    img_pose_path = os.path.join(colmap_data_path,'images.txt')
    points3d_output_path = os.path.join(colmap_data_path, 'points3D.txt')
    cameras_txt_path = os.path.join(colmap_data_path, 'cameras.txt')

    # ------------- Load groundedSAM model -----------
    # Path to the YAML file
    yaml_file = "./robot_vision/config/groundedSAM.yaml"
    # Read the YAML file
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    segmenter = ModelFactory.create("sam", config)
    
    # ------------- Initialize savers -------------
    shared_camera_params = None
    sample_token_to_cam_pose_in_cam_frame = {}
    sample_token_to_geo_in_cam_frame = {}
    # all_world_lidar_points = []
    all_point_infos = []
    test_set_img_names = []
    train_set_img_names = []

    trainset_idxes = [2]
    testset_idxes = []
    merged_traversal_idxes = trainset_idxes + testset_idxes
    nusc = NuScenes(version='v1.1', dataroot=f'/lustre/fsw/portfolios/nvr/users/ymingli/gaussian/data', verbose=True)

    #  ----------- Loop each traversal -----------
    for traversal_idx in merged_traversal_idxes:
        my_scene = nusc.scene[traversal_idx]

        all_sample_tokens = get_all_sample_tokens(nusc,my_scene['token'])
        # ──── 1. 提前解析 ego-pose ────
        ego_xyz = []
        for tok in all_sample_tokens:
            sample_rec  = nusc.get("sample", tok)
            lidar_tok   = sample_rec["data"]["LIDAR_FRONT_CENTER"]
            lidar_sd    = nusc.get("sample_data", lidar_tok)
            pose_rec    = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
            ego_xyz.append(np.array(pose_rec["translation"]))          # (x,y,z)

        # ──── 2. 设定参数 ────
        # spatial_step_m    = 0.5        # 物理步长 Δd (米)
        # target_frames   = 3000       # 目标帧数
        assert target_frames > 0 and spatial_step_m > 0

        # ──── 3. 选帧 ────
        selected_idx = [0]      # 永远保留第一帧
        last_xyz     = ego_xyz[0]

        for i in range(1, len(ego_xyz)):
            if len(selected_idx) == target_frames:          # 已达到目标帧数
                break
            step = np.linalg.norm(ego_xyz[i] - last_xyz)
            if step >= spatial_step_m:                        # 又走了 ≥ Δd
                selected_idx.append(i)
                last_xyz = ego_xyz[i]

        # ──── 4. 裁剪 token 列表 ────
        all_sample_tokens = [all_sample_tokens[i] for i in selected_idx]
        num_files = len(all_sample_tokens)
        print(f"Spatial-sampling: kept {num_files} frames "
            f"(Δd={spatial_step_m} m, target {target_frames}, "
            f"path length ≈ {spatial_step_m*(num_files-1):.1f} m)")

        
        if traversal_idx in trainset_idxes:
            sensors = train_sensors
            TESTSET = False
            # sample_ratio = train_sample_ratio
        else:
            sensors = test_sensors
            TESTSET = True
            # sample_ratio = test_sample_ratio

        # Reset video writer for each traversal
        video_writer = None

        num_files = len(all_sample_tokens)
        print("The number of samples in traversal ",traversal_idx," is: ",num_files)
        # skip_indices = np.linspace(0, num_files - 1, int(num_files * (1 - sample_ratio)), dtype=int)

        # ------------- Loop each frame -------------
        for idx, sample_token in enumerate(tqdm(all_sample_tokens, desc="Samples", unit="sample")):
            images_list = []

            img_idx = str(idx+1).zfill(3)
            sample_record = nusc.get("sample", sample_token)

            # Get ego poses
            lidar_token = sample_record["data"]['LIDAR_FRONT_CENTER']
            lidar_sample_data = nusc.get("sample_data", lidar_token)

            # Read lidar data and convert
            lidar_file_path, _, _ = nusc.get_sample_data(lidar_token)
            pc = LidarPointCloud.from_file(lidar_file_path)


            # Lidar sensor -> vehicle
            cs_record = nusc.get('calibrated_sensor', lidar_sample_data['calibrated_sensor_token'])
            sensor_to_ego_trans = np.array(cs_record['translation'])
            sensor_to_ego_rot = np.array(cs_record['rotation'])
            sensor_to_ego_rot_R = R.from_quat(sensor_to_ego_rot, scalar_first=True)

            # Vehicle -> world
            pose_record = nusc.get('ego_pose', lidar_sample_data['ego_pose_token'])
            ego_world_translation = np.array(pose_record['translation'])
            ego_world_rotation = np.array(pose_record["rotation"])
            ego_world_rotation_R = R.from_quat(ego_world_rotation, scalar_first=True)

            # Apply transformation: vehicle -> sensor
            # LidarPointCloud embeds the method to rotate and translate the point cloud
            pc.rotate(sensor_to_ego_rot_R.as_matrix())
            pc.translate(sensor_to_ego_trans)

            # Apply transformation: sensor -> world
            pc.rotate(ego_world_rotation_R.as_matrix())
            pc.translate(ego_world_translation)

            # Get points in world coordinates
            points_all = pc.points[:3, :].T 

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

                except Exception as e:
                    print(f"Error during downsampling: {e}. Using original points for this frame.")
                    points_downsampled = points_all
            elif points_all.shape[0] == 0:
                print("Warning: No points found in original LiDAR data for this frame.")
                points_downsampled = points_all # Append empty array if it was empty
            # <<< --- END OF DOWNSAMPLING STEP --- >>>
            
            list_of_cam_parameters_per_frame = []
            list_of_masks_per_frame = []

            # --- Loop through each camera channel ---
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
                    
                # ------------- Read and save the image and mask -------------
                if os.path.exists(image_path):
                    with open(image_path, 'rb') as src_file:
                        img_array = np.asarray(bytearray(src_file.read()), dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        # Undistort and save the image
                        undistorted_img = undistort_image(camera_intrinsic, distortion_coefficients, img)
                        pil_image = Image.fromarray(undistorted_img)
                        mask, _, _ = segmenter.segment(pil_image)
                        cv2.imwrite(output_path, undistorted_img)
                        images_list.append(undistorted_img)
                        # Segment and save the mask
                        mask_static = 1 - mask # Original mask is dynamic, we need static
                        mask_uint8 = (mask_static.astype(np.uint8) * 255)
                        mask_path = os.path.join(mask_output_dir, image_name)
                        cv2.imwrite(mask_path, mask_uint8)
                else:
                    print(f"Image file {image_path} does not exist.")

                # ------------- Save the camera extrinsic and intrinsic parameters -------------
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
                wolrd_cam_rotation = cam_rotation_world.inv()
                world_camera_translation = -wolrd_cam_rotation.apply(cam_translation_world)
                # Restore corresponding camera's poses
                sample_token_to_cam_pose_in_cam_frame[image_name] = np.append(wolrd_cam_rotation.as_quat(scalar_first=True), world_camera_translation)    # key: {sample_token}_{camera_channel};    value: qw, qx, qy, qz, x,y,z
                sample_token_to_geo_in_cam_frame[image_name] = cam_translation_world    # key: {sample_token}_{camera_channel};    value: x,y,z

                # ------------- Save video from images -------------
                video_name = f'traversal_{traversal_idx}_video.mp4'
                if len(images_list) == len(sensors):
                    assert all(img.shape == images_list[0].shape for img in images_list)
                    # Concatenate images horizontally
                    images_reordered = [images_list[1], images_list[0], images_list[2]]
                    concatenated_img = cv2.hconcat(images_reordered)
                    video_output_path = os.path.join(video_output_dir, video_name)
                    # Initialize video writer if not already done
                    if video_writer is None:
                        height, width, _ = concatenated_img.shape
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
                    # Write the concatenated image to the video
                    video_writer.write(concatenated_img)

                # ------------- Save camera parameters and masks for lidar to filter out -------------
                R_wc = wolrd_cam_rotation.as_matrix()
                t_wc = world_camera_translation
                # Used for mask lidar points
                list_of_cam_parameters_per_frame.append({
                    'R_wc': R_wc,
                    't_wc': t_wc,
                    'K': camera_intrinsic,
                    'image_name': image_name,
                })
                # Mask image 
                list_of_masks_per_frame.append(np.array(mask))

            # ------------- Filter per frame LiDAR points using masks -------------
            points_downsampled = filter_lidar_points(points_downsampled, list_of_cam_parameters_per_frame, list_of_masks_per_frame)
            infos = visualize_and_collect(
                points_downsampled,
                list_of_cam_parameters_per_frame,
                images_list,
                lidar_on_image_debug_dir
            )
            all_point_infos.extend(infos)
            # all_world_lidar_points.append(points_static)

        # ------------- Release video writer after each traversal -------------
        if video_writer is not None:
            video_writer.release()

    # ------------- Summarize everything after all locations -------------
    print("The number of training set is: ", len(sample_token_to_cam_pose_in_cam_frame)-len(test_set_img_names))
    print("The number of test set is: ", len(test_set_img_names))
    # write information txts
    write_list_to_txt(test_txt_path,test_set_img_names)
    write_list_to_txt(train_txt_path,train_set_img_names)
    # Write all image-pose pairs
    write_in_image_pose_pairs(img_pose_path, sample_token_to_cam_pose_in_cam_frame, TEST_FLAG=False)
    write_in_image_geo_pairs(img_geo_path, sample_token_to_geo_in_cam_frame)

    sorted_images = sorted(sample_token_to_cam_pose_in_cam_frame.keys())
    image_name_to_id = { name: i+1 for i, name in enumerate(sorted_images) }
    # Aggregate all collected LiDAR points and write to points3D.txt
    if all_point_infos: # Check if the list is not empty
        # TARGET = 2**24 - 1                # 16_777_216
        # if len(all_point_infos) > TARGET:
        #     print(f"[INFO] Too many points ({len(all_point_infos)}), "
        #           f"randomly sampling {TARGET} points.")
        #     # Randomly sample TARGET points
        #     np.random.seed(42)
        #     keep_idx = np.random.choice(len(all_point_infos), TARGET, replace=False)
        #     all_point_infos = [all_point_infos[i] for i in keep_idx]
        # Write the aggregated points to points3D.txt
        write_points3D_txt_from_infos(points3d_output_path, all_point_infos, image_name_to_id)
    else:
        print("No LiDAR points were collected to write to points3D.txt.")

    # Write camera parameters to cameras.txt    
    with open(cameras_txt_path, 'w') as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS = fx, fy, cx, cy\n")
        model, w, h, params = shared_camera_params
        params_str = " ".join(f"{p:.6f}" for p in params)
        f.write(f"1 {model} {w} {h} {params_str}\n")
    print(f"Wrote single-entry cameras.txt to {cameras_txt_path}")

    # Write key information
    key_info = {
    "spatial_step_m": spatial_step_m,
    "num_target_frames": target_frames,
    "train_set_camera": train_sensors,
    "number_of_train_set": len(sample_token_to_cam_pose_in_cam_frame)-len(test_set_img_names),
    "test_set_camera": test_sensors,
    "number_of_test_set": len(test_set_img_names),
    "number_of_lidar_points": len(all_point_infos),
    }
    with open(key_config_path, 'w') as file:
        file.write(json.dumps(key_info, ensure_ascii=False, indent=4))


def filter_lidar_points_with_tracks(points_world, cam_params_list, masks_list, image_dir):
    """
    Like filter_lidar_points, but also returns for each kept point
    the list of (image_name, u, v) where it was visible & static.
    """
    N = points_world.shape[0]
    keep = np.zeros(N, dtype=bool)
    tracks = [[] for _ in range(N)]

    for params, mask_img in zip(cam_params_list, masks_list):
        R_wc, t_wc, K = params['R_wc'], params['t_wc'], params['K']
        image_name = params['image_name']
        # load the *undistorted* image so we can sample color later
        # img = cv2.imread(os.path.join(image_dir, image_name))
        H, W = mask_img.shape

        # 1) world→cam
        pts_cam = (R_wc @ points_world.T + t_wc[:, None]).T
        z = pts_cam[:,2]
        valid = z > 0
        if not np.any(valid): 
            continue

        # 2) project
        proj = (K @ pts_cam[valid].T).T
        u = np.round(proj[:,0] / proj[:,2]).astype(int)
        v = np.round(proj[:,1] / proj[:,2]).astype(int)
        in_bounds = (u>=0)&(u<W)&(v>=0)&(v<H)
        if not np.any(in_bounds):
            continue

        idx_valid = np.nonzero(valid)[0][in_bounds]
        dynamic = mask_img[v[in_bounds], u[in_bounds]]>0
        static_idx = idx_valid[~dynamic]

        # mark them as kept
        keep[static_idx] = True
        # record their pixel locations
        for pi, ui, vi in zip(static_idx, u[in_bounds][~dynamic], v[in_bounds][~dynamic]):
            tracks[pi].append((image_name, ui, vi))

    # filter out only the kept points
    kept_idx = np.nonzero(keep)[0]
    points_kept = points_world[kept_idx]
    tracks_kept = [tracks[i] for i in kept_idx]
    return points_kept, tracks_kept


def filter_lidar_points(points_world, cam_params_list, masks_list):
    """
    用多视角 mask 过滤 LiDAR 点，只保留落在任一相机视野内且不在动态区域的点。

    Args:
        points_world: (N,3) np.ndarray，世界系 LiDAR 点
        cam_params_list: list of dict，每个 dict 包含
            {
              'R_wc': np.ndarray (3,3),  # 世界->相机 旋转
              't_wc': np.ndarray (3,),   # 世界->相机 平移
              'K':    np.ndarray (3,3)   # 相机内参
            }
        masks_list: list of np.ndarray (H,W)，对应每个视角的二值动态 mask
    Returns:
        points_static: (M,3) np.ndarray，只保留可见且静态的点
    """
    N = points_world.shape[0]
    keep = np.zeros(N, dtype=bool)

    for params, mask_img in zip(cam_params_list, masks_list):
        R_wc = params['R_wc']
        t_wc = params['t_wc']
        K     = params['K']

        # 1. 投影前变换到相机系
        pts_cam = (R_wc @ points_world.T + t_wc[:, None]).T
        z = pts_cam[:, 2]
        valid = z > 0  # 只看相机前方
        if not np.any(valid):
            continue

        # 2. 投影到像素平面
        proj = (K @ pts_cam[valid].T).T
        u = proj[:, 0] / proj[:, 2]
        v = proj[:, 1] / proj[:, 2]
        u_i = np.round(u).astype(int)
        v_i = np.round(v).astype(int)

        H, W = mask_img.shape[:2]
        in_bounds = (u_i >= 0) & (u_i < W) & (v_i >= 0) & (v_i < H)
        if not np.any(in_bounds):
            continue

        # 3. 对应原始点的索引
        idx_valid = np.nonzero(valid)[0]
        idx_in_bounds = idx_valid[in_bounds]

        # 4. 动态检测：掩码中为 1 表示动态
        dynamic_vals = (mask_img[v_i[in_bounds], u_i[in_bounds]] > 0)

        # 5. 静态且可见的点
        static_idx = idx_in_bounds[~dynamic_vals]
        keep[static_idx] = True

    return points_world[keep]

# def visualize_lidar_on_image(points3d: np.ndarray,
#                             list_of_cam_parameters_per_frame: list,
#                             image_list: np.ndarray,
#                             folder_path: str,
#                             ):
#     """
#     Visualize LiDAR points on images.
#     Args:
#         points3d: (N,3) np.ndarray, 3D points in world coordinates
#         list_of_cam_parameters_per_frame: list of dict, each dict contains
#         image_list: list of np.ndarray, images to visualize on
#         folder_path: str, path to save the visualized images
#     """
#     for i in range(len(list_of_cam_parameters_per_frame)):
#         R_wc = list_of_cam_parameters_per_frame[i]['R_wc']
#         t_wc = list_of_cam_parameters_per_frame[i]['t_wc']
#         K = list_of_cam_parameters_per_frame[i]['K']
#         image_name = list_of_cam_parameters_per_frame[i]['image_name']
#         h, w = image_list[i].shape[:2]
#         # 1. 世界->相机
#         pts_cam = (R_wc @ points3d.T + t_wc[:, None]).T
#         # 2. 只保留 z>0
#         mask_front = pts_cam[:,2] > 0
#         pts_cam = pts_cam[mask_front]
#         # 3. 投影
#         proj = (K @ pts_cam.T).T
#         u = proj[:,0] / proj[:,2]
#         v = proj[:,1] / proj[:,2]
#         # 4. 落在图像内部
#         valid = (u>=0)&(u<w)&(v>=0)&(v<h)
#         u = u[valid].astype(int)
#         v = v[valid].astype(int)
#         # 5. 画点
#         vis = image_list[i].copy()
#         for x,y in zip(u,v):
#             cv2.circle(vis, (x,y), 1, (0,0,255), -1)
#         # 6. 保存
#         save_path = os.path.join(folder_path, image_name)
#         cv2.imwrite(save_path, vis)


def visualize_and_collect(points3d: np.ndarray,
                          cam_params: list,
                          image_list: list,
                          folder_path: str):
    """
    For each 3D point, find the first camera (in this frame)
    where it projects in front, inside the image & static,
    draw it, and record its (image_name,u,v,rgb).
    Returns a list of dicts, one per point.
    """
    N = points3d.shape[0]
    seen = np.zeros(N, bool)
    infos = []

    # loop each camera view once
    for idx, params in enumerate(cam_params):
        R_wc = params['R_wc']
        t_wc = params['t_wc']
        K    = params['K']
        name = params['image_name']
        img  = image_list[idx]
        mask = params.get('mask')   # if you also stash the mask here

        h, w = img.shape[:2]
        # world→cam
        P = (R_wc @ points3d.T + t_wc[:,None]).T
        z = P[:,2]
        valid = (z>0) & (~seen)
        if not np.any(valid):
            continue

        proj = (K @ P[valid].T).T
        u = np.round(proj[:,0] / proj[:,2]).astype(int)
        v = np.round(proj[:,1] / proj[:,2]).astype(int)
        idxs = np.nonzero(valid)[0]

        # draw & record
        vis = img.copy()
        for pi, ui, vi in zip(idxs, u, v):
            if ui<0 or vi<0 or ui>=w or vi>=h:
                continue
            if mask is not None and mask[vi,ui]>0:
                continue
            # first time we see this point?
            if not seen[pi]:
                seen[pi] = True
                b,g,r = img[vi,ui]
                infos.append({
                    "xyz": tuple(points3d[pi]),
                    "image": name,
                    "u_v": (int(ui), int(vi)),
                    "rgb": (int(r), int(g), int(b))
                })
            # draw it
            cv2.circle(vis, (ui,vi), 1, (0,0,255), -1)

        # save the debug overlay
        cv2.imwrite(os.path.join(folder_path, name), vis)

        if seen.all():
            break

    return infos


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
            f.write(f"{j} {formatted_values} {camera_id} {key}\n")
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
            line = f"{key} {formatted_values} \n\n"
            f.write(line)

# def write_points3D_txt(filepath, points_array, default_rgb=(128, 128, 128), default_error=1.0):
#     """
#     Writes a NumPy array of 3D points to COLMAP's points3D.txt format.

#     Args:
#         filepath (str): The path to the output points3D.txt file.
#         points_array (np.ndarray): A NumPy array of shape (N, 3) containing X, Y, Z coordinates.
#         default_rgb (tuple, optional): Default RGB color tuple (0-255). Defaults to (128, 128, 128).
#         default_error (float, optional): Default error value. Defaults to 1.0.
#     """
#     print(f"Writing {points_array.shape[0]} points to {filepath}...")
#     r, g, b = default_rgb
#     error = default_error
#     point3D_id_counter = 1 # COLMAP IDs usually start from 1

#     try:
#         with open(filepath, 'w', encoding='utf-8') as f:
#             # Write header comments required by COLMAP
#             f.write("# 3D point list with one line of data per point:\n")
#             f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
#             f.write(f"# Number of points: {points_array.shape[0]}, mean track length: 0\n") # Placeholder track length

#             # Iterate through points and write formatted lines
#             for i in tqdm(range(points_array.shape[0]), desc="Writing points3D.txt"):
#                 x, y, z = points_array[i, 0], points_array[i, 1], points_array[i, 2]
#                 # Format: POINT3D_ID X Y Z R G B ERROR (Track is empty)
#                 line = f"{point3D_id_counter} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {error:.6f}\n"
#                 f.write(line)
#                 point3D_id_counter += 1
#         print(f"Successfully wrote points3D.txt to {filepath}")
#     except Exception as e:
#         print(f"An error occurred while writing points3D.txt: {e}")

def write_points3D_txt_from_infos(filepath, infos, image_name_to_id, default_error=1.0):
    M = len(infos)
    shared_width = 1440
    print(f"Writing {M} points to {filepath}…")
    with open(filepath, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write(f"# Number of points: {M}, mean track length: 1\n")

        for pid, info in enumerate(tqdm(infos, desc="points3D.txt", unit="pt", total=M),start=1):
            x,y,z = info["xyz"]
            r,g,b = info["rgb"]
            img = info["image"]
            u,v = info["u_v"]
            img_id = image_name_to_id[img]
            # we still need some POINT2D_IDX; 
            # if you don’t have keypoint indexing, you can reuse u+v*w:
            pt2d_idx = v * shared_width + u

            f.write(
                f"{pid} {x:.6f} {y:.6f} {z:.6f} "
                f"{r} {g} {b} "
                f"{default_error:.6f} "
                f"{img_id} {pt2d_idx}\n"
            )
    print("Done.")


def main():

    output_dir = '/lustre/fsw/portfolios/nvr/users/ymingli/gaussian/data/spatial05_frames6000'

    # keys: [[trainset], [testset]]  

    # loc_and_traversals = {6: [[6], []]}
    # trav 11: [[5], []]
    
    # loc07_trainset_idxes = [5,6,7,10,12,13] 

    # # Three traversal
    # trainset_idxes = [6, 7, 10] # Sample ratio 0.4
    # train_sample_ratio = 0.4


    # train_sample_ratio = 0.2
    # test_sample_ratio = 0

    spatial_step_m = 0.5
    target_frames = 2000 # x3
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
    channel_to_idx = {
            'CAM_FRONT_CENTER': 1,
            'CAM_FRONT_LEFT' : 2,
            'CAM_FRONT_RIGHT': 3,
            'CAM_BACK_CENTER': 4,
            'CAM_SIDE_LEFT': 5,
            'CAM_SIDE_RIGHT': 6
        }
    
    process_data(output_dir, channel_to_idx, train_sensors, test_sensors, spatial_step_m, target_frames)


if __name__=="__main__":
    main()




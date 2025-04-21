import argparse
import json
import numpy as np
from nuscenes import NuScenes
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from PIL import Image
from nuscenes.utils.data_classes import LidarPointCloud
from scipy.spatial.transform import Rotation as R
import os
from tqdm import tqdm

def visualize_npy_file(npy_file_path):
    # Load the .npy file
    depth_map = np.load(npy_file_path)

    # Print some statistics about the depth map
    # print(f"Loaded depth map from {npy_file_path}")
    # print(f"depth_map shape: {depth_map.shape}")
    # print(f"depth_map max depth: {np.max(depth_map)}")
    # print(f"depth_map min depth: {np.min(depth_map)}")

    # Visualize the depth map
    plt.figure(figsize=(9, 16))
    plt.imshow(depth_map, cmap='plasma')
    plt.colorbar(label='Depth')
    plt.axis('off')
    plt.show()

def matrix_to_quaternion_translation(matrix):
    # Extract rotation matrix
    matrix = np.array(matrix)
    rotation_matrix = matrix[:3, :3]

    # Convert rotation matrix to quaternion, w, x, y, z
    rotation = R.from_matrix(rotation_matrix)
    quat = rotation.as_quat()
    quat = [quat[-1], quat[0], quat[1], quat[2]]

    # Extract translation
    translation = matrix[:3, 3]

    return quat, translation.tolist()


def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False

    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view
    # print("viewpad", viewpad)
    # print("intrinsics", view)

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]
    # print("points", points)

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points


def map_pointcloud_to_image_origin(root_dir, img_dir, pcd_dir, cam_sample, lidar_sample,seg_mask_path,sky_mask_path):
    # print(pcd_dir)
    # scene = pcd_dir.split("/")[-3]
    # meta_data = json.load(open(f'{root_dir}/data/{scene}/{scene}.json'))
    # cam_intrinsic = meta_data["calib"]["cam03"]["cam_intrinsic"]
    # print(cam_intrinsic)

    # cam_to_velo = np.array(meta_data["calib"]["cam03"]["cam_to_velo"])
    # print(cam_to_velo)
    # quat_cam3, trl_cam3 = matrix_to_quaternion_translation(cam_to_velo)
    # print(Quaternion(quat_cam3).rotation_matrix)

    pc = LidarPointCloud.from_file(pcd_dir)
    im = Image.open(img_dir)
    # print(pc.points)
    seg_mask=np.load(seg_mask_path)
    sky_mask=np.load(sky_mask_path)

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor', lidar_sample['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)  # should be 1, 0, 0, 0
    pc.translate(np.array(cs_record['translation']))  # should be 0

    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', lidar_sample['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam_sample['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))  # should be value same as second step, but inversely applied
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)  # should be value same as second step, but inversely applied

    # Fourth step: transform from ego into the camera.
    cs_record = nusc.get('calibrated_sensor', cam_sample['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]

    # Retrieve the color from the depth.
    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 1)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)

    # print(f"mask shape before applying seg_mask and sky_mask: {mask.shape}")
    # print(f"seg mask shape: {seg_mask.shape}")
    # print(seg_mask)

    mask = np.logical_and(mask, seg_mask[points[1, :].astype(int), points[0, :].astype(int)] == 1)
    mask = np.logical_and(mask, sky_mask[points[1, :].astype(int), points[0, :].astype(int)] == 1)

    points = points[:, mask]
    coloring = coloring[mask]

    fig, ax = plt.subplots(1, 1, figsize=(9, 16))
    fig.canvas.manager.set_window_title("sample")

    ax.imshow(im)
    ax.scatter(points[0, :], points[1, :], c=coloring, s=2)
    ax.axis('off')

    plt.show()

    # plt.savefig(f"{pcd_dir}/../img.jpg", bbox_inches='tight', pad_inches=0, dpi=200)

def map_pointcloud_to_image_with_mask(root_dir, img_dir, pcd_dir, cam_sample, lidar_sample, seg_mask_path, sky_mask_path):
    pc = LidarPointCloud.from_file(pcd_dir)
    im = Image.open(img_dir)

    # Load masks
    print(seg_mask_path)
    seg_mask = np.load(seg_mask_path,allow_pickle=True)
    sky_mask = np.load(sky_mask_path,allow_pickle=True)

    # Print the shapes of the masks and image
    print(f"seg_mask shape: {seg_mask.shape}")
    print(f"sky_mask shape: {sky_mask.shape}")
    print(f"image shape: {np.array(im).shape}")

    cs_record = nusc.get('calibrated_sensor', lidar_sample['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    poserecord = nusc.get('ego_pose', lidar_sample['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    poserecord = nusc.get('ego_pose', cam_sample['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    cs_record = nusc.get('calibrated_sensor', cam_sample['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    depths = pc.points[2, :]
    coloring = depths
    points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 1)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)

    # Print the shape of the mask before applying seg_mask and sky_mask
    print(f"mask shape before applying seg_mask and sky_mask: {mask.shape}")

    # 获取mask过滤后的有效点的索引
    y_indices = points[1, mask].astype(int)
    x_indices = points[0, mask].astype(int)

    # 打印x_indices和y_indices的形状
    print(f"x_indices shape: {x_indices.shape}")
    print(f"y_indices shape: {y_indices.shape}")

    # 应用分割掩码和天空掩码
    valid_seg_mask = seg_mask[y_indices, x_indices] == 1
    valid_sky_mask = sky_mask[y_indices, x_indices] == 1

    # 合并所有掩码
    mask[mask] = np.logical_and(valid_seg_mask, valid_sky_mask)

    # 打印应用seg_mask和sky_mask后的掩码形状
    print(f"mask shape after applying seg_mask and sky_mask: {mask.shape}")

    points = points[:, mask]
    print("final points shape",points.shape)
    coloring = coloring[mask]
    print("color:",coloring)

    fig, ax = plt.subplots(1, 1, figsize=(9, 16))
    fig.canvas.manager.set_window_title("sample")

    ax.imshow(im)
    ax.scatter(points[0, :], points[1, :], c=coloring, s=2)
    ax.axis('off')

    plt.show()




def generate_depth_gt_image(nusc, img_dir, pcd_dir, cam_sample, lidar_sample, seg_mask_path, sky_mask_path, output_path):
    pc = LidarPointCloud.from_file(pcd_dir)
    im = Image.open(img_dir)

    # Load masks with allow_pickle=True
    seg_mask = np.load(seg_mask_path, allow_pickle=True)
    sky_mask = np.load(sky_mask_path, allow_pickle=True)

    # Print the shapes of the masks and image
    # print(f"seg_mask shape: {seg_mask.shape}")
    # print(f"sky_mask shape: {sky_mask.shape}")
    # print(f"image shape: {np.array(im).shape}")

    cs_record = nusc.get('calibrated_sensor', lidar_sample['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    poserecord = nusc.get('ego_pose', lidar_sample['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    poserecord = nusc.get('ego_pose', cam_sample['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    cs_record = nusc.get('calibrated_sensor', cam_sample['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    depths = pc.points[2, :]
    points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 1)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)

    # Print the shape of the mask before applying seg_mask and sky_mask
    # print(f"mask shape before applying seg_mask and sky_mask: {mask.shape}")

    # 获取mask过滤后的有效点的索引
    y_indices = points[1, mask].astype(int)
    x_indices = points[0, mask].astype(int)

    # 打印x_indices和y_indices的形状
    # print(f"x_indices shape: {x_indices.shape}")
    # print(f"y_indices shape: {y_indices.shape}")

    # 应用分割掩码和天空掩码
    valid_seg_mask = seg_mask[y_indices, x_indices] == 1
    valid_sky_mask = sky_mask[y_indices, x_indices] == 1

    # 合并所有掩码
    valid_mask = np.logical_and(valid_seg_mask, valid_sky_mask)

    # 只保留有效点的深度信息
    valid_depths = depths[mask][valid_mask]
    valid_y_indices = y_indices[valid_mask]
    valid_x_indices = x_indices[valid_mask]

    # 初始化一个二维深度图，形状与seg_mask相同
    depth_map = np.zeros_like(seg_mask, dtype=np.float32)

    # 将深度信息赋值到depth_map的对应位置
    depth_map[valid_y_indices, valid_x_indices] = valid_depths

    # 打印depth_map的一些统计信息
    # print(f"depth_map shape: {depth_map.shape}")
    # print(f"depth_map max depth: {np.max(depth_map)}")
    # print(f"depth_map min depth: {np.min(depth_map)}")

    # 保存depth_map到npy文件
    np.save(output_path, depth_map)
    # print(f"Saved depth map to {output_path}")

"""show projected image"""
# if __name__ == '__main__':
#     root_dir = f'E:/dataset/MARS/41'
#     nusc = NuScenes(version='v1.1', dataroot=root_dir, verbose=True)
#     scene = nusc.scene[0]
#     my_sample = nusc.get("sample", scene["first_sample_token"])
#     current_sample_token = scene['first_sample_token']
#     output_folder = "E:/dataset/MARS/41/sweeps/temp"
#     os.makedirs(output_folder, exist_ok=True)
#     index = 0   
#     while current_sample_token:
#         my_sample = nusc.get('sample', current_sample_token)
#         nusc.render_pointcloud_in_image(my_sample['token'],
#                                     pointsensor_channel='LIDAR_FRONT_CENTER',
#                                     camera_channel='CAM_FRONT_CENTER',
#                                     render_intensity=False,
#                                     show_lidarseg=False,
#                                     out_path=f"{index}.jpg")
#         index=index+1



"""save projected image"""
# if __name__ == '__main__':
#     root_dir = 'E:/dataset/MARS/41'
#     nusc = NuScenes(version='v1.1', dataroot=root_dir, verbose=True)
#     scene = nusc.scene[0]
#     current_sample_token = scene['first_sample_token']
#     output_folder = 'E:/dataset/MARS/41/sweeps/temp'
#     os.makedirs(output_folder, exist_ok=True)
#     index = 0

#     while current_sample_token:
#         my_sample = nusc.get('sample', current_sample_token)
#         output_path = os.path.join(output_folder, f"{index}.jpg")

#         nusc.render_pointcloud_in_image(my_sample['token'],
#                                         pointsensor_channel='LIDAR_FRONT_CENTER',
#                                         camera_channel='CAM_FRONT_CENTER',
#                                         render_intensity=False,
#                                         show_lidarseg=False,
#                                         out_path=output_path,
#                                         verbose=True,
#                                         dot_size=2)

#         index += 1
#         current_sample_token = my_sample['next'] if 'next' in my_sample else None

"""generate depth gt"""


indexs = [18]
sensors = ['CAM_FRONT_CENTER']
image_count = 0
if __name__ == '__main__':
    root_dir = '/mnt/NAS/data/zj2640/MARS/24'
    nusc = NuScenes(version='v1.1', dataroot=root_dir, verbose=True)
    print("root_dir:",root_dir)
    all_data = []
    for traversal in tqdm(range(1,2), desc="Processing traversals"):
        base_folder = f"/mnt/NAS/data/zj2640/MARS/organized/loc24/orientation/exp/1234678train_5test"
        output_folder='/mnt/NAS/data/zj2640/gt_depth/Mars_loc24/5'
        seg_mask_basepath=os.path.join(base_folder,"seg_mask")
        sky_mask_basepath=os.path.join(base_folder,"sky_mask")
        seg_mask_files=sorted(os.listdir(seg_mask_basepath))
        sky_mask_files=sorted(os.listdir(sky_mask_basepath))
        os.makedirs(output_folder,exist_ok=True)
        for index in tqdm(indexs[:traversal], desc=f"Traversal {traversal} scenes"):
            scene = nusc.scene[index]
            current_sample_token = scene['first_sample_token']

            # 收集所有传感器的数据
            while current_sample_token:
                my_sample = nusc.get('sample', current_sample_token)
                lidar_data_token = my_sample["data"]["LIDAR_FRONT_CENTER"]
                lidar_sample = nusc.get("sample_data", lidar_data_token)
                pcd_dir = lidar_sample["filename"]
                pcd_dir = f"{root_dir}/{pcd_dir}"
                for sensor in sensors:
                    if image_count % 1== 0:
                        cam_data_token = my_sample["data"][sensor]
                        cam_sample = nusc.get("sample_data", cam_data_token)
                        img_dir = cam_sample["filename"]
                        img_dir = f"{root_dir}/{img_dir}"

                        base_name = os.path.basename(img_dir)
                        name = os.path.splitext(base_name)[0] + '.npy'
                        seg_mask_path = os.path.join(seg_mask_basepath, name)
                        sky_mask_path = os.path.join(sky_mask_basepath, name)

                        output_path = os.path.join(output_folder, name)
                        
                        all_data.append((img_dir, pcd_dir, cam_sample,lidar_sample,seg_mask_path,sky_mask_path,output_path))
                        index=index+1
                    image_count+=1
                current_sample_token = my_sample['next']

    # 按照图像名称排序
    all_data.sort(key=lambda x: x[0])

    for data in all_data:
        img_dir, pcd_dir, cam_sample,lidar_sample,seg_mask_path,sky_mask_path,output_path=data
        # map_pointcloud_to_image_with_mask(root_dir, img_dir=img_dir, pcd_dir=pcd_dir, cam_sample=cam_sample, lidar_sample=lidar_sample,seg_mask_path=seg_mask_path,sky_mask_path=sky_mask_path)
        generate_depth_gt_image(img_dir=img_dir, pcd_dir=pcd_dir, cam_sample=cam_sample, lidar_sample=lidar_sample,seg_mask_path=seg_mask_path,sky_mask_path=sky_mask_path,output_path=output_path)
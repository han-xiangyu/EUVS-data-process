import open3d as o3d
import numpy as np
from plyfile import PlyData
from tools.data_process.utils import get_all_sample_tokens
from nuscenes.nuscenes import NuScenes
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
import pickle
import os
from nuscenes.utils.data_classes import LidarPointCloud
from remove_bboxes import points_in_rbbox

def shuffle(points, n_points=500000):
#     points = np.concatenate(point_list)
    return points[np.random.permutation(points.shape[0])[:n_points]]


def visualize_lidar_gs_points():
    # Read the PLY file
    plydata = PlyData.read('/home/xiangyu/Projects/gaussian-splatting/output/f640614c-7/point_cloud/iteration_30000/point_cloud.ply')
    # /home/xiangyu/Ultra/LYFT_ROOT/3dgs_points/points/location_03/point_location_03.npy

    # Get vertex data
    vertex_data = plydata['vertex'].data 

    # Create a point cloud object
    pcd1 = o3d.geometry.PointCloud()

    # Extract vertex coordinates from the PLY data
    gs_points =  np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T 


    nusc = NuScenes(version='v1.0', dataroot=f'./MARS_agent_10Hz_40ms', verbose=True)
    scene_idx = 17
    my_scene = nusc.scene[scene_idx]

    all_sample_tokens = get_all_sample_tokens(nusc,my_scene['token'])

    all_points = np.empty((0, 3))

    for idx, sample_token in enumerate(all_sample_tokens):
        
        sample_record = nusc.get("sample", sample_token)
        lidar_token = sample_record["data"]['LIDAR_FRONT_CENTER_mike']
        data_path, boxes, _ = nusc.get_sample_data(lidar_token)
    
        pcd = LidarPointCloud.from_file(data_path)
        pts = pcd.points[:3].T

        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        dims *= 1.05
        rots = np.array(
            [b.orientation.yaw_pitch_roll[0] for b in boxes]
        ).reshape(-1, 1)
        gt_boxes_3d = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)


        point_indices = points_in_rbbox(
            pts, gt_boxes_3d, origin=(0.5, 0.5, 0.5)
        )
        point_indices = np.any(point_indices, axis=-1)

        pts = pts[np.logical_not(point_indices)]



        # ego to global
        sd_record_lid = nusc.get("sample_data", lidar_token)
        ego_record_lid = nusc.get("ego_pose", sd_record_lid["ego_pose_token"])
        ego2glo_translation = np.array(ego_record_lid['translation'])
        ego2glo_rotation = Quaternion(ego_record_lid['rotation']).rotation_matrix

        # lidar to ego
        cs_record = nusc.get('calibrated_sensor', sd_record_lid['calibrated_sensor_token'])
        lidar2ego_translation = np.array(cs_record['translation'])
        lidar2ego_rotation = Quaternion(cs_record['rotation']).rotation_matrix

        # get global lidar points
        points_in_ego = pts @ lidar2ego_rotation.T + lidar2ego_translation
        points_in_global = points_in_ego @ ego2glo_rotation.T + ego2glo_translation

        all_points = np.vstack((all_points,points_in_global))
    
    all_points = shuffle(all_points)

    
    # 创建第一个点云对象，并设置颜色
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(all_points)
    pcd1.paint_uniform_color([1, 0, 0])  # 红色

    # 创建第二个点云对象，并设置颜色
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(gs_points)
    pcd2.paint_uniform_color([0, 0, 1])  
    
    # 可视化两组点云、坐标系和网格
    o3d.visualization.draw_geometries([pcd1, pcd2])

def main():
    visualize_lidar_gs_points()

if __name__=="__main__":
    main()
    
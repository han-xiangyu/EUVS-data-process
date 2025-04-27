# import open3d as o3d
# import numpy as np
# from scipy.spatial.transform import Rotation as R

# def load_cameras(image_file):
#     cams = []
#     with open(image_file, 'r') as f:
#         for line in f:
#             if line.startswith('#'):
#                 continue
#             parts = line.split()
#             if len(parts) < 9:
#                 continue
#             try:
#                 img_id = int(parts[0])
#             except ValueError:
#                 continue
#             qw, qx, qy, qz = map(float, parts[1:5])
#             tx, ty, tz = map(float, parts[5:8])
#             # build camera-to-world transform
#             r = R.from_quat([qx, qy, qz, qw])
#             R_wc = r.as_matrix()
#             R_cw = R_wc.T
#             t = np.array([tx, ty, tz])
#             cam_pos = -R_cw.dot(t)
#             T = np.eye(4)
#             T[:3, :3] = R_cw
#             T[:3, 3] = cam_pos
#             cams.append(T)
#     return cams

# def load_points(points_file):
#     pts = []
#     with open(points_file, 'r') as f:
#         for line in f:
#             if line.startswith('#'):
#                 continue
#             parts = line.split()
#             if len(parts) < 4:
#                 continue
#             try:
#                 pt_id = int(parts[0])
#             except ValueError:
#                 continue
#             x, y, z = map(float, parts[1:4])
#             pts.append([x, y, z])
#     return np.array(pts)

# # Replace with your actual file paths
# images_path = "/home/neptune/Data/MARS/city_gs_data/loc6_10_17/sparse/0/images.txt"
# points_path = "/home/neptune/Data/MARS/city_gs_data/loc6_10_17/sparse/0/points3D.txt"

# # images_path = "/home/neptune/Data/MARS/city_gs_data/loc07/sparse/0/images.txt"
# # points_path = "/home/neptune/Data/MARS/city_gs_data/loc07/sparse/0/points3D.txt"

# cameras = load_cameras(images_path)
# points = load_points(points_path)

# # Create Open3D point cloud
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# # Optionally, color points gray
# pcd.colors = o3d.utility.Vector3dVector(np.tile([0.5, 0.5, 0.5], (points.shape[0], 1)))

# # Create camera frames
# frames = []
# for T in cameras:
#     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
#     frame.transform(T)
#     frames.append(frame)

# # Visualize
# o3d.visualization.draw_geometries([pcd, *frames])



import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import os

def load_cameras(image_file):
    cams = []
    with open(image_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                img_id = int(parts[0])
            except ValueError:
                continue
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz    = map(float, parts[5:8])
            # build camera-to-world transform
            r    = R.from_quat([qx, qy, qz, qw])
            R_wc = r.as_matrix()
            R_cw = R_wc.T
            t    = np.array([tx, ty, tz])
            cam_pos = -R_cw.dot(t)
            T = np.eye(4)
            T[:3, :3] = R_cw
            T[:3, 3] = cam_pos
            cams.append(T)
    return cams

def load_points_and_colors(points_file):
    pts   = []
    cols  = []
    with open(points_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            # parse X, Y, Z
            try:
                _ = int(parts[0])  # POINT3D_ID
            except ValueError:
                continue
            x, y, z = map(float, parts[1:4])
            r, g, b = map(int,   parts[4:7])
            pts.append([x, y, z])
            # normalize to [0,1]
            cols.append([r / 255.0, g / 255.0, b / 255.0])
    return np.array(pts), np.array(cols)

if __name__ == "__main__":
    # replace these with your actual paths:
    images_path = "/home/neptune/Data/MARS/city_gs_data/loc6_10_17_dense_voxel_035/sparse/0/images.txt"
    points_path = "/home/neptune/Data/MARS/city_gs_data/loc6_10_17_dense_voxel_035/sparse/0/points3D.txt"

    # load data
    cameras = load_cameras(images_path)
    points, colors = load_points_and_colors(points_path)

    # build point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # create camera frames
    frames = []
    for T in cameras:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        frame.transform(T)
        frames.append(frame)

    # visualize together
    o3d.visualization.draw_geometries([pcd, *frames])

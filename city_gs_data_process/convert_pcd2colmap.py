#!/usr/bin/env python3
"""
convert_pcd_to_colmap.py

Reads all .pcd.bin lidar point clouds in a folder and writes them
into COLMAP’s points3D.txt format:
POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
(Here we set R=G=B=0, ERROR=0, and leave TRACK empty.)
"""

import os
import glob
import numpy as np
import argparse

def load_pcd_bin(file_path):
    """
    Load a .pcd.bin file assumed to contain float32 per-point data.
    We try dims=3,4,5 channels; then keep only x,y,z.
    """
    data = np.fromfile(file_path, dtype=np.float32)
    for dim in (3, 4, 5):
        if data.size % dim == 0:
            pts = data.reshape(-1, dim)[:, :3]
            return pts
    raise ValueError(
        f"Unexpected number of floats ({data.size}) in {file_path}; "
        "not divisible by 3, 4 or 5."
    )


def write_points3d_txt(points, output_path):
    """
    Write a COLMAP‐style points3D.txt:
      POINT3D_ID, X, Y, Z, R, G, B, ERROR, [TRACK...]
    We set color=(0,0,0), error=0, and omit any track entries.
    """
    with open(output_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {points.shape[0]}\n")
        for pid, (x, y, z) in enumerate(points, start=1):
            # pid: unique ID; here we start at 1 and go up
            f.write(f"{pid} {x:.6f} {y:.6f} {z:.6f} 0 0 0 0\n")

def main():
    parser = argparse.ArgumentParser(
        description="Convert all .pcd.bin lidar files in a folder to a single COLMAP points3D.txt"
    )
    parser.add_argument("input_dir", help="Directory containing .pcd.bin files")
    parser.add_argument("output_file", help="Path to write points3D.txt")
    args = parser.parse_args()

    # Gather all pcd.bin files (recursively, if you like)
    pattern = os.path.join(args.input_dir, "**", "*.pcd.bin")
    bin_files = sorted(glob.glob(pattern, recursive=True))
    if not bin_files:
        print(f"No .pcd.bin files found under {args.input_dir}")
        return

    all_pts = []
    for bf in bin_files:
        pts = load_pcd_bin(bf)
        print(f"Loaded {pts.shape[0]} points from {os.path.basename(bf)}")
        all_pts.append(pts)
    all_pts = np.vstack(all_pts)

    write_points3d_txt(all_pts, args.output_file)
    print(f"Wrote {all_pts.shape[0]} total points to {args.output_file}")

if __name__ == "__main__":
    main()

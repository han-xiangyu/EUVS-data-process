import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from tqdm import tqdm

def read_scale_from_transform(file_path):
    """Reads the first line and extracts the first number as scale from transform.txt."""
    with open(file_path, 'r') as f:
        line = f.readline().strip()
        scale = float(line.split()[0])  # Convert the first number to float
    return scale

def evaluate(model_paths, source_paths, mode, gt_depth_dir, json_file_path):
    # Read the scale from transform.txt
    transform_txt_path = os.path.join(source_paths, "transform.txt")
    scale = read_scale_from_transform(transform_txt_path)

    # Set the rendered depth directory
    if mode == "test":
        rendered_depth_dir = os.path.join(model_paths, "test/ours_30000/render_depth_npy")
    elif mode == "train":
        rendered_depth_dir = os.path.join(model_paths, "train/ours_30000/render_depth_npy")
    else:
        print("The mode is not valid.")
        return

    # Check if the rendered depth directory exists
    if not os.path.exists(rendered_depth_dir):
        print(f"The directory {rendered_depth_dir} does not exist. Skipping {mode} mode.")
        return

    # Load and sort rendered depth files
    rendered_files = sorted(os.listdir(rendered_depth_dir), key=lambda f: int(''.join(filter(str.isdigit, f))))

    # Initialize metrics
    total_metrics = {
        "AbsRel": 0,
        "RMSE": 0,
        "RMSE_log": 0,
        "SqRel": 0,
        "delta_125": 0,
        "delta_125_2": 0,
        "delta_125_3": 0
    }
    count = 0

    for rendered_file in tqdm(rendered_files, desc=f"Calculating {mode} depth metrics"):
        cleaned_file = rendered_file.replace('.png', '')
        rendered_depth = np.load(os.path.join(rendered_depth_dir, rendered_file)) * scale
        gt_file = os.path.join(gt_depth_dir, cleaned_file)

        if not os.path.exists(gt_file):
            print(f"File {gt_file} does not exist.")
            continue

        gt_depth = np.load(gt_file)

        # Convert to tensors
        gt_depth_map = torch.tensor(gt_depth, dtype=torch.float32)
        rendered_depth = torch.tensor(rendered_depth, dtype=torch.float32).squeeze(0)

        # # Print original shapes
        # print(f"Original gt_depth_map shape: {gt_depth_map.shape}")
        # print(f"Original rendered_depth shape: {rendered_depth.shape}")

        # Resize rendered_depth to match gt_depth_map dimensions
        if rendered_depth.shape != gt_depth_map.shape:
            # print(f"Original rendered_depth shape: {rendered_depth.shape}")
            rendered_depth = F.interpolate(
                rendered_depth.unsqueeze(0).unsqueeze(0),
                size=gt_depth_map.shape,
                mode='nearest'
            ).squeeze()
            # print(f"Resized rendered_depth shape: {rendered_depth.shape}")

        # Optional: Resize both to a smaller size
        # new_height, new_width = gt_depth_map.shape[0] // 4, gt_depth_map.shape[1] // 4
        # gt_depth_map = F.interpolate(gt_depth_map.unsqueeze(0).unsqueeze(0), size=(new_height, new_width), mode='nearest').squeeze()
        # rendered_depth = F.interpolate(rendered_depth.unsqueeze(0).unsqueeze(0), size=(new_height, new_width), mode='nearest').squeeze()

        # Ensure both depth maps have the same shape after resizing
        assert gt_depth_map.shape == rendered_depth.shape, "Depth maps have mismatched shapes after resizing."

        # Proceed with mask creation and application
        valid_mask = (gt_depth_map > 0) & (gt_depth_map < 300)
        gt_depth_map = gt_depth_map[valid_mask]
        rendered_depth = rendered_depth[valid_mask]

        if gt_depth_map.numel() == 0:
            print(f"No valid pixels found in {cleaned_file}. Skipping.")
            continue

        # Calculate metrics
        abs_diff = torch.abs(gt_depth_map - rendered_depth)
        total_metrics["AbsRel"] += torch.mean(abs_diff / gt_depth_map).item()
        total_metrics["SqRel"] += torch.mean(((gt_depth_map - rendered_depth) ** 2) / gt_depth_map).item()
        total_metrics["RMSE"] += torch.sqrt(torch.mean((gt_depth_map - rendered_depth) ** 2)).item()
        total_metrics["RMSE_log"] += torch.sqrt(torch.mean((torch.log(gt_depth_map + 1e-6) - torch.log(rendered_depth + 1e-6)) ** 2)).item()

        max_ratio = torch.max(gt_depth_map / rendered_depth, rendered_depth / gt_depth_map)
        total_metrics["delta_125"] += torch.mean((max_ratio < 1.25).float()).item()
        total_metrics["delta_125_2"] += torch.mean((max_ratio < 1.25 ** 2).float()).item()
        total_metrics["delta_125_3"] += torch.mean((max_ratio < 1.25 ** 3).float()).item()

        count += 1

    if count == 0:
        print(f"No valid data found for {mode} mode.")
        return

    # Calculate mean metrics
    mean_metrics = {key: value / count for key, value in total_metrics.items()}

    # Write results to JSON file
    print(f"Writing the results to JSON file: {json_file_path}")
    with open(json_file_path, "w") as file:
        json.dump(mean_metrics, file, indent=4)
    print(f"Results written to {json_file_path}")



if __name__ == '__main__':
    parser = ArgumentParser(description="Evaluation script parameters")
    parser.add_argument('--model_paths', '-m', required=True, type=str)
    parser.add_argument('--source_paths', '-s', required=True, type=str)
    args = parser.parse_args()

    gt_depth_dir = os.path.join(args.source_paths, "depth_gt")

    # Evaluate both train and test modes
    for mode in ['train', 'test']:
        json_file_path = os.path.join(args.model_paths, f"depth_metrics_{mode}.json")
        os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
        evaluate(args.model_paths, args.source_paths, mode, gt_depth_dir, json_file_path)

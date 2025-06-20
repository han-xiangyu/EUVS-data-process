import os
import sys
import math

def evenly_sample(file_list, sample_ratio):
    total = len(file_list)
    num_samples = max(1, math.ceil(total * sample_ratio))
    if total <= num_samples:
        return file_list
    step = total / num_samples
    return [file_list[int(i * step)] for i in range(num_samples)]

def main():
    if len(sys.argv) != 2:
        print("Usage: python sample_test_set.py <target_directory>")
        return

    target_dir = sys.argv[1]
    test_set_path = os.path.join(target_dir, "test_set.txt")
    images_dir = os.path.join(target_dir, "images")

    if not os.path.isfile(test_set_path):
        print(f"Error: '{test_set_path}' does not exist.")
        return
    if not os.path.isdir(images_dir):
        print(f"Error: '{images_dir}' does not exist.")
        return

    # 读取图像列表
    with open(test_set_path, 'r') as f:
        all_images = [line.strip() for line in f if line.strip()]

    # 均匀采样 20%
    sampled_images = evenly_sample(all_images, sample_ratio=0.2)

    # 删除未采样的图像
    image_files = set(os.listdir(images_dir))
    to_keep = set(sampled_images)
    to_delete = image_files - to_keep

    for fname in to_delete:
        file_path = os.path.join(images_dir, fname)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # 更新 test_set.txt
    with open(test_set_path, 'w') as f:
        for name in sampled_images:
            f.write(name + '\n')

    print(f"✅ Sampled {len(sampled_images)} images (~20%) and updated '{test_set_path}' in '{target_dir}'.")

if __name__ == "__main__":
    main()
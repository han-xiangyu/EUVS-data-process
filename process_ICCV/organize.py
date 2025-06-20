import os
import shutil

# 📂 1. 源文件夹路径
source_base_path = "/mnt/NAS/home/zj2640/iccv_workshop/small/Level2/nuplan_vegas_location_28/vegas_location_28"
source_images_dir = os.path.join(source_base_path, "images")
source_sparse_dir = os.path.join(source_base_path, "sparse/0")

# 📂 2. 目标文件夹路径（完全不同的新位置）
target_base_path = "/mnt/NAS/home/zj2640/iccv_workshop/small/organized/Level2/nuplan_vegas_location_28"

# 创建 train/images 和 test/images 文件夹（在目标文件夹下）
train_images_dir = os.path.join(target_base_path, "train/images")
test_images_dir = os.path.join(target_base_path, "test/images")
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)

# 读取 train_set.txt 中的图片文件名
with open(os.path.join(source_base_path, "train_set.txt"), "r") as f:
    train_files = [line.strip() for line in f if line.strip()]

# 读取 test_set.txt 中的图片文件名
with open(os.path.join(source_base_path, "test_set.txt"), "r") as f:
    test_files = [line.strip() for line in f if line.strip()]

# 🟩 复制 train 集合的图片
for file_name in train_files:
    src = os.path.join(source_images_dir, file_name)
    dst = os.path.join(train_images_dir, file_name)
    if os.path.exists(src):
        shutil.copy2(src, dst)
    else:
        print(f"⚠️ {file_name} not found in source images directory!")

# 🟦 复制 test 集合的图片
for file_name in test_files:
    src = os.path.join(source_images_dir, file_name)
    dst = os.path.join(test_images_dir, file_name)
    if os.path.exists(src):
        shutil.copy2(src, dst)
    else:
        print(f"⚠️ {file_name} not found in source images directory!")

# 📝 复制 train_set.txt 和 test_set.txt 文件到目标 train/test 文件夹
shutil.copy2(os.path.join(source_base_path, "train_set.txt"), os.path.join(target_base_path, "train/train_set.txt"))
shutil.copy2(os.path.join(source_base_path, "test_set.txt"), os.path.join(target_base_path, "test/test_set.txt"))

# 🟧 在 test 文件夹下创建 sparse/0/ 目录
target_sparse_dir = os.path.join(target_base_path, "test/sparse/0")
os.makedirs(target_sparse_dir, exist_ok=True)

# 复制源 sparse/0/ 目录下的所有 txt 文件到目标 sparse/0/
for file_name in os.listdir(source_sparse_dir):
    if file_name.endswith(".txt"):
        src = os.path.join(source_sparse_dir, file_name)
        dst = os.path.join(target_sparse_dir, file_name)
        shutil.copy2(src, dst)

# 2️⃣ 目标 train/sparse/0/ 目录
train_sparse_dir = os.path.join(target_base_path, "train/sparse/0")
os.makedirs(train_sparse_dir, exist_ok=True)
print("train_sparse_dir:", train_sparse_dir)

source_images_txt = os.path.join(source_base_path, "sparse/0/images.txt")
# 3️⃣ 读取原 images.txt 文件的所有行
with open(source_images_txt, "r") as f:
    lines = [line.strip() for line in f]

# 4️⃣ 将所有图像数据块（两行）按 IMAGE NAME 建立索引
#    -> 结果：{NAME: (元数据行, POINTS2D行)}
image_data_blocks = {}
i = 0
while i < len(lines):
    line = lines[i]
    if line.startswith("#") or line == "":
        i += 1
        continue
    metadata_line = line
    points2d_line = lines[i+1] if i+1 < len(lines) else ""
    # 获取 NAME（最后一个字段）
    name = metadata_line.split()[-1]
    image_data_blocks[name] = (metadata_line, points2d_line)
    i += 2

# 5️⃣ 读取 train_set.txt 中的图片名称
with open(os.path.join(source_base_path, "train_set.txt"), "r") as f:
    train_image_names = [line.strip() for line in f if line.strip()]

# 6️⃣ 写入目标 train/sparse/0/images.txt 文件（不写 POINTS2D[]）
target_images_txt = os.path.join(train_sparse_dir, "images.txt")
with open(target_images_txt, "w") as f:
    # 写入文件头
    f.write("# Image list with two lines of data per image:\n")
    f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
    f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n\n")
    # 依次写入每个 train 中的图片元数据行和空行
    for name in train_image_names:
        if name in image_data_blocks:
            metadata_line, _ = image_data_blocks[name]
            f.write(metadata_line + "\n")
            f.write("\n")  # 只空行，不写入 POINTS2D
        else:
            print(f"⚠️ {name} not found in images.txt!")

print("✅ train/sparse/0/images.txt 文件已生成（仅包含元数据行）！")

# 🔴 1️⃣ 源 cameras.txt 文件路径
source_cameras_txt = os.path.join(source_base_path, "sparse/0/cameras.txt")

# 读取原 cameras.txt 的所有行
with open(source_cameras_txt, "r") as f:
    camera_lines = [line.strip() for line in f]

# 将 cameras 信息按 ID 建立索引（跳过注释行）
camera_data = {}  # {camera_id: camera_line}
for line in camera_lines:
    if line.startswith("#") or line == "":
        continue
    parts = line.split()
    camera_id = parts[0]
    camera_data[camera_id] = line

# 从 images.txt 中提取 train_set.txt 中每个图片对应的 camera id
source_images_txt = os.path.join(source_base_path, "sparse/0/images.txt")
train_image_camera_ids = set()

with open(source_images_txt, "r") as f:
    lines = [line.strip() for line in f]

i = 0
while i < len(lines):
    line = lines[i]
    if line.startswith("#") or line == "":
        i += 1
        continue
    metadata_line = line
    parts = metadata_line.split()
    camera_id = parts[-2]  # 倒数第二个字段是 camera_id
    name = parts[-1]
    if name in train_files:  # 只取 train_set.txt 中的图片
        train_image_camera_ids.add(camera_id)
    i += 2  # 跳过 points2d 行

# 写入 train/sparse/0/cameras.txt 文件
target_cameras_txt = os.path.join(train_sparse_dir, "cameras.txt")
with open(target_cameras_txt, "w") as f:
    # 先写 cameras.txt 中的注释行
    for line in camera_lines:
        if line.startswith("#"):
            f.write(line + "\n")
    f.write("\n")  # 注释行后空一行
    # 写入 train_set.txt 中涉及到的相机信息
    for cam_id in train_image_camera_ids:
        if cam_id in camera_data:
            f.write(camera_data[cam_id] + "\n")
        else:
            print(f"⚠️ Camera ID {cam_id} not found in cameras.txt!")

print("✅ train/sparse/0/cameras.txt 文件已生成，且只包含 train_set.txt 中使用到的相机信息！")

# 🟨 直接复制 points.txt 文件到 train/sparse/0/
source_points_txt = os.path.join(source_base_path, "sparse/0/points3D.txt")
target_points_txt = os.path.join(train_sparse_dir, "points3D.txt")

if os.path.exists(source_points_txt):
    shutil.copy2(source_points_txt, target_points_txt)
    print("✅ points.txt 文件已成功复制到 train/sparse/0/！")
else:
    print("⚠️ 源 points.txt 文件不存在，无法复制！")
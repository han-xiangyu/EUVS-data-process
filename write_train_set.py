import os

def create_train_set(source_path):
    """
    创建 train_set.txt 文件，其中包含从 images 文件夹中获取的所有图片名称，
    并排除在 test_set.txt 中列出的图像名称。
    """
    test_set_file = os.path.join(source_path, 'test_set.txt')
    images_folder = os.path.join(source_path, 'images')
    train_set_file = os.path.join(source_path, 'train_set.txt')

    # 确保 test_set.txt 存在
    if not os.path.isfile(test_set_file):
        print(f"错误：文件 {test_set_file} 不存在。")
        return

    # 确保 images 目录存在
    if not os.path.isdir(images_folder):
        print(f"错误：目录 {images_folder} 不存在。")
        return

    # 读取 test_set.txt 中的图像名称
    with open(test_set_file, 'r') as f:
        test_images = {line.strip() for line in f if line.strip()}  # 使用集合加快查找速度

    # 获取 images 文件夹中所有图像的名称
    # 假设所有图像文件都有扩展名
    all_images = sorted(os.listdir(images_folder))

    # 从 all_images 中排除 test_set 中的图像名称
    train_images = [img for img in all_images if img not in test_images]

    # 将 train_images 写入 train_set.txt 文件
    with open(train_set_file, 'w') as f:
        for image_name in train_images:
            f.write(image_name + '\n')

    print(f"train_set.txt 创建成功，路径：{train_set_file}")
    print(f"train_set.txt 包含 {len(train_images)} 个图像名称。")


# 示例使用
source_path = '/home/xiangyu/Common/EUVS_data/Level_1_nuplan/v_loc7_level1'  # 替换为你的源路径
create_train_set(source_path)

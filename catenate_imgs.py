from PIL import Image
import os

def stitch_images(input_folder, output_folder, n):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取文件夹中的所有图像文件并排序
    images = sorted([os.path.join(input_folder, img) for img in os.listdir(input_folder) if img.endswith(('.png', '.jpg', '.jpeg'))])

    # 检查是否有足够的图像
    if len(images) != 3 * n:
        raise ValueError(f"Expected {3 * n} images for {n} sets, but found {len(images)} images.")
    
    for i in range(n):
        # 获取左中右三张图像
        middle_image = Image.open(images[i])               # 中间相机图像
        left_image = Image.open(images[i + n])             # 左相机图像
        right_image = Image.open(images[i + 2 * n])        # 右相机图像

        # 确保图像大小相同
        if not (middle_image.size == left_image.size == right_image.size):
            raise ValueError("All images must have the same size.")

        # 拼接图像（水平拼接）
        total_width = middle_image.width * 3
        max_height = middle_image.height
        stitched_image = Image.new('RGB', (total_width, max_height))

        # 把左中右图像粘贴到拼接图像中
        stitched_image.paste(left_image, (0, 0))
        stitched_image.paste(middle_image, (middle_image.width, 0))
        stitched_image.paste(right_image, (2 * middle_image.width, 0))

        # 保存拼接后的图像
        output_path = os.path.join(output_folder, f"stitched_{i+1}.jpg")
        stitched_image.save(output_path)
        print(f"Saved stitched image {i+1} to {output_path}")

# 示例使用方法
input_folder = '/home/xiangyu/Projects/zipnerf-pytorch/exp/nuplan_level1/render/test_preds_step_25000'
output_folder = '/home/xiangyu/Projects/zipnerf-pytorch/exp/nuplan_level1/render/catenated_imgs'
n = 108  # 假设有5组左中右图像
stitch_images(input_folder, output_folder, n)

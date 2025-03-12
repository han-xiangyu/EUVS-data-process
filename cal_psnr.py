import os
import math
import cv2
import numpy as np

def calculate_psnr(img1, img2):
    """
    计算两张图片之间的PSNR值。
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        # 如果MSE为0，则表示两张图片完全相同
        return float('inf')
    PIXEL_MAX = 255.0
    psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return psnr

def compute_average_psnr(folder1, folder2):
    """
    读取两个文件夹中的所有图像，根据文件名匹配，然后计算PSNR并求平均。
    """
    # 获取文件夹1中的图像文件列表
    images_folder1 = sorted([f for f in os.listdir(folder1) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
    images_folder2 = sorted([f for f in os.listdir(folder2) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])

    total_psnr = 0.0
    count = 0

    for file_name in images_folder1:
        # 检查文件夹2中是否有对应的图像文件
        if file_name in images_folder2:
            # 读取两张匹配的图像
            img_path1 = os.path.join(folder1, file_name)
            img_path2 = os.path.join(folder2, file_name)

            img1 = cv2.imread(img_path1)
            img2 = cv2.imread(img_path2)

            if img1 is None or img2 is None:
                print(f"无法读取图像: {img_path1} 或 {img_path2}")
                continue

            # 检查两张图像的尺寸是否相同
            if img1.shape != img2.shape:
                print(f"图像尺寸不一致: {file_name}, {img1.shape} vs {img2.shape}")
                continue

            # 计算PSNR
            psnr = calculate_psnr(img1, img2)
            total_psnr += psnr
            count += 1
        else:
            print(f"文件夹2中没有对应图像: {file_name}")

    # 计算平均PSNR
    average_psnr = total_psnr / count if count > 0 else 0.0
    return average_psnr

# 示例调用
folder1_path = '/home/xiangyu/Common/loc_43_case_1_T_cross/models/3DGS/test/ours_30000/gt_feat'
folder2_path = '/home/xiangyu/Common/loc_43_case_1_T_cross/models/3DGS/test/ours_30000/render_feat'
average_psnr_value = compute_average_psnr(folder1_path, folder2_path)
print(f"平均PSNR: {average_psnr_value:.2f} dB")

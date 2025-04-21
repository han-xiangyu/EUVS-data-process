import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import re
from collections import defaultdict


def resize_and_crop_image(image, target_size):
    """
    将图片裁剪或缩放到指定大小。优先裁剪，如果图片比目标大，则居中裁剪，如果比目标小，则缩放。
    
    参数:
        image (numpy.ndarray): 输入图片。
        target_size (tuple): 目标大小 (height, width)。
    
    返回:
        resized_cropped_img (numpy.ndarray): 调整后的图片。
    """
    # 获取当前图片大小
    h, w = image.shape[:2]
    target_h, target_w = target_size

    # 如果图片尺寸与目标尺寸相同，直接返回
    if h == target_h and w == target_w:
        return image

    # 如果图片比目标大，则裁剪
    if h > target_h:
        top = (h - target_h) // 2
        image = image[top:top + target_h, :]
    if w > target_w:
        left = (w - target_w) // 2
        image = image[:, left:left + target_w]

    # 如果图片比目标小，则缩放
    image = cv2.resize(image, (target_w, target_h))

    return image


def calculate_log_img_counts_from_folder(folder_path):
    """
    从指定的图片文件夹中，计算每个 traversal 包含的帧数，生成 log_img_counts 列表。

    参数:
        folder_path (str): 图片文件夹的路径。

    返回:
        log_img_counts (list of int): 每个 traversal 包含的帧数列表，按 traversal 编号排序。
    """
    # 获取图片文件名
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))]
    image_files = sorted(image_files)

    # 分组
    traversal_groups = defaultdict(list)
    pattern = r'trav_(\d+)'  # 匹配 'trav_' 后的数字

    for filename in image_files:
        match = re.search(pattern, filename)
        if match:
            traversal_number = int(match.group(1))
            traversal_groups[traversal_number].append(filename)
        else:
            print(f"警告：无法从文件名 {filename} 中提取 traversal 编号")

    # 计算 log_img_counts
    log_img_counts = []
    traversal_numbers = sorted(traversal_groups.keys())
    for traversal_number in traversal_numbers:
        image_count = len(traversal_groups[traversal_number])
        if image_count % 3 != 0:
            print(f"警告：traversal {traversal_number} 的图片数量不是 3 的倍数")
        frame_count = image_count // 3
        log_img_counts.append(frame_count)

    return log_img_counts

# def read_and_concatenate_images(base_paths, frame_index, log_img_counts, target_size):
#     """
#     从文件夹中读取当前帧的图片，并按顺序拼接。
#     每段 log 中的图片顺序为：front -> left -> right，或只包含 front 和 left。

#     参数:
#         base_paths (list of str): 各相机图片文件夹路径列表。
#         frame_index (int): 当前帧的索引（按顺序）。
#         log_img_counts (list of int): 每段 log 包含的 front 相机图片数量。

#     返回:
#         concatenated_img (numpy.ndarray): 拼接后的图像网格，如果有图片缺失则返回 None。
#     """
#     all_row_images = []

#     for base_path in base_paths:
#         image_files = sorted(os.listdir(base_path))
#         cumulative_frame_count = 0
#         cumulative_image_count = 0

#         for log_count in log_img_counts:
#             if frame_index < cumulative_frame_count + log_count:
#                 local_frame_index = frame_index - cumulative_frame_count
#                 front_start = cumulative_image_count
#                 left_start = front_start + log_count
#                 right_start = left_start + log_count if len(image_files) >= left_start + log_count else None

#                 front_img_path = os.path.join(base_path, image_files[front_start + local_frame_index])
#                 left_img_path = os.path.join(base_path, image_files[left_start + local_frame_index])

#                 # 检查 right_start 是否有效
#                 if right_start and right_start + local_frame_index < len(image_files):
#                     right_img_path = os.path.join(base_path, image_files[right_start + local_frame_index])
#                 else:
#                     right_img_path = None
#                 break
#             else:
#                 cumulative_frame_count += log_count
#                 cumulative_image_count += log_count * (3 if right_start else 2)
#         else:
#             print(f"警告: 帧索引 {frame_index} 超出范围")
#             return None

#         front_img = cv2.imread(front_img_path)
#         left_img = cv2.imread(left_img_path)
#         right_img = cv2.imread(right_img_path) if right_img_path else None

#         if front_img is None or left_img is None:
#             print(f"警告: 无法读取图片 {front_img_path} 或 {left_img_path}")
#             return None

#         front_img = resize_and_crop_image(front_img, target_size)
#         left_img = resize_and_crop_image(left_img, target_size)
        
#         if right_img is not None:
#             right_img = resize_and_crop_image(right_img, target_size)
#             row_concatenated_img = np.hstack((left_img, front_img, right_img))
#         else:
#             row_concatenated_img = np.hstack((left_img, front_img))
        
#         all_row_images.append(row_concatenated_img)

#     concatenated_img = np.vstack(all_row_images)
#     return concatenated_img



def read_and_concatenate_images(base_paths, frame_index, log_img_counts, baseline_labels, target_size):
    """
    从多个文件夹中读取当前帧的左、中、右相机图片，并按顺序拼接。
    每段 log 中的图片顺序为：front -> left -> right。

    参数:
        base_paths (list of str): 不同类型图像的文件夹路径列表。
        frame_index (int): 当前帧的索引（按顺序）。
        log_img_counts (list of int): 每段 log 包含的 front 相机图片数量。
        baseline_labels (list of str): 每个基线的标签列表。

    返回:
        concatenated_img (numpy.ndarray): 拼接后的图像网格，如果有图片缺失则返回 None。
    """
    all_row_images = []

    for base_path, baseline_label in zip(base_paths, baseline_labels):
        # 获取当前文件夹的所有图片文件名
        image_files = sorted(os.listdir(base_path))

        cumulative_frame_count = 0  # 用于确定 frame_index 属于哪个 log
        cumulative_image_count = 0  # 用于计算图片的全局索引

        for log_count in log_img_counts:
            if frame_index < cumulative_frame_count + log_count:
                # 计算当前 frame 在当前 log 中的相对索引
                local_frame_index = frame_index - cumulative_frame_count

                # 计算当前 log 中 front, left, right 的起始索引
                front_start = cumulative_image_count
                left_start = front_start + log_count
                right_start = left_start + log_count

                # 获取图片文件名
                front_img_name = image_files[front_start + local_frame_index]
                left_img_name = image_files[left_start + local_frame_index]
                right_img_name = image_files[right_start + local_frame_index]

                # 获取图片路径
                front_img_path = os.path.join(base_path, front_img_name)
                left_img_path = os.path.join(base_path, left_img_name)
                right_img_path = os.path.join(base_path, right_img_name)
                break
            else:
                # 更新累计计数
                cumulative_frame_count += log_count
                cumulative_image_count += log_count * 3  # 每个 log 有 front, left, right 三种图片
        else:
            print(f"警告: 帧索引 {frame_index} 超出范围")
            return None

        # 读取图片
        front_img = cv2.imread(front_img_path)
        left_img = cv2.imread(left_img_path)
        right_img = cv2.imread(right_img_path)

        if front_img is None or left_img is None or right_img is None:
            print(f"警告: 无法读取图片 {front_img_path}, {left_img_path}, {right_img_path}")
            return None

        # 将所有图像调整为目标尺寸
        front_img = cv2.resize(front_img, (target_size[1], target_size[0]))
        left_img = cv2.resize(left_img, (target_size[1], target_size[0]))
        right_img = cv2.resize(right_img, (target_size[1], target_size[0]))

        # 在左摄像头图片左下角添加基线名称
        # 将 OpenCV 图片转换为 PIL Image
        left_img_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        left_img_pil = Image.fromarray(left_img_rgb)

        draw = ImageDraw.Draw(left_img_pil)
        # 设置字体路径，这里需要提供 Optima 字体的路径
        font_path = "/home/xiangyu/Ultra/Download/optima/OPTIMA.TTF"  # 请替换为实际的字体文件路径
        try:
            font = ImageFont.truetype(font_path, size=160)  # 调整字体大小
        except IOError:
            print(f"无法加载字体 {font_path}，使用默认字体")
            font = ImageFont.load_default()

        text = baseline_label

        # 获取文字尺寸
        try:
            # 如果您的 Pillow 版本支持 getbbox 方法
            bbox = font.getbbox(text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            # 如果 getbbox 方法不可用，尝试使用 getsize 或其他方法
            try:
                text_width, text_height = font.getsize(text)
            except AttributeError:
                print("无法获取文字尺寸，使用默认值")
                text_width, text_height = 100, 40  # 根据需要设置默认值

        # 计算文字位置，左下角，适当调整偏移量
        x = 10
        y = left_img_pil.height - text_height - 25
        # 添加文字
        draw.text((x, y), text, font=font, fill=(255, 255, 255))

        # 将 PIL Image 转换回 OpenCV 格式
        left_img_with_label = cv2.cvtColor(np.array(left_img_pil), cv2.COLOR_RGB2BGR)

        # 拼接左、中、右图片
        row_concatenated_img = np.hstack((left_img_with_label, front_img, right_img))
        all_row_images.append(row_concatenated_img)

    # 将所有基线的结果纵向拼接
    concatenated_img = np.vstack(all_row_images)
    return concatenated_img

# def generate_video_from_folders(base_paths, log_img_counts, output_video_path, baseline_labels, fps=30):
#     """
#     生成视频，将不同文件夹中的图片拼接为网格，并合成视频。

#     参数:
#         base_paths (list of str): 包含不同类型图像的文件夹路径列表（例如深度图、渲染图等）。
#         log_img_counts (list of int): 每段 log 包含的 front 相机图片数量。
#         output_video_path (str): 输出视频的路径。
#         baseline_labels (list of str): 每个基线的标签列表。
#         fps (int): 视频的帧率。
#     """
#     all_concatenated_images = []

#     # 计算所有 log 中总的帧数
#     total_frames = sum(log_img_counts)

#     for frame_index in range(total_frames):
#         concatenated_img = read_and_concatenate_images(base_paths, frame_index, log_img_counts, baseline_labels)
#         if concatenated_img is not None:
#             all_concatenated_images.append(concatenated_img)
#         else:
#             print(f"跳过帧索引 {frame_index} 的拼接")

#     if not all_concatenated_images:
#         print("没有有效的图片用于生成视频。")
#         return

#     # 获取拼接后图片的尺寸
#     height, width, layers = all_concatenated_images[0].shape

#     # 检查视频尺寸是否过大，可能需要调整
#     max_video_width = 1920
#     max_video_height = 1080
#     if width > max_video_width or height > max_video_height:
#         print("警告: 拼接后的视频尺寸过大，可能无法编码。请考虑减少基线数量或调整图片尺寸。")

#     # 定义视频编码器和视频写入对象
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

#     # 写入每一帧
#     for img in all_concatenated_images:
#         video_writer.write(img)

#     video_writer.release()
#     print(f"视频已保存到 {output_video_path}")

def generate_video_from_folders(base_paths, log_img_counts, output_video_path, baseline_labels, fps=30):
    """
    生成视频，将不同文件夹中的图片拼接为网格，并合成视频。

    参数:
        base_paths (list of str): 包含不同类型图像的文件夹路径列表（例如深度图、渲染图等）。
        log_img_counts (list of int): 每段 log 包含的 front 相机图片数量。
        output_video_path (str): 输出视频的路径。
        baseline_labels (list of str): 每个基线的标签列表。
        fps (int): 视频的帧率。
    """
    all_concatenated_images = []

    # 计算所有 log 中总的帧数
    total_frames = sum(log_img_counts)

    # 获取目标尺寸
    target_size = None

    # 从第一个基线的第一个文件夹中读取第一张图片，确定 target_size
    first_base_path = base_paths[0]
    image_files = sorted(os.listdir(first_base_path))
    if not image_files:
        print(f"路径 {first_base_path} 中没有图片")
        return
    first_image_path = os.path.join(first_base_path, image_files[0])
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        print(f"无法读取图片 {first_image_path}")
        return
    # 设置 target_size 为第一张图片的尺寸 (height, width)
    target_size = first_image.shape[:2]

    for frame_index in range(total_frames):
        concatenated_img = read_and_concatenate_images(base_paths, frame_index, log_img_counts, baseline_labels, target_size)
        if concatenated_img is not None:
            all_concatenated_images.append(concatenated_img)
        else:
            print(f"跳过帧索引 {frame_index} 的拼接")

    if not all_concatenated_images:
        print("没有有效的图片用于生成视频。")
        return

    # 获取拼接后图片的尺寸
    height, width, layers = all_concatenated_images[0].shape

    # 检查视频尺寸是否过大，可能需要调整
    max_video_width = 1920
    max_video_height = 1080
    if width > max_video_width or height > max_video_height:
        print("警告: 拼接后的视频尺寸过大，可能无法编码。请考虑减少基线数量或调整图片尺寸。")

    # 定义视频编码器和视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 写入每一帧
    for img in all_concatenated_images:
        video_writer.write(img)

    video_writer.release()
    print(f"视频已保存到 {output_video_path}")

if __name__ == "__main__":

#     # loc 43 case 1
#     model_folder = "/home/xiangyu/Common/loc_43_case_1_T_cross/models/3DGM"

#     # 用 os.path.join 动态拼接路径
#     train_paths = [
#         os.path.join(model_folder, "train/ours_30000/gt"),
#         os.path.join(model_folder, "train/ours_30000/renders"),
#         os.path.join(model_folder, "train/ours_30000/DINO_feat")
# ]
#     # 每段 log 的图片数量列表
#     train_log_img_counts = [66, 68, 67, 66]  # 根据实际情况修改，每个 log 对应 front 的图片数量
#     output_video_path = os.path.join(model_folder, "train_set_video.mp4")
#     generate_video_from_folders(train_paths, train_log_img_counts, output_video_path, fps=6)

#     test_paths = [
#         os.path.join(model_folder, "test/ours_30000/gt"),
#         os.path.join(model_folder, "test/ours_30000/renders"),
#         os.path.join(model_folder, "test/ours_30000/DINO_feat")]
#     # 每段 log 的图片数量列表
#     test_log_img_counts = [65]  # 根据实际情况修改，每个 log 对应 front 的图片数量

#     # 输出视频的路径
#     output_video_path = os.path.join(model_folder, "test_set_video.mp4")
    
#     # 生成视频，设置帧率为30fps
#     generate_video_from_folders(test_paths, test_log_img_counts, output_video_path, fps=6)



#     # loc 37 case 1
#     model_folder = "/home/xiangyu/Ultra/MARS_multitraversal/processed_data/loc_37_case_1/models/3DGM"

#     # 用 os.path.join 动态拼接路径
#     train_paths = [
#         os.path.join(model_folder, "train/ours_30000/gt"),
#         os.path.join(model_folder, "train/ours_30000/renders"),
#         os.path.join(model_folder, "train/ours_30000/depth_map")
# ]
#     # 每段 log 的图片数量列表
#     train_log_img_counts = [75, 57, 64, 69]  # 根据实际情况修改，每个 log 对应 front 的图片数量
#     output_video_path = os.path.join(model_folder, "train_set_video.mp4")
#     generate_video_from_folders(train_paths, train_log_img_counts, output_video_path, fps=6)

#     test_paths = [
#         os.path.join(model_folder, "test/ours_30000/gt"),
#         os.path.join(model_folder, "test/ours_30000/renders"),
#         os.path.join(model_folder, "test/ours_30000/depth_map")]
#     # 每段 log 的图片数量列表
#     test_log_img_counts = [66]  # 根据实际情况修改，每个 log 对应 front 的图片数量

#     # 输出视频的路径
#     output_video_path = os.path.join(model_folder, "test_set_video.mp4")
    
#     # 生成视频，设置帧率为30fps
#     generate_video_from_folders(test_paths, test_log_img_counts, output_video_path, fps=6)


#     # loc 41 case 1
#     model_folder = "/home/xiangyu/Ultra/MARS_multitraversal/processed_data/loc_41_case_4_T_cross/models/3DGM"

#     # 用 os.path.join 动态拼接路径
#     train_paths = [
#         os.path.join(model_folder, "train/ours_30000/gt"),
#         os.path.join(model_folder, "train/ours_30000/renders"),
#         os.path.join(model_folder, "train/ours_30000/depth_map")
# ]
#     # 每段 log 的图片数量列表
#     train_log_img_counts = [71,48,71, 76]  # 根据实际情况修改，每个 log 对应 front 的图片数量
#     output_video_path = os.path.join(model_folder, "train_set_video.mp4")
#     generate_video_from_folders(train_paths, train_log_img_counts, output_video_path, fps=6)

#     test_paths = [
#         os.path.join(model_folder, "test/ours_30000/gt"),
#         os.path.join(model_folder, "test/ours_30000/renders"),
#         os.path.join(model_folder, "test/ours_30000/depth_map")]
#     # 每段 log 的图片数量列表
#     test_log_img_counts = [70]  # 根据实际情况修改，每个 log 对应 front 的图片数量

#     # 输出视频的路径
#     output_video_path = os.path.join(model_folder, "test_set_video.mp4")
    
#     # 生成视频，设置帧率为30fps
#     generate_video_from_folders(test_paths, test_log_img_counts, output_video_path, fps=6)


#      # loc 41 case 1
#     model_folder = "/home/xiangyu/Ultra/MARS_multitraversal/processed_data/loc_41_case_4_T_cross/models/vegs_with_diffusion_r_2"

#     # 用 os.path.join 动态拼接路径
#     train_paths = [
#         os.path.join(model_folder, "train/ours_30000/gt"),
#         os.path.join(model_folder, "train/ours_30000/renders"),
#         os.path.join(model_folder, "train/ours_30000/depth_map")
# ]
#     # 每段 log 的图片数量列表
#     train_log_img_counts = [71,48,71, 76]  # 根据实际情况修改，每个 log 对应 front 的图片数量
#     output_video_path = os.path.join(model_folder, "train_set_video.mp4")
#     generate_video_from_folders(train_paths, train_log_img_counts, output_video_path, fps=6)

#     test_paths = [
#         os.path.join(model_folder, "test/ours_30000/gt"),
#         os.path.join(model_folder, "test/ours_30000/renders"),
#         os.path.join(model_folder, "test/ours_30000/depth_map")]
#     # 每段 log 的图片数量列表
#     test_log_img_counts = [70]  # 根据实际情况修改，每个 log 对应 front 的图片数量

#     # 输出视频的路径
#     output_video_path = os.path.join(model_folder, "test_set_video.mp4")
    
#     # 生成视频，设置帧率为30fps
#     generate_video_from_folders(test_paths, test_log_img_counts, output_video_path, fps=6)


#     # loc 41 case 1
#     model_folder = "/home/xiangyu/Ultra/MARS_multiagent/multiagent_data/processed_data/scene_6/combined_incre/models/3DGM"

#     # 用 os.path.join 动态拼接路径
#     train_paths = [
#         os.path.join(model_folder, "train/ours_7000/gt"),
#         os.path.join(model_folder, "train/ours_7000/renders"),
#         os.path.join(model_folder, "train/ours_7000/depth_map")
# ]
#     # 每段 log 的图片数量列表
#     train_log_img_counts = [301]  # 根据实际情况修改，每个 log 对应 front 的图片数量
#     output_video_path = os.path.join(model_folder, "train_set_video.mp4")
#     generate_video_from_folders(train_paths, train_log_img_counts, output_video_path, fps=6)

#     test_paths = [
#         os.path.join(model_folder, "test/ours_7000/gt"),
#         os.path.join(model_folder, "test/ours_7000/renders"),
#         os.path.join(model_folder, "test/ours_7000/depth_map")]
#     # 每段 log 的图片数量列表
#     test_log_img_counts = [301]  # 根据实际情况修改，每个 log 对应 front 的图片数量

#     # 输出视频的路径
#     output_video_path = os.path.join(model_folder, "test_set_video.mp4")
    
#     # 生成视频，设置帧率为30fps
#     generate_video_from_folders(test_paths, test_log_img_counts, output_video_path, fps=6)



#     # nuplan loc 1
#     model_folder = "/home/xiangyu/Common/v_loc1_level2/models/VEGS_with_road_mask_plus_diffusion"
#     # 用 os.path.join 动态拼接路径
#     train_paths = [
#         os.path.join(model_folder, "train/ours_30000/gt"),
#         os.path.join(model_folder, "train/ours_30000/renders"),
#         os.path.join(model_folder, "train/ours_30000/depth_map")
# ]
#     # 每段 log 的图片数量列表
#     train_log_img_counts = [54, 45, 15, 53, 58]  # 根据实际情况修改，每个 log 对应 front 的图片数量
#     output_video_path = os.path.join(model_folder, "train_set_video.mp4")
#     generate_video_from_folders(train_paths, train_log_img_counts, output_video_path, fps=6)


#     test_paths = [
#         os.path.join(model_folder, "test/ours_30000/gt"),
#         os.path.join(model_folder, "test/ours_30000/renders"),
#         os.path.join(model_folder, "test/ours_30000/depth_map")]
#     # 每段 log 的图片数量列表
#     test_log_img_counts = [80]  # 根据实际情况修改，每个 log 对应 front 的图片数量

#     # 输出视频的路径
#     output_video_path = os.path.join(model_folder, "test_set_video.mp4")
    
#     # 生成视频，设置帧率为30fps
#     generate_video_from_folders(test_paths, test_log_img_counts, output_video_path, fps=6)


#     # nuplan loc2
#     model_folder = "/home/xiangyu/Common/v_loc2_level3/models/VEGS_only_with_diffusion"
# #     # 用 os.path.join 动态拼接路径
# #     train_paths = [
# #         os.path.join(model_folder, "train/ours_30000/video/gt"),
# #         os.path.join(model_folder, "train/ours_30000/video/renders"),
# #         os.path.join(model_folder, "train/ours_30000/video/depth_map")
# # ]
# #     # 每段 log 的图片数量列表
# #     train_log_img_counts = [89]  # 根据实际情况修改，每个 log 对应 front 的图片数量
# #     output_video_path = os.path.join(model_folder, "train_set_video.mp4")
# #     generate_video_from_folders(train_paths, train_log_img_counts, output_video_path, fps=6)


#     test_paths = [
#         os.path.join(model_folder, "test/ours_30000/gt"),
#         os.path.join(model_folder, "test/ours_30000/renders"),
#         os.path.join(model_folder, "test/ours_30000/depth_map")]
#     # 每段 log 的图片数量列表
#     test_log_img_counts = [89]  # 根据实际情况修改，每个 log 对应 front 的图片数量

#     # 输出视频的路径
#     output_video_path = os.path.join(model_folder, "test_set_video.mp4")
    
#     # 生成视频，设置帧率为30fps
#     generate_video_from_folders(test_paths, test_log_img_counts, output_video_path, fps=6)


    # # MARS loc 41 case 4
    # model_folder = "/home/xiangyu/Common/EUVS_data/Level_3_MARS/loc_41_case_4_T_cross/models/"

    # # 用 os.path.join 动态拼接路径
    # train_paths = [
    #     os.path.join(model_folder, "3DGS/train/ours_30000/gt"),
    #     os.path.join(model_folder, "3DGS/train/ours_30000/renders"),
    #     os.path.join(model_folder, "3DGM/train/ours_30000/renders"),
    #     os.path.join(model_folder, "GS_pro/train/ours_30000/renders"),
    #     os.path.join(model_folder, "PGSR/train/ours_30000/renders"),
    #     os.path.join(model_folder, "VEGS/train/ours_30000/renders"),
    #     "/home/xiangyu/Common/zip-nerf-output/exp/loc_41_case_4_T_cross/render/train_preds_step_25000",
    # ]
    # baseline_labels = ["Ground truth", "3DGS", "3DGM", "GSPro", "PGSR", "VEGS", "Zip-NeRF"]

    # # 每段 log 的图片数量列表
    # train_log_img_counts = [71, 48, 71, 76]  # 根据实际情况修改，每个 log 对应 front 的图片数量
    # output_video_path = os.path.join(model_folder, "train_set_video.mp4")
    # generate_video_from_folders(train_paths, train_log_img_counts, output_video_path, baseline_labels=baseline_labels, fps=10)

    # test_paths = [
    #     os.path.join(model_folder, "3DGS/test/ours_30000/gt"),
    #     os.path.join(model_folder, "3DGS/test/ours_30000/renders"),
    #     os.path.join(model_folder, "3DGM/test/ours_30000/renders"),
    #     os.path.join(model_folder, "GS_pro/test/ours_30000/renders"),
    #     os.path.join(model_folder, "PGSR/test/ours_30000/renders"),
    #     os.path.join(model_folder, "VEGS/test/ours_30000/renders"),
    #     "/home/xiangyu/Common/zip-nerf-output/exp/loc_41_case_4_T_cross/render/test_preds_step_25000",
    #     ]
    # # 每段 log 的图片数量列表
    # test_log_img_counts = [70]  # 根据实际情况修改，每个 log 对应 front 的图片数量

    # # 输出视频的路径
    # output_video_path = os.path.join(model_folder, "test_set_video.mp4")
    
    # # 生成视频，设置帧率为30fps
    # generate_video_from_folders(test_paths, test_log_img_counts, output_video_path, baseline_labels=baseline_labels, fps=10)


    # MARS loc 11
    model_folder = "/home/xiangyu/Common/EUVS_data/Level_3_MARS/loc_24_case_1_T_cross/models/"

    # 用 os.path.join 动态拼接路径
    train_paths = [
        os.path.join(model_folder, "3DGS/train/ours_30000/gt"),
        os.path.join(model_folder, "3DGS/train/ours_30000/renders"),
        os.path.join(model_folder, "3DGM/train/ours_30000/renders"),
        os.path.join(model_folder, "GS_pro/train/ours_30000/renders"),
        os.path.join(model_folder, "PGSR/train/ours_30000/renders"),
        os.path.join(model_folder, "VEGS/train/ours_30000/renders"),
        "/home/xiangyu/Common/zip-nerf-output/exp/loc_24_case_1_T_cross/render/train_preds_step_25000",
    ]
    baseline_labels = ["Ground truth", "3DGS", "3DGM", "GSPro", "PGSR", "VEGS", "Zip-NeRF"]

    # 从第一个基线（Ground truth）的 gt 文件夹中获取图片文件名
    gt_folder = train_paths[0]
    train_log_img_counts = calculate_log_img_counts_from_folder(gt_folder)
    print(f"训练集的 log_img_counts: {train_log_img_counts}")

    # 每段 log 的图片数量列表
    # train_log_img_counts = [42, 50, 41, 49, 44, 40]  # 根据实际情况修改，每个 log 对应 front 的图片数量
    output_video_path = os.path.join(model_folder, "train_set_video.mp4")
    generate_video_from_folders(train_paths, train_log_img_counts, output_video_path, baseline_labels=baseline_labels, fps=8)

    test_paths = [
        os.path.join(model_folder, "3DGS/test/ours_30000/gt"),
        os.path.join(model_folder, "3DGS/test/ours_30000/renders"),
        os.path.join(model_folder, "3DGM/test/ours_30000/renders"),
        os.path.join(model_folder, "GS_pro/test/ours_30000/renders"),
        os.path.join(model_folder, "PGSR/test/ours_30000/renders"),
        os.path.join(model_folder, "VEGS/test/ours_30000/renders"),
        "/home/xiangyu/Common/zip-nerf-output/exp/loc_24_case_1_T_cross/render/test_preds_step_25000",
        ]
    # # 每段 log 的图片数量列表
    # test_log_img_counts = [55]  # 根据实际情况修改，每个 log 对应 front 的图片数量
    # 从测试集的 Ground truth gt 文件夹中计算 log_img_counts
    gt_folder_test = test_paths[0]
    test_log_img_counts = calculate_log_img_counts_from_folder(gt_folder_test)
    print(f"测试集的 log_img_counts: {test_log_img_counts}")

    # 输出视频的路径
    output_video_path = os.path.join(model_folder, "test_set_video.mp4")
    
    # 生成视频，设置帧率为30fps
    generate_video_from_folders(test_paths, test_log_img_counts, output_video_path, baseline_labels=baseline_labels, fps=8)
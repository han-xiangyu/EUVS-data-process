import numpy as np
import sys

# 尝试导入 pycolmap，如果失败则无法执行实际检查
try:
    from pycolmap import Quaternion
    pycolmap_available = True
    print("pycolmap imported successfully.")
except ImportError:
    pycolmap_available = False
    print("WARNING: pycolmap could not be imported. Will only perform basic checks.")
    # 定义一个假的 Quaternion 类，使其在调用时失败，以便代码结构能运行
    class Quaternion:
        def __init__(self, arr):
            # 模拟长度检查，但如果 pycolmap 不可用，我们无法真正测试
            if arr.ndim != 1 or arr.shape[0] not in (3, 4):
                 raise ValueError(f"Input array length is {arr.shape[0]}, not 3 or 4.")
            print(" (Fake Quaternion created - pycolmap not available) ")


images_txt_path = '/home/neptune/Data/MARS/city_gs_data/loc07/sparse/0/images.txt' # 确认这是正确路径
line_num = 0
is_camera_description_line = False
error_found = False

print(f"Checking file: {images_txt_path}")

try:
    with open(images_txt_path, 'r') as f:
        for line in f:
            line_num += 1
            line = line.strip()

            if not line or line.startswith('#'):
                continue

            is_camera_description_line = not is_camera_description_line
            data = line.split()

            if is_camera_description_line:
                # print(f"Processing Pose Line {line_num}: {line[:80]}...") # 可以取消注释这行来跟踪
                quat_data = None # 初始化以备错误处理
                quat_array = None
                try:
                    # 1. 检查字段数是否足够 (至少需要 ID + 4 Quat + 3 Pos + CamID + Name = 10)
                    if len(data) < 9: # 稍微放宽一点，至少保证能取到四元数和位置
                         raise ValueError(f"Line has only {len(data)} elements, expected at least 9 for pose.")

                    # 2. 提取四元数部分的字符串
                    quat_data = data[1:5]
                    if len(quat_data) != 4:
                        # 这一步理论上不应发生，因为上面检查了 len(data)
                        raise ValueError(f"Data slice data[1:5] resulted in {len(quat_data)} elements, expected 4.")

                    # 3. 尝试转换成浮点数数组
                    quat_array = np.array(list(map(float, quat_data)))

                    # 4. *** 关键步骤：尝试创建 Quaternion 对象 ***
                    q = Quaternion(quat_array)
                    # 如果成功，可以取消注释下面这行确认
                    # print(f"  Line {line_num}: Quaternion created successfully.")

                except Exception as e:
                    print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print(f"ERROR FOUND at line {line_num}:")
                    print(f"Line content: {line}")
                    print(f"Split data: {data}")
                    print(f"Data slice data[1:5]: {quat_data if quat_data is not None else 'Slice failed or not reached'}")
                    print(f"Converted array: {quat_array if quat_array is not None else 'Conversion failed or not reached'}")
                    print(f"Error during conversion or Quaternion creation: {e}")
                    print(f"Error Type: {type(e)}")
                    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                    error_found = True
                    # 可以在找到第一个错误后停止
                    break
            # else: # 处理 2D 点行，暂时忽略
            #     pass

    if not error_found:
        print(f"\nFinished checking {line_num} lines. No errors detected by this script.")
    else:
        print(f"\nFinished checking after finding error at line {line_num}.")


except FileNotFoundError:
    print(f"Error: File not found at {images_txt_path}")
except Exception as e:
    print(f"An unexpected error occurred outside the main loop: {e}")
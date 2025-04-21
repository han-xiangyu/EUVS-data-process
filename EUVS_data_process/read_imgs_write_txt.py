
import os
from pathlib import Path

def write_file_names_to_txt(directory, n):
    """
    读取指定目录内的文件名,并将前n个文件名逐行写入到指定的输出文件中。

    :param directory: 文件夹路径
    :param n: 要写入的文件名个数
    :param output_file: 输出文件的名称，默认为 "test_set.txt"
    """
    output_txt_file = Path(directory) / "test_set.txt"
    img_dir = Path(directory) / 'images'
    try:
        # 获取文件夹中的所有文件名
        file_names = os.listdir(img_dir)

        # 仅保留文件（排除子文件夹）
        file_names = [f for f in file_names if os.path.isfile(os.path.join(img_dir, f))]
        file_names = sorted(file_names)
        # 获取前n个文件名
        file_names = file_names[:n]

        # 将文件名写入到txt文件中
        with open(output_txt_file, "w") as f:
            for file_name in file_names:
                f.write(file_name + "\n")
        
        print(f"成功将前 {n} 个文件名写入到 {output_txt_file} 中！")
    except Exception as e:
        print(f"发生错误: {e}")

# 示例调用

write_file_names_to_txt("/home/xiangyu/Ultra/Download/ithaca_loc_275", 200)
write_file_names_to_txt("/home/xiangyu/Ultra/Download/ithaca_loc_1200", 200)
write_file_names_to_txt("/home/xiangyu/Ultra/Download/ithaca_loc_1300", 200)
write_file_names_to_txt("/home/xiangyu/Ultra/Download/ithaca_loc_2300", 200)
write_file_names_to_txt("/home/xiangyu/Ultra/Download/ithaca_loc_2450", 200)
write_file_names_to_txt("/home/xiangyu/Ultra/Download/ithaca_loc_2500", 200)
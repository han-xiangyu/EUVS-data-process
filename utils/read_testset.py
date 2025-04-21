# import os

# def write_test_set(image_folder, output_file):

#     # 打开文件准备写入
#     with open(output_file, 'w') as f:
#         # 遍历文件夹中的所有文件
#         for filename in os.listdir(image_folder):
#             # 检查文件名是否符合格式并以.jpg结尾
#             if filename.endswith('.jpg'):
#                 # 分割文件名，根据下划线提取各个部分
#                 parts = filename.split('_')
                
#                 # 检查parts是否符合格式，并确保X为1
#                 if len(parts) >= 3 and parts[1] == '0001':
#                     # 将符合条件的文件名写入输出文件
#                     f.write(filename + '\n')

#     print("文件名已成功写入到 test_set.txt 中。")




# # 使用示例
# image_folder = '/home/xiangyu/Common/nuplan/loc8_level1_single_traversal/images'  # 替换为你的图片文件夹路径
# output_file = '/home/xiangyu/Common/nuplan/loc8_level1_single_traversal/test_set.txt'  # 输出文件名
# write_test_set(image_folder, output_file)



import os

# 定义图片文件夹路径
image_folder = '/home/xiangyu/Common/nuplan/nuplan_loc33/tra6/images'

# 定义输出文件路径
output_file = '/home/xiangyu/Common/nuplan/nuplan_loc33/tra6/test_set.txt'

# 打开输出文件准备写入
with open(output_file, 'w') as f:
    # 遍历文件夹中的所有文件
    for filename in os.listdir(image_folder):
        # 检查文件是否为图片（可以根据扩展名过滤）
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
            # 将图片文件名写入输出文件
            f.write(filename + '\n')

print("所有图片文件名已成功写入到 all_images.txt 中。")

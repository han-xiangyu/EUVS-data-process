# cp /mnt/NAS/data/zj2640/MARS/organized/loc24/orientation/incre/123_1cam/database.db \
#     /mnt/NAS/data/zj2640/MARS/organized/loc24/orientation/incre/1234_1cam \

# colmap feature_extractor \
#     --database_path /mnt/NAS/data/zj2640/MARS/organized/loc24/orientation/incre/1234_1cam/database.db \
# #     --image_path /mnt/NAS/data/zj2640/MARS/organized/loc24/orientation/incre/1234_1cam/images \
# #     --ImageReader.camera_model PINHOLE \
# #     --ImageReader.mask_path /mnt/NAS/data/zj2640/MARS/organized/loc24/orientation/incre/1234_1cam/masks \
# #     --image_list_path /mnt/NAS/data/zj2640/MARS/organized/loc24/orientation/incre/1234_1cam/image.txt \
# #     --SiftExtraction.use_gpu 1 \
# #     --ImageReader.single_camera 1 \

# # colmap vocab_tree_matcher \
# #     --database_path /mnt/NAS/data/zj2640/MARS/organized/loc24/orientation/incre/1234_1cam/database.db \
# #     --VocabTreeMatching.vocab_tree_path /mnt/NAS/home/zj2640/bin/vocab_tree_flickr100K_words256K.bin \
# #     --VocabTreeMatching.match_list_path /mnt/NAS/data/zj2640/MARS/organized/loc24/orientation/incre/1234_1cam/image.txt

# # colmap image_registrator \
# #     --database_path /mnt/NAS/data/zj2640/MARS/organized/loc22/orientation/incre/12_1cam/database.db \
# #     --input_path /mnt/NAS/data/zj2640/MARS/organized/loc22/orientation/1_1cam/sparse/0 \
# #     --output_path /mnt/NAS/data/zj2640/MARS/organized/loc22/orientation/incre/12_1cam/sparse/0

# # mkdir -p /mnt/NAS/data/zj2640/MARS/organized/loc24/orientation/incre/1234_1cam/sparse/0

# colmap mapper \
#     --database_path /mnt/NAS/data/zj2640/MARS/organized/loc24/orientation/incre/1234_1cam/database.db \
#     --image_path /mnt/NAS/data/zj2640/MARS/organized/loc24/orientation/incre/1234_1cam/images \
#     --input_path /mnt/NAS/data/zj2640/MARS/organized/loc24/orientation/incre/123_1cam/sparse/0 \
#     --output_path /mnt/NAS/data/zj2640/MARS/organized/loc24/orientation/incre/1234_1cam/sparse/0

#!/bin/bash

# 检查是否传入了两个参数
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_path> <destination_path>"
    exit 1
fi

# 定义传入的路径
SOURCE_PATH=$1
DEST_PATH=$2


# 拷贝数据库文件到指定的目标路径
cp "${SOURCE_PATH}/distorted/database.db" "${DEST_PATH}"

# 执行 COLMAP 特征提取
colmap feature_extractor \
    --database_path "${DEST_PATH}/database.db" \
    --image_path "${DEST_PATH}/input" \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.mask_path "${DEST_PATH}/masks" \
    --image_list_path "${DEST_PATH}/test_set.txt" \
    --SiftExtraction.use_gpu 1 \
    --ImageReader.single_camera 1

# 执行 COLMAP 词汇树匹配
colmap vocab_tree_matcher \
    --database_path "${DEST_PATH}/database.db" \
    --VocabTreeMatching.vocab_tree_path ./vocab_tree_flickr100K_words256K.bin \
    --VocabTreeMatching.match_list_path "${DEST_PATH}/test_set.txt"

# 创建稀疏模型输出目录
mkdir -p "${DEST_PATH}/sparse/0"

# 执行 COLMAP 3D 重建 (Mapper)
colmap mapper \
    --database_path "${DEST_PATH}/database.db" \
    --image_path "${DEST_PATH}/input" \
    --input_path "${SOURCE_PATH}/sparse/0" \
    --output_path "${DEST_PATH}/sparse/0"

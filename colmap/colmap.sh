#!/bin/bash

# # 主文件夹路径
# main_folder=$1  

# # 遍历主文件夹中的每个子文件夹
# for path in "$main_folder"/*; do
#     if [ -d "$path" ]; then  # 检查是否为子文件夹
#         echo "Processing $path..."

#         # Feature extraction
#         colmap feature_extractor \
#             --database_path ${path}/database.db \
#             --image_path ${path}/images \
#             --ImageReader.camera_model PINHOLE \
#             --ImageReader.mask_path ${path}/masks \
#             --SiftExtraction.use_gpu 1 \
#             --ImageReader.single_camera 1 \

#         # Exhaustive matching
#         colmap exhaustive_matcher \
#             --database_path ${path}/database.db \
#             --SiftMatching.use_gpu 1 \

#         # ##vocat tree matching
#         # # colmap vocab_tree_matcher \
#         # #     --database_path ${path}/database.db \
#         # #     --SiftMatching.use_gpu 1 \
#         # #     --VocabTreeMatching.vocab_tree_path /mnt/HDD1/zj2640/bin/vocab_tree_flickr100K_words256K.bin \

#         # 创建 sparse 文件夹
#         mkdir -p ${path}/sparse

#         # Mapping       
#         colmap mapper \
#             --database_path ${path}/database.db \
#             --image_path ${path}/images \
#             --output_path ${path}/sparse

#         # 转换 sparse 中的每个子目录为文本格式
#         for subdir in ${path}/sparse/*; do
#             if [ -d "$subdir" ]; then
#                 echo "Converting model in $subdir to text format..."
#                 colmap model_converter \
#                     --input_path $subdir \
#                     --output_path $subdir \
#                     --output_type TXT
#             fi
#         done

#         echo "Finished processing $path"
#     fi
# done

path=$1  

# conda activate segformer

# python tools/SegFormer/extract.py ${path}

colmap feature_extractor \
    --database_path ${path}/database.db \
    --image_path ${path}/input \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.mask_path ${path}/masks \
    --SiftExtraction.use_gpu 1 \
    --ImageReader.single_camera 1 \
    


# Exhaustive matching
colmap exhaustive_matcher \
    --database_path ${path}/database.db \
    --SiftMatching.use_gpu 1 \

# ##vocat tree matching
# # colmap vocab_tree_matcher \
# #     --database_path ${path}/database.db \
# #     --SiftMatching.use_gpu 1 \
# #     --VocabTreeMatching.vocab_tree_path /mnt/HDD1/zj2640/bin/vocab_tree_flickr100K_words256K.bin \

mkdir ${path}/masked_sparse

# Mapping       
colmap mapper \
    --database_path ${path}/database.db \
    --image_path ${path}/input \
    --output_path ${path}/masked_sparse

# TXT
for subdir in ${path}/masked_sparse/*; do
    if [ -d "$subdir" ]; then
        echo "Converting model in $subdir to text format..."
        colmap model_converter \
            --input_path $subdir \
            --output_path $subdir \
            --output_type TXT
    fi
done

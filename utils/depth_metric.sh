#!/bin/bash

# 定义不同的dataroot路径
dataroots=(
    # "/home/xiangyu/Common/loc_06_case_1"
    # "/home/xiangyu/Common/loc_07_case_1_T_cross"
    # "/home/xiangyu/Common/loc_11_case_1"
    # "/home/xiangyu/Common/loc_15_case_1"
    # "/home/xiangyu/Common/loc_15_case_2_T_cross"
    # "/home/xiangyu/Common/loc_24_case_1_T_cross"
    # "/home/xiangyu/Common/loc_37_case_3_T_cross"
    # "/home/xiangyu/Common/loc_41_case_1_two_turns"
    # "/home/xiangyu/Common/loc_41_case_2_straight_and_right_turn"
    # "/home/xiangyu/Common/loc_41_case_3_straight_and_left_turn"
    # "/home/xiangyu/Common/loc_41_case_5_T_cross"
    /home/xiangyu/Common/loc_43_case_1_T_cross
    # "/home/xiangyu/Common/loc_62_case_1_middle_lane_change"
    # "/home/xiangyu/Common/loc_62_case_2_right_lane_change"
    # "/home/xiangyu/Common/loc_62_case_3_cross"
    # /home/xiangyu/Common/nuplan/v_loc10_level3
    # /home/xiangyu/Common/nuplan/v_loc13_level3
    # /home/xiangyu/Common/nuplan/v_loc14_level3
    # /home/xiangyu/Common/nuplan/v_loc15_level3
    # /home/xiangyu/Common/nuplan/v_loc18_level3
    # /home/xiangyu/Common/nuplan/v_loc19_level3
    # /home/xiangyu/Common/nuplan/v_loc20_level3
    # /home/xiangyu/Common/nuplan/v_loc21_level3
    # /home/xiangyu/Common/nuplan/v_loc23_level3
    # /home/xiangyu/Common/nuplan/v_loc24_level3

)


# 遍历每个dataroot并执行命令
for dataroot in "${dataroots[@]}"; do
    model_path="$dataroot/models/3DGS"
    echo "Processing data from: $model_path"
    python data_process/metric_depth.py \
        -s "$dataroot" \
        -m "$model_path"

    # model_path="$dataroot/models/3DGM"
    # echo "Processing data from: $model_path"
    # python data_process/metric_depth.py \
    #     -s "$dataroot" \
    #     -m "$model_path"


    model_path="$dataroot/models/PGSR"
    echo "Processing data from: $model_path"
    python data_process/metric_depth.py \
        -s "$dataroot" \
        -m "$model_path"


    model_path="$dataroot/models/2DGS"
    echo "Processing data from: $model_path"
    python data_process/metric_depth.py \
        -s "$dataroot" \
        -m "$model_path"


    model_path="$dataroot/models/GSPro"
    echo "Processing data from: $model_path"
    python data_process/metric_depth.py \
        -s "$dataroot" \
        -m "$model_path"


    model_path="$dataroot/models/VEGS"
    echo "Processing data from: $model_path"
    python data_process/metric_depth.py \
        -s "$dataroot" \
        -m "$model_path"


done

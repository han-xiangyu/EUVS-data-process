import os
import json
import numpy as np
import pandas as pd

# 设置JSON文件夹信息的列表
def get_json_dirs(model_name):
    base_dirs = [
        "/home/xiangyu/Common/loc_41_case_4_T_cross/models/",
        "/home/xiangyu/Common/loc_43_case_1_T_cross/models/",
        "/home/xiangyu/Common/loc_06_case_1/models/",
        "/home/xiangyu/Common/loc_07_case_1_T_cross/models/",
        "/home/xiangyu/Common/loc_11_case_1/models/",
        "/home/xiangyu/Common/loc_15_case_1/models/",
        "/home/xiangyu/Common/loc_15_case_2_T_cross/models/",
        "/home/xiangyu/Common/loc_24_case_1_T_cross/models/",
        "/home/xiangyu/Common/loc_37_case_1/models/",
        "/home/xiangyu/Common/loc_37_case_3_T_cross/models/",
        "/home/xiangyu/Common/loc_41_case_1_two_turns/models/",
        "/home/xiangyu/Common/loc_41_case_2_straight_and_right_turn/models/",
        "/home/xiangyu/Common/loc_41_case_3_straight_and_left_turn/models/",
        "/home/xiangyu/Common/loc_41_case_5_T_cross/models/",
        "/home/xiangyu/Common/loc_62_case_1_middle_lane_change/models/",
        "/home/xiangyu/Common/loc_62_case_2_right_lane_change/models/",
        "/home/xiangyu/Common/loc_62_case_3_cross/models/"
    ]
    return [os.path.join(base_dir, model_name) for base_dir in base_dirs]

def load_metrics_from_json_dirs(json_dirs, file_name):
    metrics = {
        "SSIM": [],
        "PSNR": [],
        "LPIPS": [],
        "RMSE": [],
        "Feat_PSNR": [],
        "Cos_Similarity": []
    }
    # 遍历每个文件夹中的指定JSON文件
    for json_dir in json_dirs:
        file_path = os.path.join(json_dir, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                # 读取指标值并存储到对应列表中
                for key in data.values():
                    for metric, value in key.items():
                        if metric in metrics:
                            metrics[metric].append(value)
    return metrics

def calculate_average_metrics(metrics):
    avg_metrics = {}
    for metric, values in metrics.items():
        if values:
            mean_value = np.mean(values)
            if mean_value == 0:
                avg_metrics[metric] = 0.0
            else:
                avg_metrics[metric] = float(f"{mean_value:.4g}")
        else:
            avg_metrics[metric] = None
    return avg_metrics

def main():
    model_names = ["3DGS", "3DGM", "GS_pro", "VEGS", "PGSR", "2DGS", "OmniRe"]

    results = []

    for model_name in model_names:
        json_dirs = get_json_dirs(model_name)

        # 加载训练集和测试集的指标
        train_metrics = load_metrics_from_json_dirs(json_dirs, "train_set_results_mask.json")
        test_metrics = load_metrics_from_json_dirs(json_dirs, "test_set_results_mask.json")

        # 计算训练集和测试集的平均值
        avg_train_metrics = calculate_average_metrics(train_metrics)
        avg_test_metrics = calculate_average_metrics(test_metrics)

        # 收集结果
        results.append([
            model_name,
            avg_train_metrics.get("PSNR"), avg_test_metrics.get("PSNR"),
            avg_train_metrics.get("SSIM"), avg_test_metrics.get("SSIM"),
            avg_train_metrics.get("LPIPS"), avg_test_metrics.get("LPIPS"),
            avg_train_metrics.get("RMSE"), avg_test_metrics.get("RMSE"),
            "-", "-",  # Delta125 is not calculated, so placeholders are used
            avg_train_metrics.get("Feat_PSNR"), avg_test_metrics.get("Feat_PSNR"),
            avg_train_metrics.get("Cos_Similarity"), avg_test_metrics.get("Cos_Similarity")
        ])

    # 创建DataFrame并打印结果
    columns = [
        "Baseline", "PSNR (train)", "PSNR (test)", "SSIM (train)", "SSIM (test)",
        "LPIPS (train)", "LPIPS (test)", "RMSE (train)", "RMSE (test)",
        "Delta125 (train)", "Delta125 (test)", "Feat PSNR (train)", "Feat PSNR (test)",
        "Feat Cos Sim (train)", "Feat Cos Sim (test)"
    ]
    df = pd.DataFrame(results, columns=columns)
    pd.set_option('display.max_columns', None)  # 显示所有列
    pd.set_option('display.width', 200)  # 设置输出宽度
    print(df)

    # 生成LaTeX表格格式
    latex_table = "\\begin{table*}[h!]\n    \centering\n    \\begin{tabular}{lcccccccccccccc}\n        \\toprule\n        \\textbf{Baseline} & \\multicolumn{2}{c}{\\textbf{PSNR}} & \\multicolumn{2}{c}{\\textbf{SSIM}} & \\multicolumn{2}{c}{\\textbf{LPIPS}} & \\multicolumn{2}{c}{\\textbf{RMSE}} & \\multicolumn{2}{c}{\\textbf{Delta125}} & \\multicolumn{2}{c}{\\textbf{Feat PSNR}} & \\multicolumn{2}{c}{\\textbf{Feat Cos Sim}}\\\\\n        \\midrule\n        {} & train & test & train & test & train & test & train & test & train & test & train & test & train & test \\\\n        \\midrule\n"
    for row in results:
        latex_table += f"        {row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} & {row[5]} & {row[6]} & {row[7]} & {row[8]} & {row[9]} & {row[10]} & {row[11]} & {row[12]} & {row[13]} & {row[14]} \\\\n"
    latex_table += "        \\bottomrule\n    \\end{tabular}\n    \\caption{Quantitative results comparison for level 3, where $\\dagger$ denotes training set, loss * denotes the test set loss and Depth denotes Depth RMSE.}\n    \\label{table:level3_results}\n\\end{table*}"
    print("\n\nGenerated LaTeX Table:\n")
    print(latex_table)

if __name__ == "__main__":
    main()

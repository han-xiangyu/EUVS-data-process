# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import os
# import re
# import argparse
# from PIL import Image
# import numpy as np
# import cv2

# def make_video(root_dir, output_video, fps=30):
#     gt_dir     = os.path.join(root_dir, "gt")
#     render_dir = os.path.join(root_dir, "renders")
#     if not os.path.isdir(gt_dir) or not os.path.isdir(render_dir):
#         raise FileNotFoundError(f"请确保 `{root_dir}` 下存在 `gt/` 和 `render/` 子文件夹")

#     # 匹配 trav_{traversal}_channel_{channel}_img_{frame}.jpg
#     pattern = re.compile(r"^trav_(\d+)_channel_(\d+)_img_(\d+)\.jpg$")
#     channels_order = [2, 1, 3]  # 左(2)、中(1)、右(3)

#     # 收集所有文件信息：{ trav_id: { frame_int: frame_str, ... }, ... }
#     files = {}
#     for sub, folder in [("gt", gt_dir), ("render", render_dir)]:
#         for fn in os.listdir(folder):
#             m = pattern.match(fn)
#             if not m:
#                 continue
#             trav_id   = m.group(1)
#             frame_str = m.group(3)
#             frame_int = int(frame_str)
#             files.setdefault(trav_id, {}).setdefault(frame_int, {})["frame_str"] = frame_str

#     if not files:
#         raise RuntimeError("在 gt/render 文件夹中未发现符合命名规则的图片")

#     # 如果有多个 trav_id，就对每个都生成一个视频；只有一个就按 output_video 名称
#     trav_ids = sorted(files.keys(), key=int)
#     for trav_id in trav_ids:
#         frame_ints = sorted(files[trav_id].keys())
#         # 取第一帧、第一通道的样图来获取尺寸
#         sample_frame = frame_ints[0]
#         sample_str   = files[trav_id][sample_frame]["frame_str"]
#         sample_name  = f"trav_{trav_id}_channel_{channels_order[0]}_img_{sample_str}.jpg"
#         # 优先从 gt 里读，没有再去 render
#         for folder in (gt_dir, render_dir):
#             sample_path = os.path.join(folder, sample_name)
#             if os.path.isfile(sample_path):
#                 w, h = Image.open(sample_path).size
#                 break
#         else:
#             raise RuntimeError(f"无法获取样图尺寸（没找到 {sample_name}）")

#         # 决定输出文件名
#         if len(trav_ids) == 1:
#             out_path = output_video
#         else:
#             base, ext = os.path.splitext(output_video)
#             out_path   = f"{base}_trav_{trav_id}{ext}"

#         fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#         video  = cv2.VideoWriter(out_path, fourcc, fps, (3 * w, 2 * h))

#         for frame_int in frame_ints:
#             frame_str = files[trav_id][frame_int]["frame_str"]
#             canvas = Image.new("RGB", (3 * w, 2 * h))

#             for row, sub in enumerate(["gt", "render"]):
#                 folder = gt_dir if sub == "gt" else render_dir
#                 for col, ch in enumerate(channels_order):
#                     fn   = f"trav_{trav_id}_channel_{ch}_img_{frame_str}.jpg"
#                     path = os.path.join(folder, fn)
#                     if os.path.isfile(path):
#                         img = Image.open(path)
#                     else:
#                         print(f"警告：{path} 不存在，已用黑图填充")
#                         img = Image.new("RGB", (w, h), (0, 0, 0))
#                     canvas.paste(img, (col * w, row * h))

#             frame_bgr = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
#             video.write(frame_bgr)

#         video.release()
#         print(f"已生成：{out_path}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="将 gt/render 两行三列拼接图输出为视频，只根据实际存在的文件编号。"
#     )
#     parser.add_argument("root_dir",
#                         help="包含 gt/ 和 render/ 子文件夹的根目录")
#     parser.add_argument("output",
#                         help="输出视频文件路径，例如 out.mp4")
#     parser.add_argument("--fps", type=int, default=30,
#                         help="视频帧率，默认 30")
#     args = parser.parse_args()

#     make_video(args.root_dir, args.output, fps=args.fps)




#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
from PIL import Image
import numpy as np
import cv2

def make_video(root_dir, output_video, fps=30):
    """
    从 <root_dir>/gt 和 <root_dir>/render 读取 trav_*_channel_*_img_*.jpg，
    不要求帧号连续或对齐；缺帧时自动用同 channel 下一帧补齐（若已是最后帧，则用最后一帧）。
    按每帧生成两行三列图：第一行 gt，第二行 render，列序：左(2)、中(1)、右(3)。
    最终输出 mp4 视频。
    """
    gt_dir     = os.path.join(root_dir, "gt")
    render_dir = os.path.join(root_dir, "renders")
    if not os.path.isdir(gt_dir) or not os.path.isdir(render_dir):
        raise FileNotFoundError(f"`{root_dir}` 下需包含 `gt/` 和 `render/` 子文件夹")

    pattern = re.compile(r"^trav_(\d+)_channel_(\d+)_img_(\d+)\.jpg$")
    channels_order = [2, 1, 3]  # 左(2)、中(1)、右(3)

    # 1) 扫描所有文件，构建可用帧映射： avail[trav][sub][channel][frame_int] = frame_str
    avail = {}
    for sub, folder in [("gt", gt_dir), ("renders", render_dir)]:
        for fn in os.listdir(folder):
            m = pattern.match(fn)
            if not m:
                continue
            trav_id   = m.group(1)
            ch        = int(m.group(2))
            frame_int = int(m.group(3))
            frame_str = m.group(3)
            avail.setdefault(trav_id, {}) \
                 .setdefault(sub, {}) \
                 .setdefault(ch, {})[frame_int] = frame_str

    if not avail:
        raise RuntimeError("未在 gt/ 或 render/ 中找到符合 `trav_*_channel_*_img_*.jpg` 的图片")

    # 2) 对每个 trav_id 分别生成视频
    for trav_id, subdict in sorted(avail.items(), key=lambda x: int(x[0])):
        # collect superset of all frame_ints across both subs & channels
        frame_set = set()
        for sub in ("gt", "renders"):
            for ch_map in subdict.get(sub, {}).values():
                frame_set.update(ch_map.keys())
        frame_list = sorted(frame_set)
        if not frame_list:
            continue

        # 3) 确定单张图尺寸：任选第一个存在的文件
        sample_w, sample_h = None, None
        for sub in ("gt", "renders"):
            for ch in channels_order:
                ch_map = subdict.get(sub, {}).get(ch, {})
                if not ch_map:
                    continue
                # pick smallest frame_int in this map
                f0 = min(ch_map.keys())
                fn = f"trav_{trav_id}_channel_{ch}_img_{ch_map[f0]}.jpg"
                p  = os.path.join(gt_dir if sub=="gt" else render_dir, fn)
                if os.path.isfile(p):
                    sample_w, sample_h = Image.open(p).size
                    break
            if sample_w:
                break
        if not sample_w:
            raise RuntimeError(f"无法获取 trav_{trav_id} 的样图尺寸")

        # 4) 准备视频写入
        base, ext = os.path.splitext(output_video)
        out_path = output_video if len(avail)==1 else f"{base}_trav_{trav_id}{ext}"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video  = cv2.VideoWriter(out_path, fourcc, fps, (3*sample_w, 2*sample_h))

        # 5) 按 superset 帧号迭代，缺帧补齐
        for frm in frame_list:
            canvas = Image.new("RGB", (3*sample_w, 2*sample_h))
            for row, sub in enumerate(["gt", "renders"]):
                folder = gt_dir if sub=="gt" else render_dir
                for col, ch in enumerate(channels_order):
                    ch_map = subdict.get(sub, {}).get(ch, {})
                    if ch_map:
                        # 找到第一个 >= frm 的帧，否则取最后一帧
                        candidates = [f for f in ch_map if f >= frm]
                        pick = min(candidates) if candidates else max(ch_map)
                        frame_str = ch_map[pick]
                        fn = f"trav_{trav_id}_channel_{ch}_img_{frame_str}.jpg"
                        p  = os.path.join(folder, fn)
                        if os.path.isfile(p):
                            img = Image.open(p)
                        else:
                            print(f"警告：{p} 不存在，用黑图替代")
                            img = Image.new("RGB", (sample_w, sample_h), (0,0,0))
                    else:
                        # 该 sub/ch 完全无图
                        img = Image.new("RGB", (sample_w, sample_h), (0,0,0))
                    canvas.paste(img, (col*sample_w, row*sample_h))

            frame_bgr = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
            video.write(frame_bgr)

        video.release()
        print(f"生成完毕：{out_path}")

if __name__ == "__main__":

    root_dir = "/home/neptune/Data/MARS/city_gs_data/loc6_10_17_dense_voxel_050/output_grendalGS_no_densify_no_opacity_reset/train/ours_80000"
    output_video = "/home/neptune/Data/MARS/city_gs_data/loc6_10_17_dense_voxel_050/output_grendalGS_no_densify_no_opacity_reset/train_set_video.mp4"
    fps = 15
    make_video(root_dir, output_video, fps)


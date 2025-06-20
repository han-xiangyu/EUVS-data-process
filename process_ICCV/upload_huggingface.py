from huggingface_hub import upload_file
import os

# 1️⃣ Hugging Face 数据集仓库 ID
repo_id = "Valentina01277/EUVS"  # 👈 修改为你的用户名和数据集名称

# 2️⃣ 本地包含 zip 文件的文件夹路径
zip_folder = "/mnt/NAS/home/zj2640/iccv_workshop/small/organized/Level2_small"  # 👈 修改为本地 zip 文件夹

# 3️⃣ 遍历文件夹下所有 zip 文件，上传到仓库的 zips/ 目录
for file_name in os.listdir(zip_folder):
    if file_name.endswith(".zip"):
        file_path = os.path.join(zip_folder, file_name)
        # 远端路径：zips/filename.zip
        path_in_repo = f"Level2_small/{file_name}"

        print(f"🚀 正在上传：{file_name} 到 {path_in_repo}")
        upload_file(
            repo_id=repo_id,
            path_or_fileobj=file_path,
            path_in_repo=path_in_repo,
            repo_type="dataset"
        )
        print(f"✅ 上传完成：{file_name}")

print("🎉 所有 zip 文件已上传到 Hugging Face 仓库的 zips/ 目录！")
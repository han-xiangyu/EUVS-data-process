
import os

def filter_image_files(image_folder, file_list_txt, output_txt):
    """
    讀取一個資料夾中的所有圖片檔案，再讀取一個 txt，
    然後根據 txt 去篩選出不在這個 txt 裡的檔案，
    把這些檔名寫到另一個 txt 檔案中。

    :param image_folder: 存放圖片的資料夾路徑。
    :param file_list_txt: 包含要篩選掉的檔名的 txt 檔案路徑。
    :param output_txt: 儲存結果的 txt 檔案路徑。
    """
    # 1. 定義支援的圖片副檔名
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}

    # 2. 讀取圖片資料夾中所有的圖片檔名
    try:
        all_image_files = {f for f in os.listdir(image_folder)
                           if os.path.isfile(os.path.join(image_folder, f)) and
                           os.path.splitext(f)[1].lower() in image_extensions}
    except FileNotFoundError:
        print(f"錯誤：找不到圖片資料夾 '{image_folder}'")
        return

    # 3. 讀取需要排除的檔名清單
    try:
        with open(file_list_txt, 'r', encoding='utf-8') as f:
            # 使用 set 以加快查找速度，並移除每行結尾的換行符
            files_to_exclude = {line.strip() for line in f}
    except FileNotFoundError:
        print(f"錯誤：找不到檔名清單檔案 '{file_list_txt}'")
        return

    # 4. 篩選出不在排除清單中的檔案
    files_to_keep = sorted(list(all_image_files - files_to_exclude))

    # 5. 將結果寫入到新的 txt 檔案中
    try:
        with open(output_txt, 'w', encoding='utf-8') as f:
            for file_name in files_to_keep:
                f.write(file_name + '\n')
        print(f"成功！已將 {len(files_to_keep)} 個不在清單中的檔名寫入到 '{output_txt}'")
    except IOError as e:
        print(f"錯誤：無法寫入檔案 '{output_txt}'。錯誤訊息：{e}")

# --- 使用範例 ---
if __name__ == '__main__':
    # --- 請修改以下三個變數 ---

    # 1. 你的圖片資料夾路徑
    #    在 Windows 上路徑可能像這樣：r'C:\Users\YourUser\Desktop\images'
    #    在 macOS 或 Linux 上路徑可能像這樣：'/Users/youruser/Desktop/images'
    image_directory = r'f:\loc43\images'

    # 2. 包含要排除的檔名的 TXT 檔案路徑
    #    這個 txt 檔案的每一行都是一個你想要忽略的圖片檔名
    exclude_list_file = r'f:\loc43\test_set.txt'

    # 3. 儲存結果的 TXT 檔案路徑
    #    執行後，多出來的檔名會被存到這個檔案裡
    output_file = r'f:\loc43\train_set.txt'

    # --- 執行函式 ---
    filter_image_files(image_directory, exclude_list_file, output_file)
import os
import shutil

def move_pdfs(source_folder, destination_folder):
    # 遍历源文件夹中的所有文件和子文件夹
    for item in os.listdir(source_folder):
        item_path = os.path.join(source_folder, item)

        # 检查是否为子文件夹
        if os.path.isdir(item_path):
            # 递归调用移动函数
            move_pdfs(item_path, destination_folder)
        elif item.lower().endswith('.pdf'):
            # 构建目标文件的完整路径
            destination_file_path = os.path.join(destination_folder, item)

            # 移动文件
            shutil.move(item_path, destination_file_path)
            print(f"Moved {item} to {destination_folder}")

# 源文件夹路径
source_folder = r'E:\CCT Threshold.Data\PDF'

# 目标文件夹路径
destination_folder = r'E:\CCT Threshold.Data\all-pdf'

# 调用移动函数
move_pdfs(source_folder, destination_folder)

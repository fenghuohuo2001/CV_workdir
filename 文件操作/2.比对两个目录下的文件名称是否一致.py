"""
@Name: 2.比对两个目录下的文件名称是否一致.py
@Auth: Huohuo
@Date: 2023/7/25-16:11
@Desc: 
@Ver : 
code_idea
"""
import os

def find_different_files(dir1, dir2):
    files_dict1 = {os.path.splitext(file)[0]: os.path.join(dir1, file) for file in os.listdir(dir1)}
    files_dict2 = {os.path.splitext(file)[0]: os.path.join(dir2, file) for file in os.listdir(dir2)}

    different_files = []

    for filename, path1 in files_dict1.items():
        path2 = files_dict2.get(filename)
        if path2 and os.path.splitext(path1)[1] != os.path.splitext(path2)[1]:
            different_files.append((filename, path1, path2))

    return different_files

if __name__ == "__main__":
    folder1_path = "D:\WorkDirectory\school_project\leader\datasets_download\ocr_dataset\VOCdevkit\VOC2007\JPEGImages"  # 替换为第一个文件夹的路径
    folder2_path = "D:\WorkDirectory\school_project\leader\datasets_download\ocr_dataset\VOCdevkit\VOC2007\Annotations"  # 替换为第二个文件夹的路径


    different_files = find_different_files(folder1_path, folder2_path)

    if different_files:
        print("Different files:")
        for filename, path1, path2 in different_files:
            print(f"Filename: {filename}, Path1: {path1}, Path2: {path2}")
    else:
        print("No different files found.")



    # folder1_path = "D:\WorkDirectory\school_project\leader\datasets_download\ocr_dataset\VOCdevkit\VOC2007\JPEGImages"  # 替换为第一个文件夹的路径
    # folder2_path = "D:\WorkDirectory\school_project\leader\datasets_download\ocr_dataset\VOCdevkit\VOC2007\Annotations"  # 替换为第二个文件夹的路径

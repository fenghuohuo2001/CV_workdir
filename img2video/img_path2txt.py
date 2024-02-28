"""
@Name: img_path2txt.py
@Auth: Huohuo
@Date: 2023/7/3-17:03
@Desc: 
@Ver : 
code_idea
"""
import os

folder_path = r'D:\datasets\CULane-self\watercar/'  # 文件夹路径
output_file = r'D:\datasets\CULane-self\list\val_gt.txt'  # 输出txt文件路径

# 获取文件夹中的所有图片文件路径
image_paths = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(root, file)
            image_paths.append(image_path)

# # 将图片路径写入txt文件
# with open(output_file, 'w') as file:
#     file.write('\n'.join(image_paths))

# 将图片路径写入txt文件，并添加 "0 1 0 1" 到每个路径后面
with open(output_file, 'w') as file:
    for image_path in image_paths:
        file.write(image_path + ' 0 1 0 1\n')

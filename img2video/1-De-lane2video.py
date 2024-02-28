"""
@Name: 1-De-lane2video.py
@Auth: Huohuo
@Date: 2023/5/10-16:13
@Desc: 
@Ver : 
code_idea
"""
import cv2
import numpy
import os
import time

time_start = time.time()


# dla34_culane
# imgs_path = r"D:\WorkDirectory\mywork\CLRNet-main\CLRNet-main\work_dirs\clr\dla34_culane\20230510_172856_lr_6e-04_b_1\visualization"
# video_save_path = r"D:\WorkDirectory\mywork\CLRNet-main\CLRNet-main\work_dirs\clr\dla34_culane\20230510_172856_lr_6e-04_b_1\result.mp4"

# r34_culane
# imgs_path = r"D:\WorkDirectory\mywork\CLRNet-main\CLRNet-main\work_dirs\clr\r34_culane\20230510_173020_lr_6e-04_b_1\visualization"
# video_save_path = r"D:\WorkDirectory\mywork\CLRNet-main\CLRNet-main\work_dirs\clr\r34_culane\20230510_173020_lr_6e-04_b_1\result.mp4"

# r101_culane
# imgs_path = r"D:\WorkDirectory\mywork\CLRNet-main\CLRNet-main\work_dirs\clr\r101_culane\20230510_173150_lr_3e-04_b_1\visualization"
# video_save_path = r"D:\WorkDirectory\mywork\CLRNet-main\CLRNet-main\work_dirs\clr\r101_culane\20230510_173150_lr_3e-04_b_1\result.mp4"

# r18_culane
imgs_path = r"D:\WorkDirectory\mywork\CLRNet-main\CLRNet-main\work_dirs\clr\r18_culane\20230510_173825_lr_6e-04_b_1\visualization"
video_save_path = r"D:\WorkDirectory\mywork\CLRNet-main\CLRNet-main\work_dirs\clr\r18_culane\20230510_173825_lr_6e-04_b_1\result.mp4"

fps = 25

if video_save_path != "":
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (1640, 590)
    out = cv2.VideoWriter(video_save_path, fourcc, fps, size)


for filename in os.listdir(imgs_path):
    print(imgs_path + "/" + filename)

    img = cv2.imread(imgs_path + '/' + filename)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imshow("video", img)

    # c = cv2.waitKey(1) & 0xff

    if video_save_path != "":
        out.write(img)

    # if c == 27:
    #     break

if video_save_path!="":
    print("Save processed video to the path :" + video_save_path)
    out.release()

# 关闭所有显示的窗口
cv2.destroyAllWindows()


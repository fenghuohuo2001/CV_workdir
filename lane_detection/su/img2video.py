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

# r18_culane
imgs_path = "./data2"
video_save_path = "data2/result.mp4"

fps = 1

if video_save_path != "":
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    size = (230, 748)
    out = cv2.VideoWriter(video_save_path, fourcc, fps, size)


for filename in os.listdir(imgs_path):
    print(imgs_path + "/" + filename)


    img = cv2.imread(imgs_path + '/' + filename)
    # cv2.imshow("video12", img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imshow("video1", img)

    # new_filename = '5' + filename

    # cv2.imwrite(os.path.join(imgs_path, new_filename), img)

    img = cv2.resize(img, (230, 748))
    print(img.shape)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("video", img)

    c = cv2.waitKey(1) & 0xff

    if video_save_path != "":
        out.write(img)

    # if c == 27:
    #     break

if video_save_path!="":
    print("Save processed video to the path :" + video_save_path)
    out.release()

# 关闭所有显示的窗口
cv2.destroyAllWindows()


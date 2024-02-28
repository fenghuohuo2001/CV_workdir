# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 1.video2frame.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/9/8 18:42
@Function：
"""

import cv2

def video2frame(videos_path, frames_save_path, time_interval):
    vidcap = cv2.VideoCapture(videos_path)

    '''
    .read()按帧读取视频; ret,frame是获.read()方法的两个返回值
    如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False
    frame就是每一帧的图像，是个三维矩阵
    '''
    success, image = vidcap.read()
    count = 0
    while success:
        success, image = vidcap.read()
        count += 1
        if count % time_interval == 0:
            cv2.imencode('.jpg', image)[1].tofile(frames_save_path + '/' + "frame%d.jpg" % count)
            '''
            ret, img = cv2.imencode()
            img is in ndarray format
            '''

            print("now imwrite", count)
        # if count == 20:
        #     break
    print("the number of pic is", count)

if __name__ == '__main__':
    videos_path = r'D:\1.Desktop file\Treasure file\project\wateringcart\water_filling_nozzle\data\video\2.mp4'
    frames_save_path = r'D:\1.Desktop file\Treasure file\project\wateringcart\water_filling_nozzle\data\pic\pic_video2'
    time_interval = 30  # 隔一帧保存一次
    video2frame(videos_path, frames_save_path, time_interval)

# -*- 实例4 -*-
"""
功能：读取文件夹内所有图片
作者：fenghuohuo
日期：2021年11月8日
"""
import cv2
import cv2 as cv
import os

def read_path(file_pathname):
    # 遍历该目录下所有图片文件
    for filename in os.listdir(file_pathname):
        print(filename)
        img = cv2.imread(file_pathname+'/'+filename)
        cv2.imwrite('D:\\1.Desktop file\\picture'+"\\"+filename, img)
        # change to gray
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 单通道灰度图变为三通道灰度图
        image_np = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # save
        cv2.imwrite('D:\\1.Desktop file\\picture'+"\\"+filename, image_np)

read_path("D:\\1.Desktop file\\picture")

# 路径：/      表示非本地
#      \,\\   表示本地

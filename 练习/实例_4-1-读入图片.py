# -*- 练习 -*-
"""
功能：
作者：fenghuohuo
日期：年月日
"""
import cv2
import os
import numpy as np
root_path = "I:/Images/2017_08_03/"
dir = root_path+"images"+"/"
count = 0
for root, dir, files in os.walk(dir):
    for file in files:
        srcImg = cv2.imread(root_path+"images"+"/"+str(file))
        roiImg = srcImg[36:521, 180:745]
        cv2.imwrite(root_path+"Image"+"/"+str(file),roiImg)
        count +=1
        if count%400==0:
            print(count)

'''
root保存的就是当前遍历的文件夹的绝对路径；
dirs保存当前文件夹下的所有子文件夹的名称（仅一层，孙子文件夹不包括）
files保存当前文件夹下的所有文件的名称
'''
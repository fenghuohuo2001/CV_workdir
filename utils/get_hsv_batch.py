"""
@Name: get_hsv_batch.py
@Auth: Huohuo
@Date: 2023/2/16-17:34
@Desc: 
@Ver : 
code_idea
"""
import csv
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")#忽略警告


def writercsv(stu1):
    if os.path.isfile('../detection_dir/thresh.csv'):
        with open('test_launch.csv', 'a', newline='') as f:
            csv_write = csv.writer(f, dialect='excel')
            csv_write.writerow(stu1)
    else:
        with open('../detection_dir/thresh.csv', 'w', newline='') as f:
            csv_write = csv.writer(f, dialect='excel')
            csv_write.writerow(stu1)

# 读取数据集
data_path = 'D:\WorkDirectory\school_project\dragline_detection\pre_data'
class_name = ['dirt', 'normal', 'scratch']

# 遍历目录下图片文件
for filename in os.listdir(data_path):
    for img_path in os.listdir(data_path + '/' + filename):
        print(filename + '/' + img_path)
        image = cv2.imread(data_path + '/' + filename + '/' + img_path)

        image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)))
        # image = cv2.resize(image, (int(image.shape[1]), int(image.shape[0])))
        # image=cv2.imread('szy.png')这是直接标定图片
        HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        HSV_val = []
        def getpos(event, x, y, flags, HSV_val):
            if event == cv2.EVENT_LBUTTONDOWN:  # 定义一个鼠标左键按下去的事件
                # print(HSV[y, x])
                HSV_val.append(HSV[y, x])

        cv2.imshow("imageHSV", HSV)
        # cv2.imshow('image',image)
        cv2.setMouseCallback("imageHSV", getpos, param=HSV_val)

        cv2.waitKey(0)

        H_val = []
        S_val = []
        V_val = []

        for i in range(len(HSV_val)):
            H_val.append(HSV_val[i][0])
            S_val.append(HSV_val[i][1])
            V_val.append(HSV_val[i][2])

        H_val = [[i] for i in H_val]
        S_val = [[i] for i in S_val]
        V_val = [[i] for i in V_val]

        vals = [H_val, S_val, V_val]

        def k_mean(vals):
            HSV_range = [[], [], []]

            for i in range(3):
                cluster = KMeans(n_clusters=3)
                cluster.fit(vals[i])  # 完成聚类
                # 获取聚类中心
                center = cluster.cluster_centers_
                HSV_range[i].append(int(min(center)))
                HSV_range[i].append(int(max(center)))
            return HSV_range

        HSV_range = k_mean(vals)
        print(HSV_range)
        writercsv(HSV_range)











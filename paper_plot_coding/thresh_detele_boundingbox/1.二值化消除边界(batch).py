"""
@Name: 1.二值化消除边界.py
@Auth: Huohuo
@Date: 2023/3/20-15:59
@Desc: 
@Ver : 
code_idea
"""
import os

import cv2
import numpy as np

# data_path = r"D:/WorkDirectory/mywork/ocr.pytorch-master/ocr.pytorch-master/train_code/train_crnn/rcnn_all_data/train_data/rcnn_train"
data_path = r"D:\WorkDirectory\mywork\ocr.pytorch-master\ocr.pytorch-master\train_code\train_crnn\rcnn_all_data\train_data\single"

# save_path = r"D:/WorkDirectory/mywork/ocr.pytorch-master/ocr.pytorch-master/train_code/train_crnn/rcnn_all_data/train_data/rcnn_train_thresh/"
# save_path = r"D:/WorkDirectory/mywork/ocr.pytorch-master/ocr.pytorch-master/train_code/train_crnn/rcnn_all_data/val_data/rcnn_thresh/"
save_path = r"D:\WorkDirectory\mywork\ocr.pytorch-master\ocr.pytorch-master\train_code\train_crnn\rcnn_all_data\train_data\single_result/"
for filename in os.listdir(data_path):
    img = cv2.imread(data_path + '/' + filename)
    # cv2.imshow("img", img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # cv2.imshow("thresh", img_thresh)
    cv2.imwrite(save_path + "thresh" + filename, img_thresh)
    # print(img.shape[0])
    k_size = int(img.shape[0]/13)
    # k_size = 5
    k = np.ones((k_size, k_size), np.uint8)
    # print(k_size)
    img_erode = cv2.erode(img_thresh, k, 3)
    # cv2.imshow("erode", img_erode)
    img_dilate = cv2.dilate(img_erode, k, 3)
    # cv2.imshow("dilate", img_dilate)
    img_open = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, k, 7)
    # cv2.imshow("img_open", img_open)
    cv2.imwrite(save_path + filename, img_open)
    print(filename)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

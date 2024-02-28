# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 4.计算交并比.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/9/5 9:59
@Function：
"""
import numpy as np


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = max(boxA[2], boxB[2])
    yB = max(boxA[3], boxB[3])

    interArea = max(0, xB-xA+1) * max(0, yB-yA+1)

    Area_A = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    Area_B = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(Area_A + Area_B - interArea + 1)

    return iou

def NMS(boxes, threshold):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes).astype("float")

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    w1 = x2 - x1
    h1 = y2 - y1

    area = (w1 + 1) * (h1 + 1)
    temp = []

    idxs = np.argsort(y2)   # 从小到大排列

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]      # get max
        temp.append(i)

        x1_m = np.maximum(x1[i], x1[idxs[:last]])
        y1_m = np.maximum(y1[i], y1[idxs[:last]])

        x2_m = np.minimum(x2[i], x2[idxs[:last]])
        y2_m = np.minimum(y2[i], y2[idxs[:last]])

        w = np.minimum(0, x2_m - x1_m + 1)
        h = np.minimum(0, y2_m - y1_m + 1)

        over = (w * h)/area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(over > threshold)[0])))

    return boxes[temp].astype("int")
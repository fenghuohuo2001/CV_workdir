# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 1.pytorch实现.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/6/9 20:07
@Function：
address：https://blog.csdn.net/weixin_39263657/article/details/121780657?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_utm_term~default-4-121780657-blog-105189544.pc_relevant_paycolumn_v3&spm=1001.2101.3001.4242.3&utm_relevant_index=6
"""

import numpy as np

def NMS(boxes, scores, iou_thresh, score_thresh=0.5):
    '''
    非极大值抑制(NMS)，用于过滤目标检测网络输出大量候选bboxs,
    如果多个类别需要对每一类进行循环求NMS，下面注释中可使用偏移量法
    进行多类别NMS
    numpy类型
    @param boxes: 候选框，shape:(N, 4),bbox:[[x_min, y_min, x_max, y_max]...]
    @param scores: 候选框的的置信度分数，shape:[N,]，tensor
    @param iou_thresh: 设定的IOU置信度
    @param score_thresh: 低score的阈值
    @return:经过NMS过滤后的bbox的index
    '''

    # 使用偏移量多类别分类求NMS， c为bbox的类别，0，1，2，3...，c.shape:[N,]
    # boxes = boxes[:, :4] + c
    # 也可以直接使用torchvision.ops.nms 进行NMS, 返回的i为经过NMS后的index
    # i = torchvision.ops.nms(boxes, scores, iou_thresh)

    # 过滤掉低于分数阈值的预测框
    boxes = boxes[np.where(boxes[:, -1] >= score_thresh)[0]]

    # 获取bbox左上和右下的坐标 x_min, y_min, x_max, y_max
    xmin = boxes[:, 0]   # xmin -> [xmin1, xmin2, ...]
    ymin = boxes[:, 1]   # ymin -> [ymin1, ymin2, ...]
    xmax = boxes[:, 2]   # xmax -> [xmax1, xmax2, ...]
    ymax = boxes[:, 3]   # ymax -> [ymax1, ymax2, ...]
    scores = boxes[:, 4]  # predict bbox class score -> [score1, score2, score3]

    # 按score降序排序，argsort返回降序后的索引。argsort为升序排列，[::-1]为倒序排序，最后即升序排列
    order = scores.argsort()[::-1]

    # 计算每个bbox的面积，+1防止面积为0
    areas = (xmax - xmin + 1) * (ymax - ymin + 1)  # 计算面积
    # 保留最优的结果，即经过NMS保留后的bbox的index
    keep = []

    # 搜索最佳边框
    # 当候选列表中还有目标就执行NMS
    while order.size > 0:
        # 获取当前得分最高的bbox的index
        top1_idx = order[0]
        # 添加到候选列表中
        keep.append(top1_idx)

        # 将得分最高的边框与剩余边框进行比较
        # 以下四个坐标为当前得分最高框与剩下框的交集矩形的左上和左下四个坐标
        xx1 = np.maximum(xmin[top1_idx], xmin[order[1:]])
        yy1 = np.maximum(ymin[top1_idx], ymin[order[1:]])
        xx2 = np.minimum(xmax[top1_idx], xmax[order[1:]])
        yy2 = np.minimum(ymax[top1_idx], ymax[order[1:]])

        # 计算交集
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = w * h

        # 计算并集
        union = areas[top1_idx] + areas[order[1:]] - intersection

        # 计算交并比
        iou = intersection / union

        # 将重叠度大于给定阈值的边框剔除掉，仅保留剩下的边框，返回相应的下标
        inds = np.where(iou <= iou_thresh)[0]

        # 从剩余的候选框中继续筛选，因为当前最大的score的index已放在keep中，所以order不需要了，所以需要+1
        order = order[inds + 1]

    return keep

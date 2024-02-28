"""
@Name: NMS.py
@Auth: Huohuo
@Date: 2023/6/12-22:15
@Desc: 
@Ver : 
code_idea

应该是一系列的斜率进行nms

利用计算得分流程：
1.计算总斜率之和：
2.

"""
import numpy as np



def nms(lines):
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)


def nms(boxes, scores, threshold):
    # boxes是包含边界框坐标的数组，scores是对应的置信度分数数组
    # threshold是用于筛选重叠边界框的阈值

    # 计算边界框的面积
    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)

    # 根据分数降序排序
    order = scores.argsort()[::-1]

    keep = []  # 用于保存最终的保留框

    while order.size > 0:
        # 取出置信度最高的框
        max_idx = order[0]
        keep.append(max_idx)

        # 计算当前框与其他框的重叠面积
        x1 = np.maximum(boxes[max_idx, 0], boxes[order[1:], 0])
        y1 = np.maximum(boxes[max_idx, 1], boxes[order[1:], 1])
        x2 = np.minimum(boxes[max_idx, 2], boxes[order[1:], 2])
        y2 = np.minimum(boxes[max_idx, 3], boxes[order[1:], 3])

        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)

        overlap = (w * h) / areas[order[1:]]

        # 根据重叠面积与阈值进行筛选
        inds = np.where(overlap <= threshold)[0]
        order = order[inds + 1]  # 更新order数组

    return keep

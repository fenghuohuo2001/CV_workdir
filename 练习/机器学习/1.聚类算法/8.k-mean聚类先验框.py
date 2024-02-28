# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 8.k-mean聚类先验框.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/6/10 14:35
@Function：对于先验框的设计方法其实就是iou（交并比）的k-means，交集与并集的比值
https://blog.csdn.net/qq_45804132/article/details/118693083?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165484266816781435466351%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=165484266816781435466351&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~baidu_landing_v2~default-1-118693083-null-null.article_score_rank_blog&utm_term=tree.findtext&spm=1018.2226.3001.4450

---------------------voc2017------------------
Annotation文件夹存放的是xml文件，该文件是对图片的解释，每张图片都对于一个同名的xml文件。
ImageSets文件夹存放的是txt文件，这些txt将数据集的图片分成了各种集合。如Main下的train.txt中记录的是用于训练的图片集合
JPEGImages文件夹存放的是数据集的原图片
SegmentationClass以及SegmentationObject文件夹存放的都是图片，且都是图像分割结果图（楼主没用过，所以不清楚


"""
import glob
import xml.etree.ElementTree as ET

import numpy as np

# 计算交并比
def cas_iou(box, cluster):
    # 取聚类框与box之间较小坐标
    x = np.minimum(cluster[:, 0], box[0])   # 逐位比较 聚类框和box 对应位置上的值，保留较小值
    y = np.minimum(cluster[:, 1], box[1])   # x[:,n]表示在全部数组（维）中取第n个数据，直观来说，x[:,n]就是取所有集合的第n个数据,
    # 交集
    intersection = x * y

    area1 = box[0] * box[1]
    area2 = cluster[:, 0] * cluster[:, 1]
    iou = intersection / (area1 + area2 - intersection)

    return iou

def avg_iou(box, cluster):
    return np.mean([np.max(cas_iou(box[i], cluster)) for i in range(box.shape[0])])


def kmeans(box, k):
    # 取出一共有多少框
    row = box.shape[0]

    # 每个框各个点的位置
    distance = np.empty((row, k))

    # 最后的聚类位置
    last_clu = np.zeros((row,))

    np.random.seed()

    # 随机选5个当聚类中心
    cluster = box[np.random.choice(row, k, replace=False)]
    # cluster = random.sample(row, k)
    while True:
        # 计算每一行距离五个点的iou情况。
        for i in range(row):
            distance[i] = 1 - cas_iou(box[i], cluster)

        # 取出最小点
        near = np.argmin(distance, axis=1)

        if (last_clu == near).all():
            break

        # 求每一个类的中位点
        for j in range(k):
            cluster[j] = np.median(
                box[near == j], axis=0)

        last_clu = near

    return cluster


def load_data(path):
    data = []
    for xml_file in glob.glob('{}/*xml'.format(path)):  # 读取xml文件
        # XML是中结构化数据形式，在ET中使用ElementTree代表整个XML文档，并视其为一棵树，Element代表这个文档树中的单个节点。
        tree = ET.parse(xml_file)
        # 读取voc2007中的xml文件中的图像高和宽
        height = int(tree.findtext('./size/height'))        # 寻找第一个匹配子元素，返回其text值。匹配对象可以为tag或path。
        width = int(tree.findtext('./size/width'))
        if height <= 0 or width <= 0:
            continue

        # 对于每一个目标都获得它的宽高，并且归一化
        for obj in tree.iter('object'):     # 进入object类中
            xmin = int(float(obj.findtext('bndbox/xmin'))) / width
            ymin = int(float(obj.findtext('bndbox/ymin'))) / height
            xmax = int(float(obj.findtext('bndbox/xmax'))) / width
            ymax = int(float(obj.findtext('bndbox/ymax'))) / height
            # 转化格式
            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            # 得到归一化后的宽高
            data.append([xmax - xmin, ymax - ymin])
    return np.array(data)


if __name__ == '__main__':
    # 运行该程序会计算'./VOCdevkit/VOC2007/Annotations'的xml
    # 会生成yolo_anchors.txt
    SIZE = 416
    anchors_num = 9
    # 载入数据集，可以使用VOC的xml
    path = r'D:\1.Desktop file\pythonProject\VOCdevkit\VOCdevkit\VOC2012\Annotations'
    # path = './data/label'
    # 载入所有的xml
    # 存储格式为转化为比例后的width,height
    data = load_data(path)
    # 使用k聚类算法
    out = kmeans(data, anchors_num)
    out = out[np.argsort(out[:, 0])]
    # 有效性评价
    print('acc:{:.2f}%'.format(avg_iou(data, out) * 100))
    print(out * SIZE)
    # 图片长宽按416设置
    data = out * SIZE
    # 将聚类的得到的先验框尺寸存入txt中
    f = open("yolo_anchors.txt", 'w')
    row = np.shape(data)[0]
    for i in range(row):
        if i == 0:
            x_y = "%d,%d" % (data[i][0], data[i][1])    # %表示字符串格式化，将其他变量存入字符串特定位置
        else:                                           # 用逗号间隔
            x_y = ", %d,%d" % (data[i][0], data[i][1])
        f.write(x_y)
    f.close()

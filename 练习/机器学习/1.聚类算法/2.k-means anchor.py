# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 2.k-means anchor.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/6/9 20:07
@Function：
address：https://blog.csdn.net/weixin_39263657/article/details/121780657?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_utm_term~default-4-121780657-blog-105189544.pc_relevant_paycolumn_v3&spm=1001.2101.3001.4242.3&utm_relevant_index=6
"""

'''
使用 k-means 生成 anchors
'''
import glob
import xml.etree.cElementTree as ET
import numpy as np

'''
步骤：
1. 将bbox的的宽 和高使用其对应图像的宽高进行归一化；
2. 开始进行k-means
    （1）初始化K个簇中心；(一般k设为9，yolov5中是三个不同的feature map 各有三个不同尺度的wh，初始的anchors的wh一般是根据 coco或者voc等公共数据集得到的)
    （2）使用相似性度量，将每个样本分配给距离最近的簇中心；(这里一般使用 1-iou 作为距离度量，yolov5中则使用 gt框与anchor对应宽比和高比作为距离度量，与yolov5 NMS筛选的条件一致。
         将N个bbox与这9个anchors作距离计算，最终计算出(N,9)个距离值)
    （3）计算每个簇中所有样本的均值，更新簇中心；(找到每一行中最小的距离值，即当前bbox被分到了哪个簇中，然后计算每个簇(列)的均值以对簇中心进行更新)
    （4）重复（2）（3）步，直到均簇中心不再变化，或者达到了最大迭代次数。
'''


# 1. 对数据集中bbox的宽和高进行归一化
def load_dataset(path):
    '''
    先对bbox的左上和右下的坐标点 使用其对应图像的宽高进行归一化
    然后使用 xmax - xmin, ymax - ymin 得到归一化后的宽高
    @param path:
    @return:
    '''
    dataset = []
    for xml_file in glob.glob("{}/*xml".format(path)):  # 在文件路径下寻找xml文件
        tree = ET.parse(xml_file)   # 读取xml

        height = int(tree.findtext("./size/height"))
        width = int(tree.findtext("./size/width"))

        for obj in tree.iter("object"):
            xmin = int(obj.findtext("bndbox/xmin")) / width
            ymin = int(obj.findtext("bndbox/ymin")) / height
            xmax = int(obj.findtext("bndbox/xmax")) / width
            ymax = int(obj.findtext("bndbox/ymax")) / height

            dataset.append([xmax - xmin, ymax - ymin])

    return np.array(dataset)

# 2. 距离度量
def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]
    distances = np.empty((rows, k))

    last_clusters = np.zeros((rows,))
    np.random.seed()
    # the Forgy method will fail if the whole array contains the same rows
    # 初始化k个聚类中心（从原始数据集中随机选择k个）
    clusters = boxes[np.random.choice(rows, k, replace=False)]
    while True:
        for row in range(rows):
            # 定义的距离度量公式：d(box,centroid)=1-IOU(box,centroid)。到聚类中心的距离越小越好，
            # 但IOU值是越大越好，所以使用 1 - IOU，这样就保证距离越小，IOU值越大。
            # 计算所有的boxes和clusters的值（row，k）
            # 2-(1), 2-(2), 距离度量
            distances[row] = 1 - iou(boxes[row], clusters)
            # print(distances)
        # 将标注框分配给“距离”最近的聚类中心（也就是这里代码就是选出（对于每一个box）距离最小的那个聚类中心）。
        nearest_clusters = np.argmin(distances, axis=1)
        # 直到聚类中心改变量为0（也就是聚类中心不变了）。
        if (last_clusters == nearest_clusters).all():
            break
        # 计算每个群的中心（这里把每一个类的中位数作为新的聚类中心）
        # 2-(3) 更新簇中心(anchors)，有均值计算得出
        for cluster in range(k):
            # 这一句是把所有的boxes分到k堆数据中,比较别扭，就是分好了k堆数据，每堆求它的中位数作为新的点
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)
        last_clusters = nearest_clusters
    return clusters


# 2-(1), 2-(2), 距离度量
def iou(box, clusters):
    '''
    计算当前box与9个anchors的距离度量值
    @param box: box.shape=[2], [width, height]
    @param clusters: clusters.shape=[9, 2], 9个anchors 的 w 和 h,
    @return: iou_.shape = [9, 1]
    '''
    # 计算每个box与9个clusters的iou
    # boxes ： 所有的[[width, height], [width, height], …… ]
    # clusters : 9个随机的中心点[width, height]

    # 计算iou时只需要wh，默认以左上角顶点为原点进行计算,
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")
    intersection = x * y

    # box的面积
    box_area = box[0] * box[1]
    # 所有anchors的面积
    cluster_area = clusters[:, 0] * clusters[:, 1]

    # 当前box与9个anchors的iou值
    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


# 评估
def avg_iou(boxes, clusters):
    '''
    评估由 k-means 生成的anchors与数据集中的bboxs的重合度
    @param boxes:    boxes.shape = [N, 2], N个bbox的 w 和 h
    @param clusters: clusters.shape = [9, 2], 9个anchors 的 w 和 h,
    @return:
    '''
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])

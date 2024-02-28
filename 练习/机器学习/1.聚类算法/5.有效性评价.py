# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 5.有效性评价.py
@IDE     : PyCharm
@Author  : https://www.bilibili.com/video/BV1zU4y137WK?spm_id_from=333.880.my_history.page.click
@Date    : 2022/6/10 10:07
@Function：聚类算法属于无监督学习，不同聚类数对聚类结果影响大
因此需要进行聚类有效性评价，聚类评价指标常用的有：Rand指数，轮廓系数（Silhouette Coefficient），Calinski-Harabaz指数等等
"""
# 数据导入
import pandas as pd

datA = pd.read_excel(r'wind.xlsx')
print(datA)

data = datA["Wind"]
print(datA["Wind"])

# 这样的数据k-means函数是不能进行计算的，需要进行数据处理
import numpy as np

data = np.array(data)   # k-mean不能计算数组
print(data)

# 用遍历的方法将数组转化成列表,变成可识别格式
data = [[i] for i in data]
print(data)

# k-mean聚类
from sklearn.cluster import KMeans
cluster = KMeans(n_clusters=3)  # 设置质心个数，默认8类，不过一开始并不知道要分成几类
cluster.fit(data)   # 完成聚类
print(cluster.fit(data))

# 获取聚类标签
label = cluster.labels_
print(label)
# 获取聚类中心
center = cluster.cluster_centers_
print(center)

# ------------------二维聚类--------------------------
data1 = datA['Wind']
data1 = np.array(data1)

data2 = datA['Temp']
data2 = np.array(data2)

DATA2 = np.vstack((data1, data2)).T
# DATA3 = np.vstack((data1, data2))
print(DATA2)
# print(DATA3)

cluster_2D = KMeans(n_clusters=3)
cluster_2D.fit(DATA2)
print("cluster_2D.fit(DATA2)", cluster_2D.fit(DATA2))

label_2D = cluster_2D.labels_
print("label_2D", label_2D)
center_2D = cluster_2D.cluster_centers_
print("center_2D", center_2D)



# 轮廓系数进行评价
# 轮廓系数处于[-1,1],越大表示簇间相似度高，而不同簇相似度低，即聚类效果越好
from sklearn.metrics import silhouette_samples
# 得到每一个数据对应的轮廓系数
lkxs = silhouette_samples(DATA2, label)
print(lkxs)
# 用轮廓系数的均值表示不同聚类数的好坏
means = np.mean(lkxs)
print(means)

# 写一个循环来计算聚类数从2到n-1的轮廓系数进行聚类评价
def cluster_eval(n):
    cluster = KMeans(n_clusters=n)
    cluster.fit(DATA2)
    label = cluster.labels_
    center = cluster.cluster_centers_
    lkxs = silhouette_samples(DATA2, label, metric='euclidean')
    means = np.mean(lkxs)
    return means

y = []
for n in range(2, 6):       # 从2类开始，到总点数
    means = cluster_eval(n)
    y.append(means)
print(y)

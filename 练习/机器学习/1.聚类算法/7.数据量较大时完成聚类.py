# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 6.创建数据集完成聚类.py
@IDE     : PyCharm
@Author  : https://blog.csdn.net/oxygensss/article/details/117093286?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165482571816782425162362%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=165482571816782425162362&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-2-117093286-null-null.142^v12^pc_search_result_cache,157^v13^control&utm_term=KMeans%28n_clusters%3D2%29&spm=1018.2226.3001.4187
@Date    : 2022/6/10 10:46
@Function：
当我们数据量非常大的时候，我们可以使用部分数据来帮助我们确认质心，剩下的数据的聚类结果，使用predict来调用。
predict表示学习数据X并对X的类进行预测（对分类器.fit()之后，再预测)
fit_predict不需要分类器.fit()之后都可以预测
分类器.fit(X).predict的结果= 分类器.fit_predict(X)=分类器.fit(X).labels_
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
plt.style.use('ggplot')     # 设置绘图背景

# 自建数据集
centers = 50    # 质心数量
# 生成500*2的数据集，每一组数据有4个中心点，即数据集有四个标签
X, y = make_blobs(n_samples=500, n_features=2, centers=centers, random_state=1)
plt.scatter(X[:, 0], X[:, 1], marker='o', s=8)  # 画散点图，点样式为圆，s是点大小，默认20
plt.show()

# color = ["red", "pink", "orange", "gray"]
# for i in range(4):
#     plt.scatter(X[y == i, 0], X[y == i, 1], marker='o', s=8, c=color[i])
# plt.show()

# 基于以上分布进行k-mean聚类
# n_clusters = 3
# cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(X)  # random_state=0 初始质心位置固定
# y_pred = cluster.labels_
# y_center = cluster.cluster_centers_
# print(y_pred)
# print(y_center)

# 评估聚类个数
from sklearn.metrics import silhouette_samples

def cluster_eval(X, n):
    cluster = KMeans(n_clusters=n)
    cluster.fit_predict(X)  # 用训练器数据X拟合分类器模型并对训练器数据X进行预测,可以预测相同数据集下的新的数据
    label = cluster.labels_
    eval_silhouette = silhouette_samples(X, label, metric='euclidean')
    means = np.mean(eval_silhouette)
    return means

def get_best_n(X,num):
    y = []
    for n in range(2, num):       # 从2类开始，到总点数
        means = cluster_eval(X, n)
        y.append(means)
    print(y)
    max = np.max(y)
    max_num = np.argmax(y)
    return max, max_num

max, max_num = get_best_n(X, centers)
print(max)
print("最佳分类簇数：", max_num+2)
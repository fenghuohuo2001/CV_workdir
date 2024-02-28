# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 3.pandas读取excel.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/6/10 8:52
@Function：
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
print("goal", data)

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




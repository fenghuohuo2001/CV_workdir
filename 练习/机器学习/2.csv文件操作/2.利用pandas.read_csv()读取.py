# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 2.利用pandas.read_csv()读取.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/5/31 17:35
@Function：
"""
import numpy as np
import pandas as pd

my_matrix = pd.read_csv('./data/score1.csv')
print("my_matrix:", my_matrix)
print("my_matrix.shape:", my_matrix.shape)
# 为了后续数据分析，可将数据类型改为np.array型
my_matrix = np.array(my_matrix)
print("my_matrix[0][0]:", my_matrix[0][0])


# header : int or list of ints, default ‘infer’,指定行数用来作为列名，数据开始行数。如果文件中没有列名，则默认为0，
# index_col : int or sequence or False, default None,用作行索引的列编号或者列名

my_matrix2 = pd.read_csv('./data/score1.csv', header=None, index_col=None)
print("my_matrix2:", my_matrix2)
print("my_matrix.shape2:", my_matrix2.shape)
my_matrix = np.array(my_matrix2)
print("my_matrix2[0][0]:", my_matrix2[0][0])
# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : np.pad.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/10/9 16:22
@Function：
"""
import numpy as np

test = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])

print(test.shape)

test_pad = np.pad(test, ((0, 1), (0, 1), (0, 0)), 'constant')
print(test_pad)
print(test_pad.shape)
# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 3.pandas读取excel.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/6/10 8:52
@Function：
"""

import sklearn

# 1.数据导入
import pandas as pd

datA = pd.read_excel(r'wind.xlsx')
print(datA)

print(datA["Wind"])
# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 5.csv文件行列转化.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/7/7 22:59
@Function：
"""
# 方法1
import csv
def storFile(data,fileName):
    with open(fileName,'w',newline ='') as f:
        mywrite = csv.writer(f)
        mywrite.writerow(data)
data = [1,0,1,1,0,1,0]
storFile(data,'splitData\wodecesi.csv')
# 所得csv文件中，列表元素为 1  0  1  1  0  1  0  排列；

# 方法2
import csv
def storFile(data,fileName):
    data = list(map(lambda x:[x],data))
    with open(fileName,'w',newline ='') as f:
        mywrite = csv.writer(f)
        for i in data:
            mywrite.writerow(i)
data = [1,0,1,1,0,1,0]
storFile(data,'splitData\wodecesi.csv')

# 方法3
data.to_csv('splitData\wodecesi.csv',index = False)
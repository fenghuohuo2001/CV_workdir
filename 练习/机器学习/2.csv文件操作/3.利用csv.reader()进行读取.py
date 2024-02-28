# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 3.利用csv.reader()进行读取.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/5/31 17:49
@Function：
"""
import csv
# 读取csv文件
with open('./data/score1.csv', "r", encoding='utf8', newline='') as f:
	reader = csv.reader(f)
	# 按逐行输出的形式，输出整个表格
	for row in reader:
		print(row)
	# 如果想获取某一列，可以通过指定的列标号来查询
	for row in reader:
		print(row[0])
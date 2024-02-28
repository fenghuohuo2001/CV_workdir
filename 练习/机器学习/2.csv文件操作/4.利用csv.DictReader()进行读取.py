# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 4.利用csv.DictReader()进行读取.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/5/31 17:52
@Function：
"""
import csv
# 读取csv文件
with open('./data/score1.csv', "r", encoding='utf8', newline='') as f:
	reader = csv.DictReader(f)
	for row in reader:
		print(row)
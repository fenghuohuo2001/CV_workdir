# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 1.csv文件创建.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/5/31 16:57
@Function：CSV是一种以逗号分隔数值的文件类型，在数据库或电子表格中，常见的导入导出文件格式就是CSV格式，CSV格式存储数据通常以纯文本的方式存数数据表。
          即：操作表格数据
"""
import csv

# --------------------------------------------------
# 1-使用csv.writer()创建
headers = ['学号', '姓名', '分数']
rows = [('202001', '张三', '98'),
        ('202002', '李四', '95'),
        ('202003', '王五', '92')]
with open("./data/score1.csv", 'w', encoding='utf8', newline='') as f:     # 读写csv文件时就要设置newline=''
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(rows)

# --------------------------------------------------
# 2-使用csv.Dictwriter()创建：
headers = ['学号', '姓名', '分数']
rows = [{'学号': '202001', '姓名': '张三', '分数': '98'},
        {'学号': '202002', '姓名': '李四', '分数': '95'},
        {'学号': '202003', '姓名': '王五', '分数': '92'}]
with open('./data/score2.csv', 'w', encoding='utf8', newline='') as f:     # 读写csv文件时就要设置newline=''
    writer = csv.DictWriter(f, headers)
    writer.writeheader()
    writer.writerows(rows)

# --------------------------------------------------
headers = ['学号,姓名,分数', '\n']
csv = ['202001,张三,98', '\n',
       '202002,李四,95', '\n',
       '202003,王五,92']
with open('./data/score3.csv', 'w', encoding='utf8', newline='') as f:     # 读写csv文件时就要设置newline=''
    f.writelines(headers)  # write() argument must be str, not tuple
    f.writelines(csv)
# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 1.创建csv存数组.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/7/7 22:09
@Function：
"""
import csv
import os


def writercsv():
    stu1 = ['1', '2', '3']
    stu2 = ['4', '5', '6']
    if os.path.isfile('test_launch.csv'):
        with open('test_launch.csv', 'a', newline='') as f:
            csv_write = csv.writer(f, dialect='excel')
            csv_write.writerow(stu1)
            csv_write.writerow(stu2)
    else:
        with open('test_launch.csv', 'w', newline='') as f:
            csv_write = csv.writer(f, dialect='excel')
            csv_write.writerow(['设备名称', 'android版本号', '分辨率', 'app版本号', '第一次', '第二次', '第三次', '平均值', '测试时间'])
            csv_write.writerow(stu1)
            csv_write.writerow(stu2)


if __name__ == '__main__':
    writercsv()
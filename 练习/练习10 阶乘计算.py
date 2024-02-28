# -*- 练习10 阶乘计算 -*-
"""
功能：阶乘计算1到10
作者：fenghuohuo
日期：2021年6月7日
"""
sum,tmp = 0, 1
for i in range(1,11):
    tmp*=i
    sum+=tmp
print("得到的结果是：{}".format(sum))
# -*- 练习8 整数序列求和 -*-
"""
功能：整数序列求和
作者：fenghuohuo
日期：2021年6月7日
"""
n = input("请输入整数N：")
sum = 0
for i in range(int(n)):
    sum += i + 1
print("从1到N求和结果为：", sum)
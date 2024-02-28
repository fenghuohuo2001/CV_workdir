# -*- 练习1-4 -*-
"""
功能：def用法
作者：fenghuohuo
日期：2021年6月21日
"""
def fact(n):
    result = 1
    for i in range(1 , n + 1):
        result = result * 1
    return result

n = int(input())
print(fact(n))
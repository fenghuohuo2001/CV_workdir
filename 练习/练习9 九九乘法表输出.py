# -*- 练习9 九九乘法表输出 -*-
"""
功能：九九乘法表输出
作者：fenghuohuo
日期：2021年6月7日
"""
for i in range(1,10):
    for j in range(1,i+1):
        print ("{} * {}={:2}  ".format(j,i,i*j),end='')
    print('')
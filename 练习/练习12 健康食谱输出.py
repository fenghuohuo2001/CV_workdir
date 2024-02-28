# -*- 练习12 -*-
"""
功能：健康食谱输出
作者：fenghuohuo
日期：2021年6月7日
"""
diet = ['西红柿','花椰菜','黄瓜','牛排','虾仁']
for x in range(0,5):
    for y in range(0,5):
        if not(x == y):
            print("{}{}".format(diet[x],diet[y]))
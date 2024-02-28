# -*- 实例_3 -*-
"""
功能：平行度检测
作者：fenghuohuo
日期：2021年11月8日
"""
import numpy as np
import numpy as py
# 已知直线的长度为L 两条直线夹角为threta
L = float(input("请输入基准直线长度："))
threta = float(input("请输入待测直线与基准直线之间的夹角(度数)："))
Parallelism = L * ((threta*np.pi)/180)
print("目标直线平行度为：%.4f"%Parallelism)
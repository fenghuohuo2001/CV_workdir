# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 3.迭代法分割.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/8/4 18:53
@Function：

迭代法分割
1.求出图像 max_gray 和 min_gray，初始阈值记为 t0=(max_gray + min_gray)/2
2 根据 ti 将图像分为前景和背景，求出两者平均灰度值 ZO 和 ZB
3 求出新阈值 ti+1 = （ZO + ZB）/2
4. 倘若ti = ti+1, 则所求阈值即最终阈值，否则，跳转到2 重复迭代

用于确定最佳分割阈值


不过用大津法即可替代
"""

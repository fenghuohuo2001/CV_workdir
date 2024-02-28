# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 1.imread转为RGB.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/7/10 15:29
@Function：
"""
#第一组
image = cv2.imread(path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)          # 转化为RGB

#第二组
image = cv2.imread(path)
image = image[:, :, (2, 1, 0)]

#第三组
image = cv2.imread(path)
image = image[:,:,::-1]

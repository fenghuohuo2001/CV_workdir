# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 10.最大内接圆.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/6/18 14:04
@Function：
"""
# 读取图片，转灰度
mask = cv2.imread(mask_path)
mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

# 识别轮廓
contours, _ = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 计算到轮廓的距离
raw_dist = np.empty(mask_gray.shape, dtype=np.float32)
for i in range(mask_gray.shape[0]):
    for j in range(mask_gray.shape[1]):
        raw_dist[i, j] = cv2.pointPolygonTest(contours[0], (j, i), True)

# 获取最大值即内接圆半径，中心点坐标
minVal, maxVal, _, maxDistPt = cv2.minMaxLoc(raw_dist)
minVal = abs(minVal)
maxVal = abs(maxVal)

# 画出最大内接圆
result = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
radius = np.int(maxVal)
center_of_circle = maxDistPt
cv2.circle(result, maxDistPt, radius, (0, 255, 0), 2, 1, 0)
cv2.imshow('Maximum inscribed circle', result)
cv2.waitKey(0)
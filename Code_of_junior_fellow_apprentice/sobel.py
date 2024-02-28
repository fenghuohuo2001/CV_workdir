
'''

应该使用矩阵乘法，而不是for循环
'''
import cv2
import numpy as np

# 导入图片
img = cv2.imread('img.png')
# 转变为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 获取图片大小
sp = gray.shape
print(sp)
height = sp[0]
weight = sp[1]
# 创建卷积核
sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
# 创建全零数组来存放结果
dSobel = np.zeros((height, weight))
dSobelx = np.zeros((height, weight))
dSobely = np.zeros((height, weight))
Gx = np.zeros(gray.shape)
Gy = np.zeros(gray.shape)
# 利用for循环进行卷积，i和j都以0为起始值，height-3为终止值，步长为1
for i in range(height - 2):
    for j in range(weight - 2):
        # 对应元素相乘求和再取绝对值
        Gx[i + 1, j + 1] = abs(np.sum(gray[i:i + 3, j:j + 3] * sx))
        Gy[i + 1, j + 1] = abs(np.sum(gray[i:i + 3, j:j + 3] * sy))
        # 将Gx数组与Gy数组的元素计算平方和并开根号
        dSobel[i + 1, j + 1] = (Gx[i + 1, j + 1] * Gx[i + 1, j + 1] + Gy[i + 1, j + 1] * Gy[i + 1, j + 1]) ** 0.5
        # 分别对Gx和Gy数组的元素开平方
        dSobelx[i + 1, j + 1] = np.sqrt(Gx[i + 1, j + 1])
        dSobely[i + 1, j + 1] = np.sqrt(Gy[i + 1, j + 1])
    print("1")

# 打印结果
cv2.imshow('a', img)
cv2.imshow('b', gray)
cv2.imshow('c', np.uint8(dSobel))
cv2.imshow('d', np.uint8(dSobelx))
cv2.imshow('e', np.uint8(dSobely))
cv2.waitKey(0)
cv2.destroyAllWindows()

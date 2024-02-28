"""
@Name: 3.高斯距离.py
@Auth: Huohuo
@Date: 2023/3/9-15:27
@Desc: 
@Ver : 
code_idea
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 构建灰度图高斯距离公式
def Gaussian_Function(ndarray, center, sigma):
    center_x = center[0]
    center_y = center[1]

    y = []
    result = np.zeros(ndarray.shape, np.uint8)
    for i in range(ndarray.shape[0]):
        for j in range(ndarray.shape[1]):
            # 欧式距离
            euclidean_distance = (abs(i - center_x))**2 + (abs(j - center_y))**2
            # 高斯核函数
            gas_kernel = (-euclidean_distance / (2 * (sigma ** 2)))
            # 放大因子
            lambd = 1
            gas_distance = (1*lambd / (np.sqrt(2 * np.pi) * (sigma ** 2)))*(np.exp(gas_kernel))
            # print("gas_distance{}".format(gas_distance, '.6f'))
            result[i, j] = gas_distance
            y.append(gas_distance)
    num = int(ndarray.shape[0] * ndarray.shape[1])
    x = np.arange(0, num)
    plt.plot(x, y)
    plt.show()

    # gas_distance = (1 / np.sqrt(2 * np.pi))*(np.exp)
    return result


img_path = "../Retinex/data/crop_0.png"
img = cv2.imread(img_path, 0)
print(type(img))
print(img.shape)
# 图像中心
img_center = (int(img.shape[0]/2), int(img.shape[1]/2))
# 论文中，v=3.34 H=img.shape[0]
img_gas_dis = Gaussian_Function(img, img_center, 1)
# cv2.imshow("gas_distance", img_gas_dis)
# cv2.waitKey(0)






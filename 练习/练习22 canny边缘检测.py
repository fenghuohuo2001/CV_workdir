# -*- 练习22 -*-
"""
功能：边缘检测
作者：fenghuohuo
日期：2021年6月22日
"""
import cv2
img = cv2.imread("yuantu.png")
canny = cv2.Canny(img, 0, 1)   #255是白,在此范围内才会显现，靠近255白会显现，

cv2.imshow("canny",canny)
cv2.imshow("yuantu",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
 edges 为计算得到的边缘图像。
 image 为 8 位输入图像。
 threshold1 表示处理过程中的第一个阈值。
 threshold2 表示处理过程中的第二个阈值。
 apertureSize 表示 Sobel 算子的孔径大小。
 L2gradient 为计算图像梯度幅度（gradient magnitude）的标识。其默认值为 False。如果为 True，则使用更精确的 L2 范数进行计算（即两个方向的导数的平方和再开方），否则使用 L1 范数（直接将两个方向导数的绝对值相加）。
示例：
————————————————
版权声明：本文为CSDN博主「LGP是人间理想」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/m0_51402531/article/details/121066693
"""
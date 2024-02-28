"""
@Name: 5.k-mean阈值分割.py
@Auth: Huohuo
@Date: 2023/2/19-13:46
@Desc: 
@Ver : 
code_idea
"""
import cv2
import numpy as np

img_path = r"D:\WorkDirectory\school_project\dragline_detection\stay_cable_photo\J16\dirt\64388mm\left_back-2022-12-14-07-25-33-64388mm-65740mm.jpeg"

img = cv2.imread(img_path)
img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
img_copy = img.copy()

def kmean_cut(img):
    # kmean均值聚类
    img_reshape = img.reshape((-1, 3))
    img_float32 = np.float32(img_reshape)

    method = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    center_num = 2
    ret, label, center = cv2.kmeans(img_float32, center_num, None, method, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]

    img_result = res.reshape((img.shape))
    cv2.imshow("result", img_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def watersh(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 分水岭算法
    ret1, img10 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # （图像阈值分割，将背景设为黑色）
    cv2.namedWindow("W3")
    cv2.imshow("W3", img10)
    cv2.waitKey(delay=0)
    ##noise removal（去除噪声，使用图像形态学的开操作，先腐蚀后膨胀）
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(img10, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area(确定背景图像，使用膨胀操作)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area（确定前景图像，也就是目标）
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret2, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region（找到未知的区域）
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Marker labelling
    ret3, markers = cv2.connectedComponents(sure_fg)  # 用0标记所有背景像素点
    # Add one to all labels so that sure background is not 0, but 1（将背景设为1）
    markers = markers + 1
    ##Now, mark the region of unknown with zero（将未知区域设为0）
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)  # 进行分水岭操作
    img[markers == -1] = [0, 0, 255]  # 边界区域设为-1，颜色设置为红色
    cv2.namedWindow("W4")
    cv2.imshow("W4", img)
    cv2.waitKey(delay=0)


if __name__ == "__main__":
    kmean_cut(img_copy)
    # watersh(img_copy)
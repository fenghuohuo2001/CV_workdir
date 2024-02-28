"""
@Name: 4.GradCut分割.py
@Auth: Huohuo
@Date: 2023/2/19-13:29
@Desc: 
@Ver : 
code_idea
"""
import numpy as np
import cv2


img_path = r"D:\WorkDirectory\school_project\dragline_detection\stay_cable_photo\J16\dirt\64388mm\left_back-2022-12-14-07-25-33-64388mm-65740mm.jpeg"

img = cv2.imread(img_path)
img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
img_copy = img.copy()

# rect = (242, 0, 385, 540)       # 精确
# rect = (118, 0, 666, 540)       # 大范围
rect = (187, 0, 505, 540)       # 中等范围


def gradcut(img_copy, rect):
    mask = np.zeros(img_copy.shape[:2], np.uint8)

    # 创建零填充的背景和前景模型
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # 使用矩形分割
    cv2.grabCut(img_copy, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')

    result = cv2.bitwise_and(img, img, mask=mask2)
    cv2.imshow("result", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    gradcut(img_copy, rect)
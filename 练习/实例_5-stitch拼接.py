# -*- 练习 -*-
"""
功能：
作者：fenghuohuo
日期：年月日
"""
import sys
import cv2

if __name__ == "__main__":
    img1 = cv2.imread('stitch1-1.png')  # 图片绝对路径，
    img2 = cv2.imread('stitch1-2.png')

    stitcher = cv2.createStitcher(False)    # 老的OpenCV版本，用这一个
    # stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)  # 我的是OpenCV4

    (status, pano) = stitcher.stitch((img1, img2))
    if status != cv2.Stitcher_OK:
        print("不能拼接图片, error code = %d" % status)
        sys.exit(-1)
    print("拼接成功.")
    cv2.imshow('pano', pano)
    # cv2.imwrite("pano.jpg", pano)
    cv2.waitKey(0)

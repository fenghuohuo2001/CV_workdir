# -*- 练习19-2 -*-
"""
功能：实例
作者：fenghuohuo
日期：2021年6月22日
"""
import cv2
import sys
import numpy as np

if __name__ == '__main__':
    image = cv2.imread('rabbit.png', cv2.IMREAD_GRAYSCALE)
    #原图的高、宽
    h,w =image.shape
    #原图的四个点与投影变换对应点
    src = np.array([[0,0],[w-1,0],[0,h-1],[w-1,h-1]],np.float32)
    dst = np.array([[50,50],[w/3,50],[50,h-1],[w-1,h-1]],np.float32)
    #计算投影变换矩阵
    p = cv2.getPerspectiveTransform(src , dst)
    #利用变换矩阵进行头像的投影变换
    r = cv2.warpPerspective(image,p,(w,h),borderValue=125)
    '''
    实现投影变换功能
    cv2.warpPerspective(src,M,dsize[,dst[,flags[,borderMode[,borderValue]]]])
    '''#投影变换功能
    #显示原图和投影效果
    cv2.imshow("image",image)
    cv2.imshow("warpPerspective",r)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

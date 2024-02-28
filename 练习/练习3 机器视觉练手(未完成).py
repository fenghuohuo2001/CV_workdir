# -*- 练习3 机器视觉练手 -*-
"""
功能：
作者：fenghuohuo
日期：2021年06月03日
"""
# -*- coding: utf-8 -*-
import cv2
import sys
#主函数
if __name__ =="__main__":
    if len(sys.argv) > 1:
        #输入图像
        image = cv2. imread(sys. argv[1], cv2. CV_LOAD_IMAGE_UNCHANGED)
    else:
        print("Usge :python f irstOpenCV2. py imageFile")
    #显示图像
    cv2. imshow(" image", image)
    cv2. waitKey(0)
    cv2. destroyAllWindows()
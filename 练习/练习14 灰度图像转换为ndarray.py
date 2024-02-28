# -*- 练习14 -*-
"""
功能：灰度图像转换为矩阵
作者：fenghuohuo
日期：2021年6月8日
"""
import sys
import cv2
import numpy as n
#主函数
if __name__ =="__main__":
    img0 = cv2.imread('rabbit.png')
    img = cv2.imread('rabbit.png', cv2.IMREAD_GRAYSCALE) #或用img = cv2.imread('kkatw.png',0)
    print(img)
else:
    print("Usge:python imgToAarry.py imageFile")
#显示图像
cv2.imshow("img1",img0)
cv2.imshow("img",img)       #第一个参数是显示图像的窗口的名字，第二个参数是要显示的图像（imread读入的图像），窗口大小自动调整为图片大小
cv2.waitKey(0)              #顾名思义等待键盘输入，单位为毫秒，即等待指定的毫秒数看是否有键盘输入，
                            #若在等待时间内按下任意键则返回按键的ASCII码，程序继续运行。若没有按下任何键，超时后返回-1。
                            #参数为0表示无限等待。不调用waitKey的话，窗口会一闪而逝，看不到显示的图片。
cv2.destroyAllWindows()  #cv2.destroyAllWindow()销毁所有窗口
                            #cv2.destroyWindow(wname)销毁指定窗口



"""
@Name: 1.直方图.py
@Auth: Huohuo
@Date: 2023/3/9-15:21
@Desc: 
@Ver : 
code_idea
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils import show
from get_his import Gray_histogram
from Msrcr import Msrcr

show = show()




# img_path = "../Retinex/data/img.png"
# img_path = "../Retinex/data/crop_0.png"
# img_path = "../Retinex/data/img.png"
# img_path = "../Retinex/data/gamma10.png"


img_path = "../Retinex/data2/3848.jpg"
# img_path = "../Retinex/data2/3842.jpg"
# img_path = "../Retinex/data2/3169.jpg"
# img_path = "../Retinex/data2/3168.jpg"


# img_path = r"D:\WorkDirectory\mywork\myself_OCR\MSRCR\light_src\4878.jpg"
savepath = r"D:\WorkDirectory\mywork\myself_OCR\data_badlight\img\4878.jpg"



img = cv2.imread(img_path)
# ------------------------------------------
#                   原图msrcr + gamma
# 这里的MSRCR算法就是加入gamma改进后的算法，在msrcr处理过程中就加好了gamma变化
# ------------------------------------------

msrcr_src = Msrcr(img, oppo=False)
# msrcr_src = Msrcr(img, oppo=True)       # 反转图像
src, src_ssr, src_msr, src_msrcr = msrcr_src.main()     # 获得灰度拉伸后的图片
show.color_pic_show(1, 4, src, src_ssr, src_msr, src_msrcr)   # show
# show.color_pic_show(1, 3, src, src_ssr, src_msr)   # show

cv2.imwrite(savepath, src_msr)

# cv2.imwrite("paper_result5/src_ssr_stretch.png", src_ssr)
# cv2.imwrite("paper_result5/src_msr_stretch.png", src_msr)

# cv2.imwrite("paper_result5/src_ssr_gamma.png", src_ssr)
# cv2.imwrite("paper_result5/src_msr_gamma.png", src_msr)

# show.color_pic_show(1, 4, 255-src, 255-src_ssr, 255-src_msr, 255-src_msrcr)   # show
# cv2.imwrite("../Retinex/data/src_msrcr.png", 255-src_msrcr)

# 直方图绘制
# src_gary = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# src_msrcr_gray = cv2.cvtColor(src_msrcr, cv2.COLOR_BGR2GRAY)
# show.gray_pic_show(1, 2, src_gary, src_msrcr_gray)

# show_retinex = Gray_histogram(src_gary, show=True)     # 实例化
# his_old, his_new, img_equal = show_retinex.get_his()
# his_old.save("../Retinex/data/his_old.png")

# ---------------------------------
#      这里的gamma变化只是用来测试效果
#      与源码无关
# ---------------------------------

def adjust_gamma(imgs, gamma=10.0):
    new_imgs = np.power(imgs/255, gamma)
    return new_imgs
# img_ga = adjust_gamma(img)

# cv2.imwrite("../Retinex/data/gamma10.png", img_ga*255)
# src_msrcr = adjust_gamma(src_msrcr)
# show.color_pic_show(1, 3, img, src_msrcr, img_ga)


# show.color_pic_show(1, 4, 255-src, 255-src_ssr, 255-src_msr, 255-src_msrcr)   # show













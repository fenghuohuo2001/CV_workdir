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
from img_to_gray_his import Gray_histogram
from Msrcr import Msrcr

show = show()


img_path = "../Retinex/data/img_gray.png"
img = cv2.imread(img_path)
# --------------展示原图--------------------
# show.color_pic_show(1, 1, img)   # show
# --------------灰度化--------------------
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# show.gray_pic_show(1, 1, img_gray)   # show
# --------------直方图--------------------
# show_his = Gray_histogram(255-img_gray, show=True)
# his_old, his_new, img_equal = show_his.get_his()
# --------------均衡化--------------------
# show.gray_pic_show(1, 2, img_gray, img_equal)   # show
# --------------直方图--------------------
# show_his = Gray_histogram(img_gray, show=True)
# his_old, his_new, img_equal = show_his.get_his()
# --------------均衡化--------------------
# show.gray_pic_show(1, 2, img_gray, img_equal)   # show
# --------------二值化--------------------
# gray_ret, gray_th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# equal_ret, equal_th = cv2.threshold(img_equal, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# show.gray_pic_show(1, 2, gray_th, equal_th)   # show
# # --------------灰度图msrcr--------------------没意义
# gray_th = cv2.cvtColor(gray_th, cv2.COLOR_GRAY2BGR)
# gray_th, msrcr_gray = Msrcr(gray_th, oppo=True)
# gray_ssr, gray_msr, gray_msrcr = msrcr_gray.main()
# show.gray_pic_show(1, 4, gray_th, gray_ssr, gray_msr, gray_msrcr)   # show
# # --------------均衡灰度图msrcr--------------------没意义
# equal_th = cv2.cvtColor(equal_th, cv2.COLOR_GRAY2BGR)
# msrcr_equal = Msrcr(equal_th, oppo=True)
# equal_th, equal_ssr, equal_msr, equal_msrcr = msrcr_equal.main()
# show.gray_pic_show(1, 4, equal_th, equal_ssr, equal_msr, equal_msrcr)   # show
# --------------原图msrcr--------------------
# msrcr_src = Msrcr(img, oppo=False)
msrcr_src = Msrcr(img, oppo=True)
src, src_ssr, src_msr, src_msrcr = msrcr_src.main()
show.color_pic_show(1, 4, 255-src, src_ssr, src_msr, src_msrcr)   # show
show_retinex = Gray_histogram(src_msrcr, show=True)
his_old, his_new, img_equal = show_retinex.get_his()
show.color_pic_show(1, 4, 255-src, 255-src_ssr, 255-src_msr, 255-src_msrcr)   # show


# src_msrcr_gray = cv2.cvtColor(src_msrcr, cv2.COLOR_BGR2GRAY)
# ssr_ret, ssr_th = cv2.threshold(np.hstack(src_ssr), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# msr_ret, msr_th = cv2.threshold(np.hstack(src_msr), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# msrcr_ret, msrcr_th = cv2.threshold(255-src_msrcr_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imshow("Retinex", np.hstack([ssr_th, msr_th, msrcr_th]))   # show
# cv2.imwrite("paper_total.jpg", np.hstack([src, ssr_th, msr_th, msrcr_th]))
# cv2.waitKey(0)
print(src_msrcr.shape)
# show.cv_show(0, src_msrcr=src_msrcr, msrcr_th=msrcr_th, opsrc_msrcr=255-src_msrcr)











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
import os
import time

from utils import show
from get_his import Gray_histogram
from Msrcr import Msrcr

time_all_s = time.time()

show = show()

# 完整数量数据集
# img_path = r"D:\WorkDirectory\mywork\myself_OCR\paper_used_data\train\img"
# savepath = r"D:\WorkDirectory\mywork\myself_OCR\paper_used_data\train\img_en"

img_path = r"D:\WorkDirectory\mywork\myself_OCR\paper_used_data\test\img"
savepath = r"D:\WorkDirectory\mywork\myself_OCR\paper_used_data\test\img_en"

# 测试代码正确性
# img_path = r"D:\WorkDirectory\mywork\myself_OCR\paper_used_data\train\test"
# savepath = r"D:\WorkDirectory\mywork\myself_OCR\paper_used_data\train\test_en"

for filename in os.listdir(img_path):
    time_start = time.time()
    print(img_path + '/' + filename)

    img = cv2.imread(img_path + '/' + filename)

    # --------------原图msrcr--------------------
    msrcr_src = Msrcr(img, oppo=False)
    src, src_ssr, src_msr, src_msrcr = msrcr_src.main()

    # def adjust_gamma(imgs, gamma=10.0):
    #     new_imgs = np.power(imgs/255, gamma)
    #     return new_imgs
    # img_ga = adjust_gamma(src_msr)
    cv2.imwrite(savepath+ "\\" + filename, src_msr)
    print("used time: ", time.time()-time_start)


print("total used time: ", time_all_s - time.time())










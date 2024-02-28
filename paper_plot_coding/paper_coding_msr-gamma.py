"""
@Name: paper_coding_msr-gamma.py
@Auth: Huohuo
@Date: 2023/4/24-9:21
@Desc: 
@Ver : 
code_idea
"""

"""
@Name: Msrcr.py
@Auth: Huohuo
@Date: 2023/3/9-16:02
@Desc: 
@Ver : 
code_idea
"""
import numpy as np
import cv2

class Msrcr():
    def __init__(self, img, oppo=True):
        self.img = img
        self.oppo = oppo

    def singleScaleRetinexProcess(self, img, sigma):
        temp = cv2.GaussianBlur(img, (0, 0), sigma)
        gaussian = np.where(temp == 0, 0.01, temp)
        retinex = np.log10(img + 0.01) - np.log10(gaussian)
        return retinex

    def multiScaleRetinexProcess(self, img, sigma_list):
        retinex = np.zeros_like(img * 1.0)
        for sigma in sigma_list:
            retinex = self.singleScaleRetinexProcess(img, sigma)
        retinex += retinex / len(sigma_list)
        return retinex

    def colorRestoration(self, img, alpha, beta):
        img_sum = np.sum(img, axis=2, keepdims=True)
        color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))
        return color_restoration

    def multiScaleRetinexWithColorRestorationProcess(self, img, sigma_list, G=192, b=-30, alpha=125, beta=56):
        # G=5, b=25, alpha=125, beta=46)
        # G=192, b=-30, alpha=125, beta=56 原本论文数据
        img = np.float64(img) + 1.0
        img_retinex = self.multiScaleRetinexProcess(img, sigma_list)
        img_color = self.colorRestoration(img, alpha, beta)
        img_msrcr = G * (img_retinex * img_color + b)
        return img_msrcr

    def simplestColorBalance(self, img, low_clip, high_clip):
        total = img.shape[0] * img.shape[1]
        for i in range(img.shape[2]):
            # counts是新列表unique元素在旧列表中第一次出现的位置，并以列表形式储存在s中
            unique, counts = np.unique(img[:, :, i], return_counts=True)    # 去除其中重复的元素
            current = 0
            for u, c in zip(unique, counts):
                if float(current) / total < low_clip:
                    low_val = u
                if float(current) / total < high_clip:
                    high_val = u
                current += c
            img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)
        return img

    # 转uint8 以及灰度拉伸
    def touint8(self, img, gamma=10):
        # print(img.shape[2])
        for i in range(img.shape[2]):
            img[:, :, i] = (img[:, :, i] - np.min(img[:, :, i])) / (np.max(img[:, :, i]) - np.min(img[:, :, i])) * 255
            img[:, :, i] = np.power(img[:, :, i]/255, gamma)*255
        #
        # img = np.power(img, gamma)*255

        img = np.uint8(np.minimum(np.maximum(img, 0), 255))     # np.clip
        return img

    # # 多尺度gamma
    # def touint8(self, img, gamma_list=[5, 8, 10]):
    #     # print(img.shape[2])
    #     board = np.zeros(img.shape)
    #     for i in range(img.shape[2]):
    #         img[:, :, i] = (img[:, :, i] - np.min(img[:, :, i])) / (np.max(img[:, :, i]) - np.min(img[:, :, i])) * 255
    #         for gamma in gamma_list:
    #             img[:, :, i] = np.power(img[:, :, i]/255, gamma)*255
    #             board[:, :, i] += img[:, :, i]/3
    #
    #     # img = np.uint8(np.minimum(np.maximum(img, 0), 255))     # np.clip
    #     board = np.uint8(np.minimum(np.maximum(board, 0), 255))     # np.clip
    #     return board

    def SSR(self, img, sigma=300):  # 300
        ssr = self.singleScaleRetinexProcess(img, sigma)
        ssr = self.touint8(ssr)
        return ssr

    def MSR(self, img, sigma_list=[15, 80, 250]):   # 15, 80, 250
        msr = self.multiScaleRetinexProcess(img, sigma_list)
        msr = self.touint8(msr)
        return msr

    def MSRCR(self, img, sigma_list=[15, 80, 250], low_clip=0.01, high_clip=0.99):
        msrcr = self.multiScaleRetinexWithColorRestorationProcess(img, sigma_list)
        msrcr = self.touint8(msrcr)
        msrcr = self.simplestColorBalance(msrcr, low_clip, high_clip)
        return msrcr

    def main(self):
        image = self.img
        if self.oppo:
            image = 255-image
        # cv2.imwrite("result/oppo.jpg", image)

        ssr = self.SSR(image)
        msr = self.MSR(image)
        msrcr = self.MSRCR(image)
        # cv2.imshow("Retinex", np.hstack([image, ssr, msr, msrcr]))
        # cv2.imwrite("result/paper_total.jpg", np.hstack([image, ssr, msr, msrcr]))
        # cv2.imwrite("result/paper.jpg", msrcr)
        # cv2.waitKey(0)
        return image, ssr, msr, msrcr

import os
import time


# 完整数量数据集
img_path = r"D:\WorkDirectory\mywork\myself_OCR\paper_used_data\train\img"
savepath = r"D:\WorkDirectory\mywork\myself_OCR\paper_used_data\train\img_en"

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

    cv2.imwrite(savepath+ "\\" + filename, src_msr)
    print("used time: ", time.time()-time_start)

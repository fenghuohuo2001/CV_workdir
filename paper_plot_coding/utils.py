"""
@Name: utils.py
@Auth: Huohuo
@Date: 2023/3/9-15:31
@Desc: 
@Ver : 
code_idea
"""
import cv2
from matplotlib import pyplot as plt


class show():
    def __init__(self):
        pass
    # 灰度图展示
    def gray_pic_show(self, horizontal_number_of_img, vertical_number_of_img, *args):
        plt.figure(dpi=200, figsize=(8, 4))
        for i in range(len(args)):
            plt.subplot(horizontal_number_of_img, vertical_number_of_img, i + 1)
            plt.imshow(args[i], cmap="gray")  # 如果是展示灰度图，用这个
            plt.xticks([]), plt.yticks([])
        plt.show()
        return 0

    # 彩色图展示
    def color_pic_show(self, horizontal_number_of_img, vertical_number_of_img, *args):
        # plt.figure(dpi=1080 ,figsize=(1080, 1920))
        plt.figure()
        for i in range(len(args)):
            plt.subplot(horizontal_number_of_img, vertical_number_of_img, i + 1)
            plt.imshow(args[i][:, :, ::-1])  # 如果是cv.imread读取，用这个
            plt.xticks([]), plt.yticks([])
        plt.show()
        return 0

    # 单张清晰图展示
    def cv_show(self, show_time_use_s, **kwargs):
        for k, v in kwargs.items():
            cv2.imshow("{}".format(k), v)
        cv2.waitKey(show_time_use_s // 1000)
        cv2.destroyAllWindows()
        return 0

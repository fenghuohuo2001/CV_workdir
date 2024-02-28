"""
@Name: pic_show.py
@Auth: Huohuo
@Date: 2023/2/16-15:12
@Desc: 
@Ver : 
code_idea
"""
import matplotlib.pyplot as plt
import cv2


# 注意需要转换格式
def pic_show(horizontal_number_of_img, vertical_number_of_img, *args):
    plt.figure()

    for i in range(len(args)):
        plt.subplot(horizontal_number_of_img, vertical_number_of_img, i+1)
        plt.imshow(args[i][:, :, ::-1])         # 如果是cv.imread读取3通道彩色图，用这个
        plt.imshow(args[i], cmap="gray")            # 如果是展示灰度图，用这个
        # plt.imshow(args[i])                   # 如果是Image.open读取，用这个

    plt.show()
    return 0

def plt_show(horizontal_number_of_img, vertical_number_of_img, *args):
    plt.figure()

    for i in range(len(args)):
        plt.subplot(horizontal_number_of_img, vertical_number_of_img, i+1)
        plt.plot(args[i])         # 如果是cv.imread读取3通道彩色图，用这个
    plt.show()
    return 0

# 单张图片cvimshow展示，清晰度更高
def cv_show(show_time_use_s, **kwargs):
    for k, v in kwargs.items():
        cv2.imshow("{}".format(k), v)
    cv2.waitKey(show_time_use_s//1000)
    cv2.destroyAllWindows()
    return 0
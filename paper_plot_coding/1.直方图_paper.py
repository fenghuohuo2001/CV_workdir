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
import matplotlib.pyplot as plt

img_path = r"D:\WorkDirectory\cv_workdir\paper_plot_coding\paper_plot\img.png"
# 读取图像
import cv2
import numpy as np
import matplotlib.pyplot as plt
import collections

class GrayHistogram:
    def __init__(self, img, show=False):
        self.img = img
        self.show = show

    # 计算灰度图的直方图
    def draw_hist(self, gray):
        hist_new = []
        hist_result = []
        hist_key = []
        gray1 = list(gray.ravel())  # gray.ravel()将多维数组转换为一维数组
        obj = dict(collections.Counter(gray1))  # collections.Counter(gray1)统计出现频率，并定义为字典
        obj = sorted(obj.items(), key=lambda item: item[0])
        for each in obj:
            hist1 = []  # 注意！！！ 这里将hist1清空了
            key = list(each)[0]  # 灰度级
            each = list(each)[1]  # 出现频次
            hist_key.append(key)  # 出现过的灰度级 从小到大排序的数组
            hist1.append(each)  # 灰度级出现频次 按出现过的灰度级 （即 hist_key）对应排序
            hist_new.append(hist1)  # 调出来 避免清空

        # 检查从0-255每个通道是否都有个数，没有的话添加并将值设为0
        for i in range(0, 256):
            if i in hist_key:
                num = hist_key.index(i)
                hist_result.append(hist_new[num])
            else:
                hist_result.append([0])
        hist_result = np.array(hist_result)

        return hist_result

    def get_hist(self):
        # 直方图转化
        hist_old = self.draw_hist(self.img)

        if self.show:
            plt.figure(figsize=(6, 3), dpi=200)
            plt.subplot(1, 2, 1)
            # 并绘制直方图
            plt.plot(hist_old)
            plt.xlabel('Pixel Value')
            plt.ylabel('Count')
            plt.title('Original Histogram')
            # 添加箭头和坐标轴说明
            plt.annotate('Count', xy=(255, 0), xycoords='data', xytext=(255, np.max(hist_old) * 0.5),
                         arrowprops=dict(arrowstyle='->'), fontsize=10)
            plt.annotate('Pixel Value', xy=(0, np.max(hist_old)), xycoords='data', xytext=(30, np.max(hist_old) * 1.1),
                         arrowprops=dict(arrowstyle='->'), fontsize=10)

            plt.subplot(1, 2, 2)
            plt.imshow(self.img, cmap='gray')
            plt.xlabel('Columns')
            plt.ylabel('Rows')
            plt.title('Original Image')

            plt.tight_layout()
            plt.show()

        return hist_old


# 读取图像
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# 创建灰度直方图对象
histogram = GrayHistogram(img, show=True)

# 获取灰度直方图
hist = histogram.get_hist()












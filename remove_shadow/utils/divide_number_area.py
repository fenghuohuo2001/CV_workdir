"""
@Name: divide_number_area.py
@Auth: Huohuo
@Date: 2023/7/6-8:39
@Desc: 
@Ver : 
code_idea
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------
# 将得到的二值化图沿水平方向进行投影，得到需要分割的区间坐标，分割成为不同的图片
# 尝试过后，发现想要从投影图中分离出区域不好确定区域边界，通过固定阈值分割的方法用不了
# 平滑之后也无法确定边界
# ----------------------------------------------------------------
def area_his(img_thresh, img_src):
    # 将二值化图沿水平方向投影
    img_his = np.sum(img_thresh, axis=0)

    # 绘制投影直方图
    plt.plot(img_his)
    plt.xlabel('Column')
    plt.ylabel('Projection')
    plt.title('Horizontal Projection Histogram')
    plt.savefig("../result/get_area/get_area/Histogram.png")
    plt.show()
    plt.close()

    # ----------------------------------------------
    #           滑动窗口法对直方图进行平滑
    #    平滑是肯定要做的，否则部分噪声会影响区域的判断
    # ----------------------------------------------
    delta = 100     # 窗口宽度
    windows_num = len(img_his) // delta
    for i in range(windows_num):
        img_his[i * delta: (i + 1) * delta] = np.sum(img_his[i * delta: (i + 1) * delta]) / delta  # 滑动窗口取平均

    # 末尾余量需要单独处理
    img_his[windows_num * delta: len(img_his)-1] = np.sum(img_his[windows_num * delta: len(img_his)-1]) / delta

    # 绘制投影直方图
    plt.plot(img_his)
    plt.xlabel('Column')
    plt.ylabel('Projection')
    plt.title('Horizontal Projection Histogram smooth')
    plt.savefig("../result/get_area/get_area/Histogram_smooth.png")
    plt.show()
    plt.close()

# ----------------------------------------------------------------
# 按照 膨胀 -> 获取轮廓 -> 获取原图对应区域的方法得到需要处理的区域
# ----------------------------------------------------------------
def area_contour(img_thresh, img_src):
    # 膨胀
    kernel_size_d = 151
    kernel_d = np.ones((kernel_size_d, kernel_size_d), np.uint8)
    img_dilate = cv2.dilate(img_thresh, kernel_d, iterations=1)
    cv2.imwrite("../result/get_area/get_area/twice_dilate.jpg", img_dilate)

    # 查找轮廓
    contours, _ = cv2.findContours(img_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cut_areas = []
    # 分割出每个轮廓区域
    for i, contour in enumerate(contours):
        # 计算轮廓边界框
        x, y, w, h = cv2.boundingRect(contour)

        # 根据边界框从原图中分割出对应的区域
        cut_area = img_src[y:y+h, x:x+w]

        # 保存轮廓区域为单独的图像文件
        cv2.imwrite(f'../result/get_area/get_area/contour_{i}.jpg', cut_area)

        # 将分割结果依次存入list中
        cut_areas.append(cut_area)

    return cut_areas

# --------------------------------------------
#         图像中存在斜条纹，将图像旋转45°进行去除
# --------------------------------------------
def rotate(image, angle = 45):
    # 获取图像尺寸
    height, width = image.shape[:2]

    # 计算旋转中心点
    center = (width // 2, height // 2)

    # 计算旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 执行图像旋转
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_image


# --------------------------------------------
#         傅里叶变化后彩色图一维高斯滤波
# filter_range 滤波器作用区域与中心点的距离（中心点保存主要特征信息）
# --------------------------------------------
def gas_filter1d_color(img, i, filter_range = 100 ,window_size = 15, window_h = 8, sigma=3):
    fft = np.fft.fft2(img, axes=(0, 1))
    fft = np.fft.fftshift(fft, axes=(0, 1))

    magnitude_spectrum = 20 * np.log(np.abs(fft))
    cv2.imwrite(f"../result/get_area/get_area_fft/Magnitude Spectrum before gauss1d color{i}.jpg",
                magnitude_spectrum.astype(np.uint8))

    # 构造高斯核
    h, w = img.shape[0], img.shape[1]
    centy1 = int(h / 2)
    gauss_map = fft.copy()
    # -------------------伪一维高斯滤波--------------------
    gas_kernel = np.zeros((window_h*2+1, window_size))
    # 构造高斯卷积核
    for x in range(window_h*2+1):
        for y in range(window_size):
            gas_kernel[x, y] = np.exp(-((x - window_h) ** 2 + (y - window_size//2) ** 2) / (2 * sigma ** 2))

    for c in range(3):
        for x in range(w-window_size):
            if x < int(w/2)-filter_range or x > int(w/2)+filter_range:
                # print(x)
                # 获取当前像素位置周围窗口内的像素值
                window = fft[centy1-window_h:centy1+window_h+1, max(0, x):min(w, x + window_size), c]
                # print(window)
                # 对窗口内的像素值进行高斯计算，并将结果赋值给高斯滤波后的数组
                gauss_map[centy1-window_h:centy1+window_h+1, max(0, x):min(w, x + window_size), c] = window * gas_kernel


    magnitude_spectrum_afterfilter = 20 * np.log(np.abs(gauss_map))
    cv2.imwrite(f"../result/get_area/get_area_fft/Magnitude Spectrum gauss1d color{i}.jpg", magnitude_spectrum_afterfilter.astype(np.uint8))

    blur_img = np.fft.ifft2(gauss_map, axes=(0, 1))
    result = np.abs(blur_img)


    return result.astype(np.uint8)

# --------------------------------------------
#         傅里叶变化后彩色图一维中值滤波
# filter_range 滤波器作用区域与中心点的距离（中心点保存主要特征信息）
# --------------------------------------------
def median_filter1d_color(img, i, filter_range = 100 ,window_size = 15, window_h = 8):
    fft = np.fft.fft2(img, axes=(0, 1))
    fft = np.fft.fftshift(fft, axes=(0, 1))

    magnitude_spectrum = 20 * np.log(np.abs(fft))
    cv2.imwrite(f"../result/get_area/get_area_fft/Magnitude Spectrum before gauss1d color{i}.jpg",
                magnitude_spectrum.astype(np.uint8))

    # 构造高斯核
    h, w = img.shape[0], img.shape[1]
    centy1 = int(h / 2)
    median_map = fft.copy()
    # -------------------伪一维中值滤波-------------------
    for c in range(3):
        for x in range(w):
            if x < int(w/2)-filter_range or x > int(w/2)+filter_range:
                # print(x)
                # 获取当前像素位置周围窗口内的像素值
                window = fft[centy1-window_h:centy1+window_h+1, max(0, x-window_size//2):min(w, x + window_size//2+1), c]
                # print(window)
                # 对窗口内的像素值进行高斯计算，并将结果赋值给高斯滤波后的数组
                median_map[centy1-window_h:centy1+window_h+1, max(0, x-window_size//2):min(w, x + window_size//2+1), c] = np.mean(window)


    magnitude_spectrum_afterfilter = 20 * np.log(np.abs(median_map))
    cv2.imwrite(f"../result/get_area/get_area_fft/Magnitude Spectrum gauss1d color{i}.jpg", magnitude_spectrum_afterfilter.astype(np.uint8))

    blur_img = np.fft.ifft2(median_map, axes=(0, 1))
    result = np.abs(blur_img)


    return result.astype(np.uint8)

# ------------------------------------
#      对待处理区域再次进行fft+滤波
# ------------------------------------
def fft_gauss(cut_areas):
    for i, cut_area in enumerate(cut_areas):
        print(f"-----------------{i}------------------")
        print(cut_area.shape[1]//64, cut_area.shape[1]//32)
        print(cut_area.shape)
        # result = gas_filter1d_color(
        #     cut_area,
        #     i,
        #     filter_range=0,
        #     window_size=cut_area.shape[1]//64,
        #     window_h=cut_area.shape[1]//32,
        #     sigma=300
        # )
        # cut_area = rotate(cut_area)

        result = median_filter1d_color(
            cut_area,
            i,
            # filter_range=cut_area.shape[1]//128,
            filter_range=30,
            window_size=15,
            window_h=8,
        )

        # result = gas_filter1d_color(cut_area, i, cut_area.shape[1]//8, 0, cut_area.shape[1]//32, 300)
        cv2.imwrite(f"../result/get_area/get_area_fft/result{i}.jpg", result)



if __name__ == "__main__":
    import time

    time_start = time.time()

    # 二值化图
    img_thresh_path = "../result/get_area/divide22part/split.jpg"
    img = cv2.imread(img_thresh_path, 0)        # 以灰度图的形式读入
    # 原图(fft+gauss处理后的)
    img_src_path = "../result/fft/gauss1d_color.jpg"
    img_src = cv2.imread(img_src_path)          # 以彩色图的形式读入

    # area_his(img, img_src)        # 直方图分割法（fail）
    cut_areas = area_contour(img, img_src)      # 轮廓分割法 返回待处理区域

    # 对待处理区域再次进行fft+滤波
    fft_gauss(cut_areas)

    # 统计用时
    print("use time is :", time.time()-time_start)

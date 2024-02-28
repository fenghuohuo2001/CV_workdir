"""
@Name: 2.斜拉索分割(单张).py
@Auth: Huohuo
@Date: 2023/2/16-14:19
@Desc: 
@Ver :

代码思路：
1.读取图像，重置分辨率
2.灰度化显示，去噪，均衡化
3.二值化阈值截取目标区域 用hsv分离白色区域
4.目标区域重新二值化
"""
import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import collections

# 彩色图像展示
def color_pic_show(horizontal_number_of_img, vertical_number_of_img, *args):
    # plt.figure(dpi=1080 ,figsize=(1080, 1920))
    plt.figure()
    for i in range(len(args)):
        plt.subplot(horizontal_number_of_img, vertical_number_of_img, i+1)
        plt.imshow(args[i][:, :, ::-1])     # 如果是cv.imread读取，用这个
    plt.show()
    return 0
# 灰度图展示
def gray_pic_show(horizontal_number_of_img, vertical_number_of_img, *args):
    plt.figure(dpi=200, figsize=(8, 4))
    for i in range(len(args)):
        plt.subplot(horizontal_number_of_img, vertical_number_of_img, i+1)
        plt.imshow(args[i], cmap="gray")            # 如果是展示灰度图，用这个
    plt.show()
    return 0
# plt绘图展示
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

# 图像膨胀腐蚀去毛刺
def open(img, kernel_size, iterations):
    img_dilate = cv2.dilate(img, (kernel_size, kernel_size), iterations=iterations)  # 膨胀
    img_erode = cv2.erode(img_dilate, (kernel_size, kernel_size), iterations=iterations)  # 腐蚀
    return img_erode

# 图像均衡化
# 计算灰度图的直方图
def clc_histogram(gray):
    hist_new = []
    num = []
    hist_result = []
    hist_key = []
    gray1 = list(gray.ravel())      # gray.ravel()将多维数组转换为一维数组
    obj = dict(collections.Counter(gray1))      # collections.Counter(gray1)统计出现频率，并定义为字典
    obj = sorted(obj.items(), key=lambda item: item[0])
    # .items()，将字典中每对 key 和 value 组成一个元组，并把这些元组放在列表中返回。
    # 想对元素第二个字段排序，则key=lambda y: y[1]
    for each in obj:
        hist1 = []              # 注意！！！ 这里将hist1清空了
        key = list(each)[0]     # 灰度级
        each = list(each)[1]    # 出现频次
        hist_key.append(key)    # 出现过的灰度级 从小到大排序的数组
        hist1.append(each)      # 灰度级出现频次 按出现过的灰度级 （即 hist_key）对应排序
        hist_new.append(hist1)  # 调出来 避免清空

    # 检查从0-255每个通道是否都有个数，没有的话添加并将值设为0
    for i in range(0, 256):
        if i in hist_key:
            num = hist_key.index(i)
            # index() 方法检测字符串中是否包含子字符串i
            # 若存在i，则返回num 为i所在位置 eg：apple中 index(e)会返回4（num= 4）
            hist_result.append(hist_new[num])
        else:
            hist_result.append([0])
    # 最大灰度级没达到256时 补足0
    if len(hist_result) < 256:
        for i in range(0, 256 - len(hist_result)):
            hist_result.append([0])
    hist_result = np.array(hist_result)

    return hist_result

# 计算均衡化
def clc_result(hist_new, lut, gray):
    sum = 0
    Value_sum = []
    hist1 = []
    binValue = []

    for hist1 in hist_new:
        for j in hist1:
            binValue.append(j)
            sum += j
            Value_sum.append(sum)

    min_n = min(Value_sum)
    max_num = max(Value_sum)

    # 生成查找表
    for i, v in enumerate(lut):
        lut[i] = int(254.0 * Value_sum[i] / max_num + 0.5)
    # 计算
    result = lut[gray]
    return result

# 灰度直方图均衡化
def historgram_avg(img_gas):
    gray = cv2.cvtColor(img_gas, cv2.COLOR_BGR2GRAY)
    # 创建空的查找表
    lut = np.zeros(256, dtype=img_gas.dtype)
    # 直方图转化
    hist_src = clc_histogram(gray)
    # 并绘制直方图
    # plt.subplot(1, 2, 1)
    # plt.plot(hist_src)

    img_result = clc_result(hist_src, lut, gray)
    hist_new = clc_histogram(img_result)
    # 并绘制直方图
    # plt.subplot(1, 2, 2)
    # plt.plot(hist_new)
    # plt.show()

    # plt_show(1, 2, hist_src, hist_new)
    # cv_show(0, gray=gray, result=img_result)
    return img_result

def sobel(img):
    # 归一化
    img = np.float32(img)/255

    # 计算x和y方向的梯度
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

    # 计算合并后的幅值和方向（角度）
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)   # angleInDegrees=True是角度制 False是弧度制
    # cv_show(0, mag=mag)
    return mag

# 创建模板进行与操作，裁剪下不规则区域（管道区域）
def cut_area(img_gas, contours, color):
    img_write = np.zeros((img_gas.shape[0], img_gas.shape[1], 3), np.uint8)
    if color == 0:
        img_write.fill(255)
        cv2.fillPoly(img_write, contours, color=(0, 0, 0))
    elif color == 255:
        img_write.fill(0)
        cv2.fillPoly(img_write, contours, color=(255, 255, 255))
    # cv_show(0, img_write=img_write)
    # print("img_write.shape", img_write.shape)
    # print("img_gas.shape", img_gas.shape)
    # img_result = cv2.bitwise_and(img_gas, img_write)
    img_result = cv2.bitwise_and(img_write, img_gas)
    return img_result
# 轮廓检测方法
def contour_detection(img_gas):
    # canny
    img_canny = cv2.Canny(img_gas, 0, 200)
    img_sobel = sobel(img_gas)
    # cv_show(1000000, img_canny=img_canny, img_sobel=img_sobel)
    # gray_pic_show(1, 2, img_canny, img_sobel)

    # 二值化
    ret, img_thr = cv2.threshold(img_gas, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 自适应
    # ret, img_thr = cv2.threshold(img_gas, 175, 255, cv2.THRESH_BINARY)  # 手动
    # cv_show(1000, img_canny=img_canny, img_thr=img_thr)
    # gray_pic_show(1, 3, img_canny, img_sobel, img_thr)
    img_thr_reverse = 255 - img_thr
    # 查找轮廓
    contours, hierarchy = cv2.findContours(img_thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv2.findContours(img_sobel, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 创建空白画布
    img_write = np.zeros((img_gas.shape[0], img_gas.shape[1]), np.uint8)
    img_write.fill(255)

    # 绘制轮廓
    cv2.drawContours(img_gas, contours, -1, (0, 0, 255), 3)
    cv2.drawContours(img_write, contours, -1, (0, 0, 255), 3)

    # 翻转后轮廓绘制
    contours_reverse, hierarchy_reverse = cv2.findContours(img_thr_reverse, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_write_reverse = np.zeros((img_gas.shape[0], img_gas.shape[1]), np.uint8)
    img_write_reverse.fill(255)
    cv2.drawContours(img_write_reverse, contours_reverse, -1, (0, 0, 255), 3)

    # cv_show(0, img_canny=img_canny, img_thr=img_thr, img_gas=img_gas, img_write=img_write, img_write_reverse=img_write_reverse)
    # cv_show(0, img_gas=img_gas, img_thr=img_thr)
    gray_pic_show(1, 2, img_write, img_write_reverse)
    return 0

# HSV阈值检测方法
# 用来定位到管道之后的污渍检测效果还不错
def thresh_detection(img_gas, area, lower_black, upper_black):
    # 转换到hvs色彩空间
    img_hvs = cv2.cvtColor(img_gas, cv2.COLOR_BGR2HSV)
    # 在HSV色彩空间中，黑色的色相值为0，饱和度为0，亮度可以取任意值。
    # 因此，我们可以设定一个黑色的阈值范围，来检测黑色污渍。
    # 二值化
    img_mask = cv2.inRange(img_hvs, lower_black, upper_black)
    # 轮廓检测
    contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 保留面积>100的轮廓 先提取轮廓在计算像素面积是不计算目标边缘一圈像素，将其作为轮廓
    contours = [x for x in contours if cv2.contourArea(x) > area]
    # print("len(contours) is ", len(contours))

    # 绘制轮廓
    img_gas1 = img_gas.copy()
    cv2.drawContours(img_gas1, contours, -1, (0, 0, 255), 3)
    # 白板轮廓
    img_write = np.zeros((img_gas.shape[0], img_gas.shape[1]), np.uint8)
    img_write.fill(0)
    cv2.drawContours(img_write, contours, -1, (0, 0, 255), 3)

    img_result = cut_area(img_gas, contours, 255)
    # cv_show(0, img_result=img_result)

    # # 绘制矩形框
    # for contour in contours:
    #     rect = cv2.minAreaRect(contour)
    #     box = cv2.boxPoints(rect)
    #     box = np.int0(box)
    #     cv2.drawContours(img_gas, [box], 0, (0, 0, 255), 2)

    # cv_show(0, img_gas=img_gas, img_gas1=img_gas1, img_write=img_write)
    # cv_show(0, img_gas1=img_gas1, img_write=img_write, img_mask=img_mask)
    # cv_show(0, img_gas1=img_gas1)

    return img_result, contours



def Segmentation(img_path):
    # img_path = r"D:\WorkDirectory\school_project\dragline_detection\stay_cable_photo\J16\dirt\64388mm\left_back-2022-12-14-07-25-33-64388mm-65740mm.jpeg"

    img = cv2.imread(img_path)
    # print(img.shape)
    # 重置图像分辨率
    img_resize = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
    # print(img_resize.shape)
    # cv_show(0, img_resize=img_resize)
    # 图像去噪(高斯滤波效果最好)
    img_gas = cv2.GaussianBlur(img_resize, (5, 5), 0)     # 高斯
    # img_blur = cv2.blur(img_gray, (5, 5))               # 均值
    # img_mid = cv2.medianBlur(img_gray, 5)               # 中值
    # cv_show(1000, img_gas=img_gas, img_blur=img_blur, img_mid=img_mid)    # 对比效果
    # cv_show(0, img_gas=img_gas)
    # gray_pic_show(1, 3, img_gas, img_blur, img_mid)       # 作图
    img_open = open(img_gas, 5, 5)
    # cv_show(0, img_open=img_open, img_gas=img_gas)

    # 灰度直方图均衡化
    # img_avggray = historgram_avg(img_open)
    # 轮廓检测方法
    # contour_detection(img_avggray)

    # 在./utils/get_hsv.py中获得
    # 展示使用阈值
    # lower_black = np.array([10, 15, 80])
    # upper_black = np.array([25, 70, 180])
    # k-mean聚类阈值
    lower_black = np.array([10, 15, 35])
    upper_black = np.array([40, 55, 240])
    # HSV阈值检测方法 面积分割阈值为10000
    img_segmentation, contour_segmentation = thresh_detection(img_open, 10000, lower_black, upper_black)

    cv2.imwrite("src.jpg", img_resize)
    cv2.imwrite("cut_area.jpg", img_segmentation)
    # cv_show(0, img_gas=img_gas, img_result=img_segmentation)
    return img_segmentation, img_resize

def cut_main(img_segmentation, img):
    # img_segmentation已经经过去噪 & 开运算 & 重置分辨率 操作
    # 二值化之前需要进行灰度均衡化 这里会对空白区域均衡化，因此不要使用
    # segimg_avggray = historgram_avg(img_segmentation)
    # cv_show(0, segimg_avggray=segimg_avggray)
    # segimg_gray = cv2.cvtColor(img_segmentation, cv2.COLOR_BGR2GRAY)

    # 二值化轮廓检测方法
    # contour_detection(segimg_gray)

    # hsv阈值分割 污渍
    # 在./utils/get_hsv.py中获得
    lower_black = np.array([10, 50, 55])
    upper_black = np.array([18, 80, 70])
    # HSV阈值检测方法 面积分割阈值为10000
    img_dirt, contour_dirt = thresh_detection(img_segmentation, 400, lower_black, upper_black)
    if len(contour_dirt) == 1:
        print("A stain has been detected")
    elif len(contour_dirt) >= 1:
        print("{} stains have been detected".format(len(contour_dirt)))
    else:
        print("nothing has been detected")
    cv2.drawContours(img, contour_dirt, -1, (0, 0, 255), 3)

    # cv_show(0, img=img)
    cv2.imwrite("result_of_dirt_detection.jpg", img)

def main():
    # 读取数据集
    data_path = 'D:\WorkDirectory\school_project\dragline_detection\pre_data'
    class_name = ['dirt', 'normal', 'scratch']

    # 遍历目录下图片文件
    for filename in os.listdir(data_path):
        for img_path in os.listdir(data_path + '/' + filename):
            print(filename + '/' + img_path)
            # img = cv2.imread(data_path + '/' + filename + '/' + img_path)
            time_start = time.time()
            img_segmentation, img_resize = Segmentation(data_path + '/' + filename + '/' + img_path)
            # cv_show(0, img_segmentation=img_segmentation)
            color_pic_show(1, 2, img_resize, img_segmentation)
            cut_main(img_segmentation, img_resize)
            time_end = time.time()
            print("spend time : {}".format(time_end - time_start))

if __name__=="__main__":
    main()
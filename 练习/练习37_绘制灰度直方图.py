'''
们要做的就是将每个灰度级出现的次数统计出来，
这里我使用了一个循环遍历然后通过 count 来数出每个灰度级出现的次数。
之后读取出来以后就是类似于一个字典的形式，
前面的key是灰度级，
后面的value就是这个灰度级出现的次数。
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import collections


# 计算灰度图的直方图
def clczhifangtu(gray):
    hist_new = []
    num = []
    hist_result = []
    hist_key = []
    gray1 = list(gray.ravel())      # gray.ravel()将多维数组转换为一维数组
    obj = dict(collections.Counter(gray1))      # collections.Counter(gray1)统计出现频率，并定义为字典
    obj = sorted(obj.items(), key=lambda item: item[0])
    # .items()，将字典中每对 k
    # ey 和 value 组成一个元组，并把这些元组放在列表中返回。
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
def clcresult(hist_new, lut, gray):
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


def main():
    path = "35.jpg"
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("35_gray.jpg", gray)

    # 创建空的查找表
    lut = np.zeros(256, dtype=gray.dtype)
    # 直方图转化
    hist_new = clczhifangtu(gray)
    # 并绘制直方图
    plt.plot(hist_new)
    plt.show()

    result = clcresult(hist_new, lut, gray)
    hist_new = clczhifangtu(result)
    # 并绘制直方图
    plt.plot(hist_new)
    plt.show()
    cv2.imshow('yuantu', gray)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


main()

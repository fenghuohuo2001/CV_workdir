"""
@Name: get_number_area.py
@Auth: Huohuo
@Date: 2023/7/5-15:45
@Desc: 
@Ver : 
code_idea
"""
import cv2
import numpy as np

# ----------------------------------------------
#        利用图像的HSV特征提取白色区域
# ----------------------------------------------
def get_white_hsv(img):
    # 转hsv色彩空间
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义白色区域hsv范围[0, 0, =>100<=]  [255, =>80<=, 255]
    lower_white = np.array([0, 0, 150], dtype=np.uint8)
    upper_white = np.array([255, 50, 255], dtype=np.uint8)

    # 根据HSV范围创建掩膜
    mask = cv2.inRange(img_hsv, lower_white, upper_white)

    # 对原始图像应用掩膜，提取白色区域
    result = cv2.bitwise_and(img, img, mask=mask)

    '''
        后续可通过二值化，开运算，找轮廓的流程定位区域
    '''
    # cv2.imshow("result", result)

    return result

# ----------------------------------------------
#        利用图像的二值化阈值提取白色区域
# ----------------------------------------------
def get_white_gray(img, lower_thresh=145, upper_thresh=255):
    # 转灰度空间
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # 局部二值化
    block_size = 301  # 区域大小
    constant = 5  # 常数C用于调整阈值
    img_thresh1 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant)

    # 自适应均值阈值二值化
    img_thresh2 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 自适应高斯阈值二值化
    img_thresh3 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # 阈值二值化
    _, img_thresh4 = cv2.threshold(img_gray, lower_thresh, upper_thresh, cv2.THRESH_BINARY)

    result = img_thresh4
    # cv2.imshow("result", result)

    return result

# ----------------------------------------------
#              对彩色图像进行腐蚀+膨胀
# ----------------------------------------------
def dilated_color(img):
    # 定义膨胀卷积核
    kernel_size_d = 31
    kernel_size_e = 7
    kernel_d = np.ones((kernel_size_d, kernel_size_d), np.uint8)
    kernel_e = np.ones((kernel_size_e, kernel_size_e), np.uint8)

    # 分离色彩通道
    b, g, r = cv2.split(img)

    # 对每个通道进行膨胀
    # 对每个通道进行膨胀操作
    erode_b = cv2.erode(b, kernel_e, iterations=1)
    dilated_b = cv2.dilate(erode_b, kernel_d, iterations=1)
    erode_g = cv2.erode(g, kernel_e, iterations=1)
    dilated_g = cv2.dilate(erode_g, kernel_d, iterations=1)
    erode_r = cv2.erode(r, kernel_e, iterations=1)
    dilated_r = cv2.dilate(erode_r, kernel_d, iterations=1)

    # 合并膨胀后的通道
    dilated_image = cv2.merge((dilated_b, dilated_g, dilated_r))

    result = dilated_image

    # cv2.imshow("result", result)


    return result

# ----------------------------------------------
#              对彩色图像进行亮度拉伸
# ----------------------------------------------
def strech(image):
    # 分离彩色图像的通道
    b, g, r = cv2.split(image)

    # 对每个通道进行灰度拉伸
    min_value_b = np.min(b)
    max_value_b = np.max(b)
    stretched_b = cv2.convertScaleAbs(b, alpha=255.0 / (max_value_b - min_value_b),
                                      beta=-255.0 * min_value_b / (max_value_b - min_value_b))

    min_value_g = np.min(g)
    max_value_g = np.max(g)
    stretched_g = cv2.convertScaleAbs(g, alpha=255.0 / (max_value_g - min_value_g),
                                      beta=-255.0 * min_value_g / (max_value_g - min_value_g))

    min_value_r = np.min(r)
    max_value_r = np.max(r)
    stretched_r = cv2.convertScaleAbs(r, alpha=255.0 / (max_value_r - min_value_r),
                                      beta=-255.0 * min_value_r / (max_value_r - min_value_r))

    # 合并灰度拉伸后的通道
    stretched_image = cv2.merge((stretched_b, stretched_g, stretched_r))

    result = stretched_image

    # cv2.imshow("result", result)

    return result


# ----------------------------------------------
#             对彩色图像进行gamma变换
# ----------------------------------------------
def gamma(image, gamma=0.5):
    # 分离彩色图像的通道
    b, g, r = cv2.split(image)

    # 对每个通道进行Gamma变化
    adjusted_b = np.power(b / 255.0, gamma) * 255.0
    adjusted_g = np.power(g / 255.0, gamma) * 255.0
    adjusted_r = np.power(r / 255.0, gamma) * 255.0
    # 合并调整后的通道
    adjusted_image = cv2.merge((adjusted_b, adjusted_g, adjusted_r))
    result = adjusted_image

    # cv2.imshow("result", result)

    result = cv2.convertScaleAbs(result)

    return result

# ---------------------------------------------------
#   为了便于阅读以及与下面分布处理区分，
#   将对整张图片进行处理的流程写成函数
# ---------------------------------------------------
def deal_whole_pic(img):
    # 开运算
    result = dilated_color(img)
    cv2.imwrite("../result/get_area/dilated_color.jpg", result)

    # 亮度增强---strech
    # result = strech(result)
    # cv2.imwrite("../result/get_area/strech_color.jpg", result)
    # 亮度增强---gamma
    result = gamma(result)
    cv2.imwrite("../result/get_area/gamma_color.jpg", result)

    # hsv获取白色区域
    # result = get_white_hsv(result1)
    # cv2.imwrite("../result/get_area/get_white_hsv.jpg", result)

    # gray获取白色区域
    result = get_white_gray(result)
    cv2.imwrite("../result/get_area/get_white_gray.jpg", result)
    return result

# ---------------------------------------------------
#    由于上下两部分图片亮度差距过大，此处分为上下两部分进行处理
#    切分点人为确定 547/1440 ≈ 0.37
# ---------------------------------------------------
def split22img2deal(image, split_ratio = 0.37):
    # 获取图像尺寸和上下分割位置
    image_height = image.shape[0]
    split_line = int(image_height * split_ratio)

    # 分割图像为上下两部分
    upper_image = image[:split_line, :]
    cv2.imwrite("../result/get_area/divide22part/color_up.jpg", upper_image)
    lower_image = image[split_line:, :]
    cv2.imwrite("../result/get_area/divide22part/color_low.jpg", lower_image)

    # 对上半进行处理   上半阈值(low=145， up=255)
    up_result = dilated_color(upper_image)
    cv2.imwrite("../result/get_area/divide22part/dilated_color_up.jpg", up_result)

    # up_result = gamma(up_result, gamma=0.7)
    # cv2.imwrite("../result/get_area/divide22part/gamma_color_up.jpg", up_result)

    up_result = get_white_gray(up_result, 40, 255)
    cv2.imwrite("../result/get_area/divide22part/get_white_gray_up.jpg", up_result)


    # 对下半进行处理   下半阈值(low=145， up=255)
    low_result = dilated_color(lower_image)
    cv2.imwrite("../result/get_area/divide22part/dilated_color_low.jpg", low_result)

    low_result = gamma(low_result)
    cv2.imwrite("../result/get_area/divide22part/gamma_color_low.jpg", low_result)

    low_result = get_white_gray(low_result, 150, 255)
    cv2.imwrite("../result/get_area/divide22part/get_white_gray_low.jpg", low_result)


    # 垂直拼接上下两部分图像
    merged_image = np.vstack((up_result, low_result))
    result = merged_image

    cv2.imwrite("../result/get_area/divide22part/split.jpg", result)
    return result


if __name__ == '__main__':
    img_path = "../result/fft/gauss1d_color.jpg"
    img = cv2.imread(img_path)

    # 将整张图片进行处理
    # result = deal_whole_pic(img)

    # 将图片分为上下两部分进行处理
    result = split22img2deal(img)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

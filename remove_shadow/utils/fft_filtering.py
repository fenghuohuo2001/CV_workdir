"""
@Name: fft_filtering.py
@Auth: Huohuo
@Date: 2023/7/4-14:50
@Desc: 
@Ver : 
code_idea
"""
import cv2
import numpy as np
import time

# ---------------------------------------------
#   使用傅里叶变换进行频域滤波，以抑制阴影的频率成分
#    radius = 50   调整此参数以控制阴影抑制的效果
#   low_cutoff  低频截止频率
#   scale = 2   图像展示尺度
# ---------------------------------------------
def fft_fiter_count(img, scale = 1, low_cutoff = 1):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (int(img.shape[1]/scale), int(img.shape[0]/scale)))
    print(img.shape)
    # 利用众数获取 图像亮度恢复值
    counts = np.bincount(np.array(img.copy()).flatten())
    mode = np.argmax(counts)
    print(mode)
    # fft
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # 构建振幅谱
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    # 对频谱进行阴影抑制
    # 定义阴影的频率范围（不在中心附近）
    sum_by_row = np.sum(magnitude_spectrum, axis=1)
    center_row = np.argmax(sum_by_row)
    print("center_row", center_row)

    rows, cols = img.shape

    # 构建频带通滤波器
    mask = np.zeros((rows, cols), dtype=np.uint8)

    # 消除竖条纹
    # mask[0: center_row - low_cutoff, :] = 1
    # mask[center_row + low_cutoff:, :] = 1

    mask[0: center_row-1, :] = 1
    mask[center_row+2, :] = 1

    # 消除横条纹
    # mask[0: center_row - 1, :] = 1
    # mask[center_row + 1, :] = 1

    # 应用频带通滤波器
    fshift = fshift * mask

    # 逆傅里叶变换
    f_ishift = np.fft.ifftshift(fshift)
    restored_image = np.fft.ifft2(f_ishift)
    restored_image = np.abs(restored_image)

    # 显示结果
    img_src = cv2.resize(img, (cols//4, rows//4))
    cv2.imshow("Original Image", img_src)
    restored_image = cv2.resize(restored_image.astype(np.uint8)+int(mode), (cols//4, rows//4))
    cv2.imshow("Restored Image", restored_image.astype(np.uint8)+int(mode))
    cv2.imshow("Restored Image src", restored_image.astype(np.uint8))
    show_magnitude_spectrum = cv2.resize(magnitude_spectrum.astype(np.uint8), (cols//4, rows//4))
    cv2.imshow("Magnitude Spectrum", show_magnitude_spectrum)
    cv2.imwrite("../result/fft/Magnitude Spectrum.jpg", magnitude_spectrum)



    img_rgb = cv2.cvtColor(restored_image.astype(np.uint8)+int(mode), cv2.COLOR_GRAY2RGB)

    return img_rgb

def fft_fiter_mid(img, scale = 1, low_cutoff = 20):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # img = cv2.bitwise_not(img)
    img = cv2.resize(img, (int(img.shape[1]/scale), int(img.shape[0]/scale)))
    print(img.shape)

    # fft
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # 构建振幅谱
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    rows, cols = img.shape

    # 对频谱进行阴影抑制
    # 定义阴影的频率范围（在中心附近）
    center_row = rows // 2
    center_col = cols // 2

    # 构建频带通滤波器
    mask = np.ones((rows, cols), dtype=np.uint8)

    # 消除竖条纹
    mask[center_row - low_cutoff: center_row, 0: center_col-250] = 0
    mask[center_row - low_cutoff: center_row, center_col+250:] = 0
    mask[center_row: center_row + low_cutoff+1, 0: center_col-250] = 0
    mask[center_row: center_row + low_cutoff+1, center_col+250:] = 0
    # mask = 1

    # 应用频带通滤波器
    fshift_mask = fshift * mask

    # fshift - fshift_mask

    # 滤波后的频谱图
    magnitude_spectrum_afterfilter = 20 * np.log(np.abs(fshift_mask))

    # 逆傅里叶变换
    f_ishift = np.fft.ifftshift(fshift_mask)
    restored_image = np.fft.ifft2(f_ishift)
    restored_image = np.abs(restored_image)

    # 显示结果
    # cv2.imshow("Original Image", img)
    cv2.imshow("Restored Image", restored_image.astype(np.uint8)+60)
    cv2.imwrite("../result/fft/Restored Image fft.jpg", restored_image.astype(np.uint8)+60)
    cv2.imshow("Magnitude Spectrum", magnitude_spectrum.astype(np.uint8))
    cv2.imwrite("../result/fft/Magnitude Spectrum b now.jpg", magnitude_spectrum.astype(np.uint8))
    cv2.imwrite("../result/fft/Magnitude Spectrum a now.jpg", magnitude_spectrum_afterfilter.astype(np.uint8))

    img_rgb = cv2.cvtColor(restored_image.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    return img_rgb


# -------------------------------------------------------------------
#     要在Python中实现类似ImageJ软件的带通滤波器，
#     可以使用SciPy库中的信号处理模块（scipy.signal）和NumPy库来进行频域滤波。
# -------------------------------------------------------------------
from scipy import fftpack
from scipy.ndimage import gaussian_filter

def bandpass_filter(image, low_cutoff=20, high_cutoff=50):
    # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 进行傅里叶变换
    f = fftpack.fft2(gray_image)
    fshift = fftpack.fftshift(f)

    # 构建频率域掩膜
    rows, cols = gray_image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones_like(gray_image)
    mask[crow - high_cutoff:crow + high_cutoff, ccol - high_cutoff:ccol + high_cutoff] = 0
    mask[crow - low_cutoff:crow + low_cutoff, ccol - low_cutoff:ccol + low_cutoff] = 1

    # 应用频率域滤波器
    fshift_filtered = fshift * mask

    # 进行逆傅里叶变换
    f_filtered = fftpack.ifftshift(fshift_filtered)
    image_filtered = np.real(fftpack.ifft2(f_filtered))

    # 对结果进行灰度拉伸
    image_filtered_stretched = cv2.normalize(image_filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # cv2.imshow("Restored Image", image_filtered_stretched)
    cv2.imwrite("../result/fft/Restored Image bandpass.jpg", image_filtered_stretched)

    return image_filtered_stretched

# ------------------------------------------------------------
# 陷波滤波器（Notch Filter）是一种用于去除特定频率噪声或干扰信号的滤波器。
# 它的工作原理是在频率响应中创建一个“陷阱”或“洞”，以抑制或消除特定频率的信号成分。
# center_freq = 20  # 设置为竖直条纹的频率
# bandwidth = 10  # 设置带宽
# ------------------------------------------------------------
def notch_filter(image, center_freq=20, bandwidth=10):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 获取图像的傅里叶变换
    f = fftpack.fftshift(fftpack.fft2(image))

    # 构建频域的网格坐标
    rows, cols = image.shape
    r, c = np.mgrid[:rows, :cols]
    r -= rows // 2
    c -= cols // 2

    # 计算距离中心频率的距离
    d = np.sqrt(r**2 + c**2)

    # 创建带通滤波器
    mask = np.logical_and(d >= center_freq - bandwidth / 2, d <= center_freq + bandwidth / 2)

    # 应用滤波器
    f[mask] = 0

    # 滤波后的频谱图
    magnitude_spectrum_afterfilter = 20 * np.log(np.abs(f + 0.001))

    # 执行逆傅里叶变换
    filtered_image = np.abs(fftpack.ifft2(fftpack.ifftshift(f)))

    cv2.imwrite("../result/fft/Restored Image notch_filter.jpg", filtered_image)
    cv2.imwrite("../result/fft/magnitude_spectrum_afterfilter notch_filter.jpg", magnitude_spectrum_afterfilter)


    return filtered_image

# --------------------------------------------
#         傅里叶变化后定点多核高斯滤波
# https://blog.csdn.net/xiaoxifei/article/details/103901675
# --------------------------------------------
def gas_filter2d(img, sigma=300,centX=3300, centY=55):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fft = np.fft.fft2(img)
    fft = np.fft.fftshift(fft)

    # 构造高斯核
    w, h = img.shape

    # 高斯核1
    gas_map1 = np.zeros((w, h))
    centx1 = centX
    centy1 = int(w/2)

    for i in range(w):
        for j in range(h):
            dis = np.sqrt((i-centy1)**2 + (j-centx1)**2)
            gas_map1[i, j] = 1.0-np.exp(-0.5 * dis/sigma)

    # 高斯核2
    gas_map2 = np.zeros((w, h))
    centx2 = h-centx1
    centy2 = int(w / 2)

    for i in range(w):
        for j in range(h):
            dis = np.sqrt((i - centy2) ** 2 + (j - centx2)**2)
            gas_map2[i, j] = 1.0 - np.exp(-0.5 * dis / sigma)

    # # 高斯核3
    # gas_map3 = np.zeros((w, h))
    # centx3 = int(h/2)
    # centy3 = centY
    #
    # for i in range(w):
    #     for j in range(h):
    #         dis = np.sqrt((i - centy3) ** 2 + (j - centx3)**2)
    #         gas_map3[i, j] = 1.0 - np.exp(-0.5 * dis / sigma)
    #
    # # 高斯核4
    # gas_map4 = np.zeros((w, h))
    # centx4 = int(h/2)
    # centy4 = w-centy3
    #
    # for i in range(w):
    #     for j in range(h):
    #         dis = np.sqrt((i - centy4) ** 2 + (j - centx4)**2)
    #         gas_map4[i, j] = 1.0 - np.exp(-0.5 * dis / sigma)

    # 滤波
    # blur_fft = fft*gas_map1*gas_map2*gas_map3*gas_map4
    # blur_fft = fft*gas_map3*gas_map4
    blur_fft = fft*gas_map1*gas_map2

    magnitude_spectrum_afterfilter = 20 * np.log(np.abs(blur_fft))
    cv2.imwrite("../result/fft/Magnitude Spectrum gauss.jpg", magnitude_spectrum_afterfilter.astype(np.uint8))



    blur_img = np.fft.ifft2(blur_fft)
    result = np.abs(blur_img)

    return result.astype(np.uint8)

# --------------------------------------------
#         傅里叶变化后一维高斯滤波
# https://blog.csdn.net/xiaoxifei/article/details/103901675
# --------------------------------------------
def gas_filter1d(img, sigma=300,centX=3300, centY=55):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fft = np.fft.fft2(img)
    fft = np.fft.fftshift(fft)

    # 构造高斯核
    h, w = img.shape
    centy1 = int(h / 2)
    gauss_map = fft.copy()
    window_size = 30
    # -------------------伪一维高斯滤波--------------------
    window_h = 5
    filter_range = 30
    for i in range(w):
        if i < int(w/2)-filter_range or i > int(w/2)+30:
            # 获取当前像素位置周围窗口内的像素值
            window = fft[centy1-window_h:centy1+window_h+1, max(0, i - window_size // 2):min(w, i + window_size // 2 + 1)]
            # print(window)
            # 对窗口内的像素值进行中值计算，并将结果赋值给中值滤波后的数组
            gauss_map[centy1-window_h:centy1+window_h+1, max(0, i - window_size // 2):min(w, i + window_size // 2 + 1)] = np.mean(window)


    magnitude_spectrum_afterfilter = 20 * np.log(np.abs(gauss_map))
    cv2.imwrite("../result/fft/Magnitude Spectrum gauss1d.jpg", magnitude_spectrum_afterfilter.astype(np.uint8))



    blur_img = np.fft.ifft2(gauss_map)
    result = np.abs(blur_img)

    # 对结果进行灰度拉伸
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return result.astype(np.uint8)

# --------------------------------------------
#         傅里叶变化后彩色图一维高斯滤波
# --------------------------------------------
def gas_filter1d_color(img, sigma=3):
    fft = np.fft.fft2(img, axes=(0, 1))
    fft = np.fft.fftshift(fft, axes=(0, 1))

    # 构造高斯核
    h, w = img.shape[0], img.shape[1]
    centy1 = int(h / 2)
    gauss_map = fft.copy()
    # -------------------伪一维高斯滤波--------------------
    window_size = 15
    window_h = 8    # 单向长度
    gas_kernel = np.zeros((window_h*2+1, window_size))
    # 构造高斯卷积核
    for x in range(window_h*2+1):
        for y in range(window_size):
            gas_kernel[x, y] = np.exp(-((x - window_h) ** 2 + (y - window_size//2) ** 2) / (2 * sigma ** 2))

    # 滤波器作用区域与中心点的距离（中心点保存主要特征信息）
    filter_range = 100
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
    cv2.imwrite("../result/fft/Magnitude Spectrum gauss1d color.jpg", magnitude_spectrum_afterfilter.astype(np.uint8))

    blur_img = np.fft.ifft2(gauss_map, axes=(0, 1))
    result = np.abs(blur_img)


    return result.astype(np.uint8)

# --------------------------------------------
#         傅里叶变化后彩色图一维均值滤波
# --------------------------------------------
def mean_filter1d_color(img):
    fft = np.fft.fft2(img, axes=(0, 1))
    fft = np.fft.fftshift(fft, axes=(0, 1))

    # 构造高斯核
    h, w = img.shape[0], img.shape[1]
    centy1 = int(h / 2)
    gauss_map = fft.copy()
    window_size = 30
    # -------------------伪一维均值滤波--------------------
    window_h = 5
    filter_range = 30
    for c in range(3):
        for i in range(w):
            if i < int(w/2)-filter_range or i > int(w/2)+filter_range:
                # 获取当前像素位置周围窗口内的像素值
                window = fft[centy1-window_h:centy1+window_h+1, max(0, i - window_size // 2):min(w, i + window_size // 2 + 1), c]
                # print(window)
                # 对窗口内的像素值进行均值计算，并将结果赋值给均值滤波后的数组
                gauss_map[centy1-window_h:centy1+window_h+1, max(0, i - window_size // 2):min(w, i + window_size // 2 + 1), c] = np.mean(window)


    magnitude_spectrum_afterfilter = 20 * np.log(np.abs(gauss_map))
    cv2.imwrite("../result/fft/Magnitude Spectrum mean1d color.jpg", magnitude_spectrum_afterfilter.astype(np.uint8))



    blur_img = np.fft.ifft2(gauss_map, axes=(0, 1))
    result = np.abs(blur_img)

    return result.astype(np.uint8)

# --------------------------------------------
#         傅里叶变化后中值滤波
# --------------------------------------------
def median_filter(img, sigma=300,centX=3300, centY=55):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fft = np.fft.fft2(img)
    fft = np.fft.fftshift(fft)

    h, w = img.shape

    # 构建中值滤波器
    mid_map = fft.copy()
    centx1 = centX
    centy1 = int(h / 2)
    window_size = 30

    # ------------------二维-------------------------
    # for i in range(w):
    #     for j in range(h):
    #         # 获取当前像素位置周围窗口内的像素值
    #         window = fft[max(0, i - window_size // 2):min(w, i + window_size // 2 + 1),
    #                  max(0, j - window_size // 2):min(h, j + window_size // 2 + 1)]
    #         # 对窗口内的像素值进行中值计算，并将结果赋值给中值滤波后的数组
    #         mid_map[i, j] = np.median(window)
    # ----------------------------------------------

    # ---------------------一维----------------------
    window_h = 20
    filter_range = 70
    for i in range(w):
        if i < 3700+filter_range or i > 4200-filter_range:
            # 获取当前像素位置周围窗口内的像素值
            window = fft[centy1-window_h:centy1+window_h+1, max(0, i - window_size // 2):min(w, i + window_size // 2 + 1)]
            # print(window)
            # 对窗口内的像素值进行中值计算，并将结果赋值给中值滤波后的数组
            mid_map[centy1-window_h:centy1+window_h+1, max(0, i - window_size // 2):min(w, i + window_size // 2 + 1)] = np.median(window)

    # ----------------------------------------------


    magnitude_spectrum_afterfilter = 20 * np.log(np.abs(mid_map))
    cv2.imwrite("../result/fft/Magnitude Spectrum mid.jpg", magnitude_spectrum_afterfilter.astype(np.uint8))



    blur_img = np.fft.ifft2(mid_map)
    result = np.abs(blur_img)

    # 对结果进行灰度拉伸
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return result.astype(np.uint8)

if __name__ == "__main__":
    time_start = time.time()
    # img_path = "../object.png"
    # img_path = "../img_blur.jpg"
    img_path = "../data/test.jpg"
    img = cv2.imread(img_path)

    # result = fft_fiter_mid(img)
    # result = bandpass_filter(img)
    # result = notch_filter(img)
    # result = gas_filter2d(img)
    result = gas_filter1d_color(img)
    cv2.imwrite("../result/fft/gauss1d_color.jpg", result)



    # result = mean_filter1d_color(img)
    # cv2.imwrite("../result/fft/mean_color.jpg", result)

    # result = median_filter(img)
    # cv2.imwrite("../result/fft/median.jpg", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("used time is :", time.time() - time_start)


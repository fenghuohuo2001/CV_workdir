"""
@Name: demo-remove_shadow.py
@Auth: Huohuo
@Date: 2023/7/4-16:57
@Desc: 
@Ver : 
code_idea
"""
import cv2
import numpy as np

# ---------------------------------------------
#   使用傅里叶变换进行频域滤波，以抑制阴影的频率成分
#    radius = 50   调整此参数以控制阴影抑制的效果
#   low_cutoff  低频截止频率
#   scale = 2   图像展示尺度
# ---------------------------------------------
def fft_fiter(img, scale = 1, low_cutoff = 10):
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

    rows, cols = img.shape

    # 构建频带通滤波器
    # mask = np.zeros((rows, cols), dtype=np.uint8)
    mask = np.ones((rows, cols), dtype=np.uint8)

    # 消除竖条纹
    # mask[20: center_row - low_cutoff, :] = 1
    # mask[center_row + low_cutoff:-20, :] = 1

    mask[center_row - low_cutoff:center_row, :] = 0
    mask[center_row:center_row + low_cutoff, :] = 0

    # 应用频带通滤波器
    fshift = fshift * mask

    # 逆傅里叶变换
    f_ishift = np.fft.ifftshift(fshift)
    restored_image = np.fft.ifft2(f_ishift)
    restored_image = np.abs(restored_image)

    # 显示结果
    # cv2.imshow("Original Image", img)
    # cv2.imshow("Restored Image", restored_image.astype(np.uint8)+int(mode))
    cv2.imshow("Restored Image", restored_image.astype(np.uint8))
    # cv2.imshow("Magnitude Spectrum", magnitude_spectrum.astype(np.uint8))

    img_rgb = cv2.cvtColor(restored_image.astype(np.uint8)+int(30), cv2.COLOR_GRAY2RGB)

    return img_rgb

# ---------------------------------------------
#                  灰度拉伸
# ---------------------------------------------
def strech(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 计算图像的最小灰度值和最大灰度值：
    min_value = np.min(gray_image)
    max_value = np.max(gray_image)

    # 灰度拉伸
    stretched_image = cv2.convertScaleAbs(gray_image, alpha=255 / (max_value - min_value),
                                          beta=-255 * min_value / (max_value - min_value))

    return stretched_image

# -------------------------------------------------
#                读入图片
# -------------------------------------------------
img_path = "./data/test.jpg"
img_gray = cv2.imread(img_path, 0)
cv2.imwrite("gray.jpg", img_gray)

img = cv2.imread(img_path)

img_blur = fft_fiter(img)
cv2.imwrite("result/fft/img_blur.jpg", img_blur)

result = strech(img_blur)
cv2.imwrite("result.jpg", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
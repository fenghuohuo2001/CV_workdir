"""
@Name: 1-改变色彩.py
@Auth: Huohuo
@Date: 2024/2/28-10:51
@Desc: 
@Ver : 
code_idea
"""
import os
import cv2
import numpy as np

# ---------------------------------------------
# 通过随机添加马赛克masaic来增强
# ---------------------------------------------
def masaic_process(img):
    print(1//10)
    pass

# ---------------------------------------------
# 通过腐蚀膨胀来图像增强
# 缺点：不好定义算子核大小，需要自适应
# ---------------------------------------------
def morphological_process(img):
    pass

# ---------------------------------------------
# 添加高斯模糊等
# ---------------------------------------------
def gaussian_process(img, kernel_size=5, sigma=5, show=True):
    img_gas = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    if show:
        cv2.imshow("img_gas", img_gas)
    return img_gas


# ---------------------------------------------
# 添加高斯噪声等
# ---------------------------------------------
def gaussian_noise_process(img, mean=0, stddev=100, show=True):
    # 生成高斯噪声
    noise = np.random.normal(mean, stddev, img.shape).astype('uint8')
    # 添加高斯噪声
    img_gas_noise = cv2.add(img, noise)
    if show:
        cv2.imshow("img_gas_noise", img_gas_noise)
    return img_gas_noise


# ---------------------------------------------
# 改变图像亮度
# 应该根据图像亮度分布改变
# ---------------------------------------------
def brightness_trans_process(img, brightness_gain=50, direction="up",show=True):
    # 将图像转换为浮点数类型
    image_float = img.astype(np.float32)
    # 项下转换
    if direction== "up":
        brightness_gain=brightness_gain
    elif direction == "down":
        brightness_gain=-brightness_gain

    # 对每个通道进行亮度变换
    img_light_trans = np.clip(image_float + brightness_gain, 0, 255).astype(np.uint8)

    if show:
        cv2.imshow("img_light_trans_"+direction, img_light_trans)
    return img_light_trans

def main():
    # image = cv2.imread("data/367_14.png")
    image = cv2.imread("data/377_3.png")
    image_h, image_w = image.shape[:2]
    print(image_h, image_w)
    image_left = image[:, 0:image_w//2]
    cv2.imshow("left", image_left)

    masaic_process(image_left)
    gaussian_process(image_left, 5, 19, True)
    gaussian_noise_process(image_left, 0, 100, True)
    brightness_trans_process(image_left, 50, "down", True)
    brightness_trans_process(image_left, 50, "up", True)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


    return 0


if __name__ == '__main__':
    main()

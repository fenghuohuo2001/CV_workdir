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
    img_h, img_w = img.shape[:2]
    img_gas = img.copy()
    img_left = img[:, 0:img_w//2]
    img_left_gas = cv2.GaussianBlur(img_left, (kernel_size, kernel_size), sigma)
    img_gas[:, 0:img_w//2] = img_left_gas
    if show:
        print("show gaussian result")
        cv2.imshow("img_gas", img_gas)
    return img_gas


# ---------------------------------------------
# 添加高斯噪声等
# ---------------------------------------------
def gaussian_noise_process(img, mean=0, stddev=100, show=True):
    img_h, img_w = img.shape[:2]
    img_gas_noise = img.copy()
    img_left = img[:, 0:img_w // 2]
    # 生成高斯噪声
    noise = np.random.normal(mean, stddev, img_left.shape).astype('uint8')
    # 添加高斯噪声
    img_left_gas_noise = cv2.add(img_left, noise)
    img_gas_noise[:, 0:img_w//2] = img_left_gas_noise
    if show:
        print("show gaussian noise result")
        cv2.imshow("img_gas_noise", img_gas_noise)
    return img_gas_noise


# ---------------------------------------------
# 改变图像亮度
# 应该根据图像亮度分布改变
# ---------------------------------------------
def brightness_trans_process(img, brightness_gain=50, direction="up",show=True):
    img_h, img_w = img.shape[:2]
    img_brightness_trans = img.copy()
    img_left = img[:, 0:img_w // 2]

    # 将图像转换为浮点数类型
    image_left_float = img_left.astype(np.float32)
    # 项下转换
    if direction == "up":
        brightness_gain=brightness_gain
    elif direction == "down":
        brightness_gain=-brightness_gain

    # 对每个通道进行亮度变换
    img_left_light_trans = np.clip(image_left_float + brightness_gain, 0, 255).astype(np.uint8)
    img_brightness_trans[:, 0:img_w // 2] = img_left_light_trans
    if show:
        cv2.imshow("img_light_trans_"+direction, img_brightness_trans)
    return img_brightness_trans

def main():
    folder_path = "data/"
    imwrite_path = "result/"

    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            image = cv2.imread(os.path.join(folder_path, filename))
            print(filename)
            # 获取图像文件名称
            img_index = filename.split(".")[0]

            # masaic_process(image_left)
            img_gas = gaussian_process(image, 5, 19, False)
            cv2.imwrite(imwrite_path + img_index + "_gas.png", img_gas)

            img_gas_noise = gaussian_noise_process(image, 0, 100, False)
            cv2.imwrite(imwrite_path + img_index + "_gas_noise.png", img_gas_noise)

            img_bright_down = brightness_trans_process(image, 50, "down", False)
            cv2.imwrite(imwrite_path + img_index + "_bright_down.png", img_bright_down)

            img_bright_up = brightness_trans_process(image, 50, "up", False)
            cv2.imwrite(imwrite_path + img_index + "_bright_up.png", img_bright_up)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return 0


if __name__ == '__main__':
    main()

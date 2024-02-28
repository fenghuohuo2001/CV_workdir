# -*- 练习 -*-
"""
功能：
作者：fenghuohuo
日期：2021年11月29日
"""
import numpy as np
import matplotlib.pyplot as plt


def zip_image_by_svd(origin_image, rate=0.8):
    # 显示原图像
    plt.figure(figsize=(12, 12))
    plt.title("Origin Image")
    plt.imshow(origin_image)
    plt.show()

    print("\n\n======开始压缩======")
    # 提前开辟结果存放空间
    result = np.zeros(origin_image.shape)

    # 对原图像进行SVD分解
    u_shape = 0
    s_shape = 0
    vT_shape = 0

    for chan in range(3):
        # 对该层进行SVD分解
        U, sigma, V = np.linalg.svd(origin_image[:, :, chan])
        n_sigmas = 0
        temp = 0

        # 计算达到保留率需要的奇异值数量
        while (temp / np.sum(sigma)) < rate:
            temp += sigma[n_sigmas]
            n_sigmas += 1

        # 构建奇异值矩阵
        S = np.zeros((n_sigmas, n_sigmas))

        for i in range(n_sigmas):
            S[i, i] = sigma[i]

        # 构建结果
        result[:, :, chan] = (U[:, 0:n_sigmas].dot(S)).dot(V[0:n_sigmas, :])
        u_shape = U[:, 0:n_sigmas].shape
        s_shape = S.shape
        vT_shape = V[0:n_sigmas, :].shape

    # 归一化到[0, 1]
    for i in range(3):
        MAX = np.max(result[:, :, i])
        MIN = np.min(result[:, :, i])
        result[:, :, i] = (result[:, :, i] - MIN) / (MAX - MIN)

    # 调整到[0, 255]
    result = np.round(result * 255).astype('int')

    # 显示压缩结果
    plt.figure(figsize=(12, 12))
    plt.imshow(result)
    plt.title("Result Image")
    plt.show()

    # 计算压缩率
    zip_rate = (origin_image.size - 3 * (
                u_shape[0] * u_shape[1] + s_shape[0] * s_shape[1] + vT_shape[0] * vT_shape[1])) / (origin_image.size)

    print("保留率：        ", rate)
    print("所用奇异值数量为：", n_sigmas)
    print("原图大小：       ", origin_image.shape)
    print("压缩后用到的矩阵大小：3 x ({} + {} + {})".format(u_shape, s_shape, vT_shape))
    print("压缩率为：       ", zip_rate)


# 定义主函数
def main():
    # 读入图像
    image_path = 'rabbit.png'
    origin_image = plt.imread(image_path)

    # 利用自定义SVD图像压缩模块对图像进行压缩
    zip_image_by_svd(origin_image, rate=0.5)


# 主函数调用
if __name__ == "__main__":
    main()

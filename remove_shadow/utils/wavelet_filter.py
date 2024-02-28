"""
@Name: wavelet_filter.py
@Auth: Huohuo
@Date: 2023/7/5-8:59
@Desc: 
@Ver : 
code_idea
"""
import cv2
import numpy as np
import pywt     # pip install PyWavelets

def wavelet_filter(img, scale=1):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (int(img.shape[1] / scale), int(img.shape[0] / scale)))

    # 执行小波变换
    wave_type = "sym8"
    coeffs = pywt.dwt2(img, wave_type)
    cA, (cH, cV, cD) = coeffs

    # 设置阈值来去除噪声
    threshold = 50
    # cA = pywt.threshold(cA, threshold, mode="soft")
    # cH = pywt.threshold(cH, threshold, mode="soft")
    cV = pywt.threshold(cV, threshold, mode="soft")
    # cD = pywt.threshold(cD, threshold, mode="soft")

    # 重构图像
    reconstructed_image = pywt.idwt2((cA, (cH, cV, cD)), wave_type)

    # 显示原始图像和去噪后的图像
    # cv2.imshow("Original Image", img)
    # cv2.imshow("Denoised Image", reconstructed_image.astype(np.uint8))
    cv2.imwrite("../result/wavelet/Denoised Image.jpg", reconstructed_image.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    img_path = "../data/test.jpg"
    img = cv2.imread(img_path)
    wavelet_filter(img)

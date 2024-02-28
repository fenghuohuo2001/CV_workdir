"""
@Name: get_hog.py
@Auth: Huohuo
@Date: 2023/2/17-15:48
@Desc: 
@Ver : 
code_idea
1. 检测窗口
2. 归一化
3. 计算梯度
4. 对每一个cell块进行梯度直方图规定权重投影
5. 对每一个重叠block块内的cell进行对比度归一化
6. 将所有block直方图向量组合成为大的HOG特征向量

"""
import os
import cv2
import numpy as np


# 单张图片cvimshow展示，清晰度更高
def cv_show(show_time_use_s, **kwargs):
    for k, v in kwargs.items():
        cv2.imshow("{}".format(k), v)
    cv2.waitKey(show_time_use_s)
    cv2.destroyAllWindows()
    return 0

# 计算hog特征
def calculate_hog(img_resize):
    # 归一化
    img_normalization = np.float32(img_resize)/255

    # 计算x和y方向的梯度
    gx = cv2.Sobel(img_normalization, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img_normalization, cv2.CV_32F, 0, 1, ksize=1)

    # 计算合并后的幅值和方向（角度）
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)   # angleInDegrees=True是角度制 False是弧度制
    cv_show(200, mag=mag)

    return 0

def main():
    # 读取数据集
    data_path = 'D:\WorkDirectory\school_project\dragline_detection\pre_data'
    class_name = ['dirt', 'normal', 'scratch']

    # 遍历目录下图片文件
    for filename in os.listdir(data_path):
        print(filename)
        for img_path in os.listdir(data_path + '/' + filename):
            print(img_path)
            img = cv2.imread(data_path + '/' + filename + '/' + img_path)
            img_resize = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
            # cv_show(0, img_resize=img_resize)
            calculate_hog(img_resize)


        print('the number of img is:', len(os.listdir(data_path + '/' + filename)))

if __name__ == "__main__":
    main()
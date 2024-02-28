"""
@Name: HOG_SVM_detection.py
@Auth: Huohuo
@Date: 2023/2/17-9:51
@Desc: 
@Ver : 
code_idea:
1. 读取数据集
    data_path = 'D:\WorkDirectory\school_project\dragline_detection\pre_data'
    pre_data/dirt
    pre_data/normal
    pre_data/scratch
2. 图像预处理
    1）gamma矫正
    2）cell归一化
3. 梯度计算
4. 权重投票
5. 对比归一化
6. 收集检测窗口HOG特征序列
7. SVM分类
"""
import os

import cv2


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
            img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))
            hog_svm_detection(img)
        print('the number of img is:', len(os.listdir(data_path + '/' + filename)))

def hog_svm_detection(img):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    # Detect people in the image
    (rects, weights) = hog.detectMultiScale(img,
                                            winStride=(2, 4),   # 窗口大小
                                            padding=(8, 8),
                                            scale=1.2,
                                            useMeanshiftGrouping=False)
    for (x, y, w, h) in rects:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("hog-detector", img)
    # cv2.imwrite("hog-detector.jpg", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()
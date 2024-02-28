"""
@Name: video_process.py
@Auth: Huohuo
@Date: 2023/6/14-11:08
@Desc: 
@Ver : 
code_idea
"""
import cv2
import numpy as np

# ------------------------------------
#               sobel算子
# ------------------------------------
def sobel(img, x_weight, y_weight):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)

    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)

    dst = cv2.addWeighted(absX, x_weight, absY, y_weight, 0)
    return dst

# ------------------------------------
#               形态学运算
# open = True:      开运算
# open = False:     闭运算
# ------------------------------------
def open_close(img, k_x=7, k_y=15, open=False):
    morph_kernel = np.ones((k_x, k_y), np.uint8)
    if open:
        img_erode = cv2.erode(img, morph_kernel, 1)
        img_dilate = cv2.dilate(img_erode, morph_kernel, 1)
        return img_dilate

    else:
        img_dilate = cv2.dilate(img, morph_kernel, 1)
        img_erode = cv2.erode(img_dilate, morph_kernel, 1)
        return img_erode

# ------------------------------------
#               segment
# ------------------------------------
def segment(img):
    # roi区域手动提取
    img_h, img_w, c = img.shape
    # print(img.shape)
    # img = img[5:-5, 100:190]

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化
    thresh, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # canny
    # img_blur = cv2.GaussianBlur(img_binary, (3, 3), 0)
    # img_blur = cv2.medianBlur(img_binary, 3, 0)
    img_blur = cv2.blur(img_binary, (3, 3))

    # open or close
    img_open = open_close(img_blur)
    cv2.imshow("img_open", img_open)

    # 边缘检测
    img_edge = cv2.Canny(img_open, 50, 150)
    # img_edge = sobel(img_open, 0, 1)


    return img, img_edge, img_binary



# 打开视频文件
video_path = 'data2/result.mp4'  # 视频文件路径
cap = cv2.VideoCapture(video_path)

# 检查视频文件是否成功打开
if not cap.isOpened():
    print("无法打开视频文件")
    exit()



# 读取并显示视频帧
while True:
    # 逐帧读取视频
    ret, frame = cap.read()

    # 如果成功读取帧
    if ret:

        frame, canny, cut = segment(frame)
        # 在帧上绘制帧率


        # 在窗口中显示帧
        cv2.imshow('Video Frame', frame)
        cv2.imshow('Video Canny', canny)
        cv2.imshow('Video Cut', cut)


        # 按下 'q' 键退出循环
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    else:
        # 视频帧读取完毕或发生错误时退出循环
        break


# 释放资源并关闭窗口
cap.release()
cv2.destroyAllWindows()

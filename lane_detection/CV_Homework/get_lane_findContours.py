"""
@Name: get_lane.py
@Auth: Huohuo
@Date: 2023/6/6-9:31
@Desc: 
@Ver : 
code_idea

data/Railway.mp4

"""
import cv2
import numpy as np
import time


# ------------------------------------
#           图像增强
# ------------------------------------
def img_enhance(img):
    img_result = img
    return img_result

# ------------------------------------
#           感兴趣区域提取
'''
height, width = edges.shape
    mask = np.zeros_like(edges)
    # 定义三角区域顶点
    region_of_interest_vertices = np.array([[(0, height), (width / 2, height / 2), (width, height)]], dtype=np.int32)
    cv2.fillPoly(mask, region_of_interest_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
'''
# ------------------------------------
def get_roi_area(img, h1, h2, w1, w2):
    img_h, img_w = img.shape
    '''
    取高： int(img_h/2):-5
    取宽： 220:390
    '''
    img = img[int(img_h/2):-5, 220:390]
    img = img[h1:h2, w1:w2]
    return img


# ------------------------------------
#           原图上绘制直线
#   lines : 每一条直线对应的（ρ，θ）的集合
#   img ： 需要绘制的图片
#   sigma： 绘制的直线长度
# ------------------------------------
def show_line(lines, img, sigma=50):
    slopes = []
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        # 斜率
        slope = b / a

        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + sigma * (-b))
        y1 = int(y0 + sigma * (a))
        x2 = int(x0 - sigma * (-b))
        y2 = int(y0 - sigma * (a))

        cv2.line(img, (x1 + 220, y1 - 5 + int(img.shape[0] / 2)), (x2 + 220, y2 - 5 + int(img.shape[0] / 2)), (0, 0, 255), 2)
        slopes.append(slope)

    return img, slopes

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
def open_close(img, k_x=3, k_y=3, open=True):
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
#       灰度化 + 边缘检测 + hough检测
# ------------------------------------
def detection_lane_hough(img):
    print(img.shape)
    img_h = img.shape[0]
    img_w = img.shape[1]

    # 灰度化
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    # img_blur = cv2.medianBlur(img_gray, 3, 0)
    img_blur = cv2.blur(img_gray, (3, 3))
    # 边缘检测
    img_edge = cv2.Canny(img_blur, 50, 150)
    # img_edge = sobel(img_edge, 1, 0)
    img_edge = open_close(img_edge, k_x=3, k_y=3, open=False)
    # img_edge = open_close(img_edge, k_x=3, k_y=3, open=True)
    # cv2.imshow("canny", img_canny)

    # 手动选取ROI区域    img = img[int(img_h/2):-5, 220:390]
    h_top = int(img_h/2)
    h_bottom = -5
    w_left = 220
    w_right = 390
    img_cut = img_edge[h_top:h_bottom, w_left:w_right]

    # hough检测 返回的是每一条直线对应的（ρ，θ）
    # ρ的精度为1， θ精度为1°，长度阈值200
    # lines = cv2.HoughLines(img_cut, 1, np.pi/180, 50)

    # 绘制检测结果
    # img, slopes = show_line(lines, img, sigma=50)

    contours, hierarchy = cv2.findContours(img_cut, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建空白图像，绘制近视曲线
    img_show = np.zeros_like(img_cut)
    # 设置近似精度，精度设置为周长的0.02
    epsilion = 0.01 * cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilion, True)
    print(approx)
    # 平移
    approx[:, :, 0] += 220
    approx[:, :, 1] += int(img_h/2)

    cv2.drawContours(img, [approx], -1, (255, 255, 255), 2)

    return img, img_edge, img_cut


# ------------------------------------
#           kalman滤波
# ------------------------------------
# class LaneTracker:
#     def __init__(self, n_lanes, proc_noise_):



# 打开视频文件
video_path = 'data/Railway.mp4'  # 视频文件路径
cap = cv2.VideoCapture(video_path)

# 检查视频文件是否成功打开
if not cap.isOpened():
    print("无法打开视频文件")
    exit()

# 获取视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)
print("视频帧率: {:.2f}".format(fps))

# 初始化计时器和帧计数器
start_time = time.time()
frame_count = 0

# 读取并显示视频帧
while True:
    # 逐帧读取视频
    ret, frame = cap.read()

    # 如果成功读取帧
    if ret:

        frame, canny, cut = detection_lane_hough(frame)
        # 在帧上绘制帧率
        cv2.putText(frame, "FPS: {:.2f}".format(frame_count / (time.time() - start_time)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 在窗口中显示帧
        cv2.imshow('Video Frame', frame)
        cv2.imshow('Video Canny', canny)
        cv2.imshow('Video Cut', cut)

        # 帧计数器加一
        frame_count += 1

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # 视频帧读取完毕或发生错误时退出循环
        break

# 计算实际处理后的帧率
end_time = time.time()
processed_fps = frame_count / (end_time - start_time)
print("实时帧率: {:.2f}".format(processed_fps))

# 释放资源并关闭窗口
cap.release()
cv2.destroyAllWindows()


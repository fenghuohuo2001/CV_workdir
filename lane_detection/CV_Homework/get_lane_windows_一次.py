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

from matplotlib import pyplot as plt


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
#             滑动窗口法
# ------------------------------------
def find_lane_windows(img_edge):
    img_h, img_w = img_edge.shape
    # 建立直方图，统计纵向像素点
    histogram = np.sum(img_edge, axis=0)
    # # 可视化
    # x = np.arange(histogram.shape[0])
    # # 绘制直方图
    # plt.bar(x, histogram)
    # # 添加标题和标签
    # plt.title('Histogram')
    # plt.xlabel('Bins')
    # plt.ylabel('Frequency')



    # 建立一张空图用于可视化
    img_out = np.dstack((img_edge, img_edge, img_edge))

    # 找到直方图中的中点和两个峰值点,注意直方图是一维的
    midpoint = np.int8(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # 滑动窗口数量
    windows_num = 3
    # 滑动窗口边框
    margin = 200
    # 中心窗口最小像素个数
    minpixs = 100
    # 通过滑动窗口个数确定窗口高度
    windows_height = np.int8(img_h // windows_num)

    # 识别图像中非0像素的位置
    nonzero = img_edge.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    # 存储左右车道线的像素索引
    leftx_current = leftx_base
    rightx_current = rightx_base

    # 创建空列表接收车道线像素索引
    left_lane_inds = []
    right_lane_inds = []

    # 逐窗口处理，从上到下遍历图像，计算每个窗口边界，确定窗口内的非零像素
    for window in range(windows_num):
        # 窗口边界坐标
        win_y_low = img_h - (window + 1) * windows_height
        win_y_high = img_h - window * windows_height
        win_x_left_low = leftx_current - margin
        win_x_left_high = leftx_current + margin
        win_x_right_low = rightx_current - margin
        win_x_right_high = rightx_current + margin

        # 绘制窗口
        cv2.rectangle(img_out, (win_x_left_low, win_y_low), (win_x_left_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(img_out, (win_x_right_low, win_y_low), (win_x_right_low, win_y_high), (0, 255, 0), 2)

        # 定义在窗口内的非0像素，用坐标点的形式输出
        good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                         (nonzero_x >= win_x_left_low) & (nonzero_x < win_x_left_high)).nonzero()[0]
        good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                          (nonzero_x >= win_x_right_low) & (nonzero_x < win_x_right_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # 若找到的左车道线像素个数大于minpixs，重新定位下一个窗口在左车道线像素平均位置
        if len(good_left_inds) > minpixs:
            leftx_current = np.int8(np.mean(nonzero_x[good_left_inds]))
        # 若找到的右车道线像素个数大于minpixs，重新定位下一个窗口在右车道线像素平均位置
        if len(good_right_inds) > minpixs:
            rightx_current = np.int8(np.mean(nonzero_x[good_right_inds]))

    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        pass
    leftx = nonzero_x[left_lane_inds]
    lefty = nonzero_y[left_lane_inds]
    rightx = nonzero_x[right_lane_inds]
    righty = nonzero_y[right_lane_inds]

    return leftx, lefty, rightx, righty, img_out


# ------------------------------------
#           滑动窗口车道线检测
# ------------------------------------
def detection_lane_windows(img):
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
    img_edge = sobel(img_edge, 1, 0)
    img_edge = open_close(img_edge, k_x=3, k_y=3, open=False)
    # img_edge = open_close(img_edge, k_x=3, k_y=3, open=True)
    # cv2.imshow("canny", img_canny)

    # 手动选取ROI区域    img = img[int(img_h/2):-5, 220:390]
    h_top = int(img_h/2)
    h_bottom = -5
    w_left = 220
    w_right = 390
    img_cut = img_edge[h_top:h_bottom, w_left:w_right]


    leftx, lefty, rightx, righty, img_out = find_lane_windows(img_cut)

    # 拟合
    left_fit = np.polyfit(lefty, leftx, 1)
    right_fit = np.polyfit(righty, rightx, 1)

    # 采样生成绘图点
    ploty = np.linspace(0, img_h-1, img_h)
    try:
        # 用二次曲线拟合
        left_fitx = left_fit[0]*ploty + left_fit[1] + 220
        right_fitx = right_fit[0]*ploty + right_fit[1] + 220
    except TypeError:
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty + 220
        right_fitx = 1 * ploty + 220


    # # 可视化
    # img_out[lefty, leftx] = [255, 0, 0]
    # img_out[righty, rightx] = [0, 0, 255]
    #
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.show()
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.show()

    left_fit_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_fit_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((left_fit_pts, right_fit_pts))

    cv2.polylines(img, np.int_([pts]), isClosed=False, color=(0, 255, 0), thickness=2)

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

        frame, canny, cut = detection_lane_windows(frame)
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


"""
@Name: 1-车道线跟踪.py
@Auth: Huohuo
@Date: 2023/6/29-10:16
@Desc: 
@Ver : 
code_idea
"""
import cv2
import numpy as np
import time

# ------------------------------------
#           数据集地址-MP4
# ------------------------------------
data_path = r"D:\WorkDirectory\cv_workdir\lane_detection\watercar_getlane\data\watercar.mp4"

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
#   return：lane = [left_lane_top, left_lane_bottom, right_lane_top, right_lane_bottom]
# ------------------------------------
def show_line(lines, img, left_mid_point, right_mid_point, slope_left_input, slope_right_input, sigma=200):

    global left_lane_top
    global left_lane_bottom
    global right_lane_top
    global right_lane_bottom
    print("*-------------------------frame--------------------------*")

    '''
        检测出的结果应该只有两条直线，left_slopes和 right_slopes中的值应该始终为1，
        若 len(left_slopes) 和 len(right_slopes) 出现大于1的情况
        应该将最后的值与初始进行比较，符合条件的车道线才会被绘制
        目前仅考虑了case：两条车道线斜率相反
    '''
    # 初始化，设置为0
    left_slopes = []
    right_slopes = []
    left_mid_point_xs = left_mid_point
    right_mid_point_xs = right_mid_point

    left_slope_compared = slope_left_input
    right_slope_compared = slope_right_input

    # 成功检测标识符
    signal_success = 0


    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        # 斜率
        slope = b / a
        # print("slope", slope)

        temp = 0
        # 斜率大于0，在显示图像上，应该是上右下左
        if 0 < slope < 0.4:
            x0 = a * rho
            y0 = b * rho
            # bottom
            x1 = int(x0 + sigma * (-b))
            y1 = int(y0 + sigma * (a))
            # top
            x2 = int(x0)
            y2 = int(y0)

            # 先判断直线中点位置是否符合要求，若不符合要求，不计入right_slopes
            left_mid_point_x = 0.5 * (x1 + x2)

            if abs(left_mid_point_x-left_mid_point) > 10:
                print("point Different!!!!!!!!!!!")
                continue
            if abs(slope - left_slope_compared) > 0.1:
                print("slope Different!!!!!!!!!!!")
                continue

            # 若同一边出现2条及以上直线
            if len(left_slopes) >= 1:
                continue
            # print("----left_slopes-----", slope)
            # print("x1, y1", x1, y1)
            # print("x2, y2", x2, y2)
            # print("left_mid_point_x", left_mid_point_x)
            left_mid_point_xs = left_mid_point_x
            left_slope_compared = slope

            cv2.line(img, (x1 + 220, y1 - 5 + int(img.shape[0] / 2)),
                     (x2 + 220, y2 - 5 + int(img.shape[0] / 2)), (0, 0, 255), 2)
            signal_success = 1
            left_slopes.append(slope)

            left_lane_top = [x1, y1]
            left_lane_bottom = [x2, y2]
        # 斜率小于0，在显示图像上，应该是上左下右
        elif -0.3 < slope < 0:
            x0 = a * rho
            y0 = b * rho
            # top
            x1 = int(x0 - 50 * (-b))            # 50 是为了将直线下移一些
            y1 = int(y0 - 50 * (a))
            # bottom
            x2 = int(x0 - (sigma+50) * (-b))    # +50 是为了将直线下移一些
            y2 = int(y0 - (sigma+50) * (a))

            # 先判断直线中点位置是否符合要求，若不符合要求，不计入right_slopes
            right_mid_point_x = 0.5 * (x1 + x2)

            if abs(right_mid_point_x - right_mid_point) > 20:
                print("point Different!!!!!!!!!!!")
                continue
            if abs(slope - right_slope_compared) > 0.1:
                print("slope Different!!!!!!!!!!!")
                continue

            # 若同一边出现2条及以上直线
            if len(right_slopes) >= 1:
                continue
            # print("----right_slopes-----", slope)
            # print("x1, y1", x1, y1)
            # print("x2, y2", x2, y2)
            # print("right_mid_point_x", right_mid_point_x)
            right_mid_point_xs = right_mid_point_x
            right_slope_compared = slope

            cv2.line(img, (x1 + 220, y1 - 5 + int(img.shape[0] / 2)),
                     (x2 + 220, y2 - 5 + int(img.shape[0] / 2)), (0, 0, 255), 2)
            right_slopes.append(slope)
            signal_success = 1

            right_lane_top = [x1, y1]
            right_lane_bottom = [x2, y2]

    # 在交叉路口会检测失败，被带偏，通过这里去掉筛选条件来矫正
    if signal_success == 0:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            # 斜率
            slope = b / a
            # print("slope", slope)

            temp = 0
            # 斜率大于0，在显示图像上，应该是上右下左
            if 0 < slope < 0.4:
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + sigma * (-b))
                y1 = int(y0 + sigma * (a))
                x2 = int(x0)
                y2 = int(y0)

                # 先判断直线中点位置是否符合要求，若不符合要求，不计入right_slopes
                left_mid_point_x = 0.5 * (x1 + x2)

                if abs(left_mid_point_x - left_mid_point) > 10:
                    print("point Different!!!!!!!!!!!")
                    pass
                if abs(slope - left_slope_compared) > 0.1:
                    print("slope Different!!!!!!!!!!!")
                    pass

                # 若同一边出现2条及以上直线
                if len(left_slopes) >= 1:
                    continue
                # print("----left_slopes-----", slope)
                # print("x1, y1", x1, y1)
                # print("x2, y2", x2, y2)
                # print("left_mid_point_x", left_mid_point_x)
                left_mid_point_xs = left_mid_point_x
                left_slope_compared = slope

                cv2.line(img, (x1 + 220, y1 - 5 + int(img.shape[0] / 2)),
                         (x2 + 220, y2 - 5 + int(img.shape[0] / 2)), (0, 0, 255), 2)
                signal_success = 1
                left_slopes.append(slope)
                left_lane_top = [x1, y1]
                left_lane_bottom = [x2, y2]

            # 斜率小于0，在显示图像上，应该是上左下右
            elif -0.3 < slope < 0:
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 - 50 * (-b))  # 50 是为了将直线下移一些
                y1 = int(y0 - 50 * (a))
                x2 = int(x0 - (sigma + 50) * (-b))  # +50 是为了将直线下移一些
                y2 = int(y0 - (sigma + 50) * (a))

                # 先判断直线中点位置是否符合要求，若不符合要求，不计入right_slopes
                right_mid_point_x = 0.5 * (x1 + x2)

                if abs(right_mid_point_x - right_mid_point) > 20:
                    print("point Different!!!!!!!!!!!")
                    pass
                if abs(slope - right_slope_compared) > 0.1:
                    print("slope Different!!!!!!!!!!!")
                    pass

                # 若同一边出现2条及以上直线
                if len(right_slopes) >= 1:
                    continue
                # print("----right_slopes-----", slope)
                # print("x1, y1", x1, y1)
                # print("x2, y2", x2, y2)
                # print("right_mid_point_x", right_mid_point_x)
                right_mid_point_xs = right_mid_point_x
                right_slope_compared = slope

                cv2.line(img, (x1 + 220, y1 - 5 + int(img.shape[0] / 2)),
                         (x2 + 220, y2 - 5 + int(img.shape[0] / 2)), (0, 0, 255), 2)
                right_slopes.append(slope)
                signal_success = 1
                right_lane_top = [x1, y1]
                right_lane_bottom = [x2, y2]

    return img, left_slopes, right_slopes, left_mid_point_xs, right_mid_point_xs, left_slope_compared, right_slope_compared, [left_lane_top, left_lane_bottom, right_lane_top, right_lane_bottom]


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
#     灰度化 + 边缘检测 + hough检测
#
# ------------------------------------
def detection_lane_hough(img, left_mid_point, right_mid_point, slope_left_input, slope_right_input):
    # print(img.shape)
    img_h = img.shape[0]
    img_w = img.shape[1]

    # 灰度化
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("img_gray", img_gray)
    # img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    # cv2.imshow("img_blur", img_blur)
    # img_blur = cv2.medianBlur(img_gray, 3, 0)
    img_blur = cv2.blur(img_gray, (3, 3))
    # 边缘检测
    img_edge = cv2.Canny(img_blur, 50, 150)
    # img_edge = sobel(img_edge, 1, 0)
    # img_edge = open_close(img_edge, k_x=3, k_y=3, open=False)
    # img_edge = open_close(img_edge, k_x=3, k_y=3, open=True)
    # cv2.imshow("canny", img_edge)

    # 手动选取ROI区域    img = img[int(img_h/2):-5, 220:390]
    h_top = int(img_h/2)
    h_bottom = -5
    w_left = 220
    w_right = 390
    img_cut = img_edge[h_top:h_bottom, w_left:w_right]
    # cv2.imshow("img_cut", img_cut)


    # hough检测 返回的是每一条直线对应的（ρ，θ）
    # ρ的精度为1， θ精度为1°，长度阈值200
    lines = cv2.HoughLines(img_cut, 1, 2*np.pi/180, 60)

    # 绘制检测结果
    # lane = [left_lane_top, left_lane_bottom, right_lane_top, right_lane_bottom]
    img, left_slopes, right_slopes, left_mid_point_xs, right_mid_point_xs, slope_lefts, slope_rights, lane = show_line(lines, img, left_mid_point, right_mid_point, slope_left_input, slope_right_input)

    return img, img_edge, img_cut, left_mid_point_xs, right_mid_point_xs, slope_lefts, slope_rights, lane


# ------------------------------------
#           kalman滤波
# ------------------------------------
# class LaneTracker:
#     def __init__(self, n_lanes, proc_noise_):



# 打开视频文件
video_path = 'data/Railway.mp4'  # 视频文件路径
cap = cv2.VideoCapture(video_path)

# 定义视频编写器的输出文件名、编码器、帧率和分辨率等参数
output_file = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4编码器
fps = cap.get(cv2.CAP_PROP_FPS)  # 使用与输入视频相同的帧率
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 使用与输入视频相同的宽度
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 使用与输入视频相同的高度

# 创建视频编写器对象
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))


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

# 设置初始值
left_mid_point = 61.5
right_mid_point = 130.5
slope_left_input = 0.249328
slope_right_input = -0.2867454


# 读取并显示视频帧
while True:
    # 逐帧读取视频
    ret, frame = cap.read()
    # cv2.imshow("frame", frame)

    # 如果成功读取帧
    if ret:
        # lane = [left_lane_top, left_lane_bottom, right_lane_top, right_lane_bottom]
        frame, canny, cut, left_mid_point_xs, right_mid_point_xs, slope_lefts, slope_rights, lane = detection_lane_hough(frame, left_mid_point, right_mid_point, slope_left_input, slope_right_input)
        left_mid_point = left_mid_point_xs
        right_mid_point = right_mid_point_xs
        slope_left_input = slope_lefts
        slope_right_input = slope_rights
        # 在帧上绘制帧率
        cv2.putText(frame, "FPS: {:.2f}".format(frame_count / (time.time() - start_time)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 保存视频
        out.write(frame)

        # 输出直线
        # lane = [left_lane_rho, left_lane_theta, right_lane_rho, right_lane_theta]
        print("lane left_lane is: ", lane[0], lane[1])
        print("lane right_lane_rho is: ", lane[2], lane[3])



        # 在窗口中显示帧
        cv2.imshow('Video Frame', frame)
        # cv2.imshow('Video Canny', canny)
        # cv2.imshow('Video Cut', cut)

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
out.release()
cv2.destroyAllWindows()



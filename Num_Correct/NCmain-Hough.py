"""
@Name: main-Hough.py
@Auth: Huohuo
@Date: 2023/7/28-15:36
@Desc: 
@Ver : 
code_idea
"""
import cv2
import os
import numpy as np
import time

# ------------------------------------------------
#                自适应直线检测（弃用）
#  需要根据横/纵向图片，确定自适应的hough阈值
# 横向：最少四字符至少有8个点，最多六字符+校验码（七字符，有空隙当八字符来算），gap设为w/(nums=7*2)
# 纵向：四字符至少有8个点，最多六字符+校验码（七字符，有空隙当八字符来算），gap设为h/(nums=7*2)
# 中间阈值可能还需要调整
# input: 边缘图像
#
# ------------------------------------------------
def auto_line_detect(img):
    # 首先通过长宽判断是横向图像还是纵向图像
    img_h, img_w = img.shape
    print("img.shape", img.shape)

    # 若为横向图像，需要按比例确定缩短尺寸，暂定0.7
    if img_h < img_w:
        scale = 0.7
        threshold = img_w//30
        min_length = int(img_w * 0.9)
        print('Horizontal min_length', min_length)
        # gap = int(img_w / (7 * 3))
        gap = int(img_w * 0.7)
        lines = cv2.HoughLinesP(img, rho=1, theta=np.pi/30, threshold=threshold,
                                minLineLength=min_length, maxLineGap=gap)
        return lines
    # 若为纵向图像
    elif img_h > img_w:
        scale = 0.7
        threshold = img_h // 100
        min_length_y = int(img_h * 0.9)
        print('Vertical min_length', min_length_y)
        # gap = int(img_h / (7 * 3))
        gap_y = int(img_h * 0.5)
        lines = cv2.HoughLinesP(img, rho=10, theta=np.pi/30, threshold=8,
                                minLineLength=min_length_y, maxLineGap=gap_y)
        return lines

        # -----------------------------main--------------------------
        # # 直线检测
        # lines = auto_line_detect(edge)
        # if lines is not None:
        #     for line in lines:
        #         # print(line)
        #         line = line[0].astype(np.uint8)
                # cv2.line(image, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 2)
                # cv2.line(image, (lines[2], line[3]), (lines[0], line[1]), (255, 0, 0), 2)
        # else:
            # continue

# -------------------------------------------------------
#                      形态学运算
# open = True:      开运算
# open = False:     闭运算
# -------------------------------------------------------
def open_close(img, k_x=3, k_y=3, frequency=1, open=True):
    morph_kernel = np.ones((k_x, k_y), np.uint8)
    if open:
        img_erode = cv2.erode(img, morph_kernel, frequency)
        img_dilate = cv2.dilate(img_erode, morph_kernel, frequency)
        return img_dilate

    else:
        img_dilate = cv2.dilate(img, morph_kernel, frequency)
        img_erode = cv2.erode(img_dilate, morph_kernel, frequency)
        return img_erode

# -------------------------------------------------------
#       若图像中白色像素多余黑色像素，则将图像反转
# -------------------------------------------------------
def check_and_reverse_image(img):
    # 统计黑色和白色像素个数
    write_num = np.sum(img) // 255
    black_num = img.shape[0] * img.shape[1] - write_num


    # 判断是否需要反转图像
    if write_num > black_num and img.shape[0]>img.shape[1]:
        img = 255 - img
        print("图像已反转！")
    return img

# -------------------------------------------------------
#            获取最大区域，并用矩形框包裹
# 1) cv2.findContours中设置cv2.RETR_EXTERNAL，只检测外部轮廓
# 2) 遍历所有轮廓，计算轮廓面积，滤去面积小于 图像长宽最小值//5的平方（实验得到） 数值的轮廓
# 3) 计算矩形中点坐标
# 4) 返回原图、中点、矩形框坐标
# -------------------------------------------------------
def Get_Max_Rect(img, src):
    global center
    img_h, img_w = img.shape

    centers = []
    boxs = []
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # 最小外接矩形
        rect = cv2.minAreaRect(contour)

        if cv2.contourArea(contour) < (min(img_h, img_w)//5)**2:
            print("轮廓面积过小，已滤去")
            continue
        # --------------------------------------------
        # 获取最小外接矩形的四个顶点,
        # 这部分获得的box坐标不一定是从左上角开始排列的
        # 因此将边框斜率均求出，求出后，通过判断是横向/纵向编号来分配对应的斜率
        # --------------------------------------------
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 计算矩形中点坐标
        x_center = int((box[0][0] + box[2][0])/2)
        y_center = int((box[0][1] + box[2][1])/2)

        center = [x_center, y_center]
        # 在原始图像上绘制最小外接矩形的轮廓
        cv2.drawContours(src, [box], 0, (0, 255, 0), 2)       # draw
        # cv2.drawContours(src, contours, -1, (0, 0, 255), 3)       # draw

        centers.append(center)
        boxs.append(box)
    return src, centers, boxs




# ------------------------------------------------
#               处理流程v1（耗时0.08s）
# 1）  读取灰度图
# 2）  二值化
# 3）  开运算消除杂点
# 4）  膨胀(连通域需要遍历周围像素点，计算时间复杂度太高了，直接膨胀矩阵运算更快)
# 5）  轮廓检测(详见Get_Max_Rect) return 原图、中点、矩形框坐标
# 6）  判断轮廓检测结果
# 7）  若检测结果仅一个：
#         判断为横向编号还是纵向编号
#            若为横向编号： 选取矩形短边斜率为矫正参数
#            若为纵向编号： 选取矩形长边斜率为矫正参数
# 8）  若检测结果有多个：
#         采用最小二乘法来拟合得到斜率，用作矫正参数
# 9)   得到矫正参数（斜率）后，转为与x轴的夹角，分情况（钝角、锐角、负角度）进行纠正
# 10）  以图像中点为旋转中心，构建旋转矩阵，对图像进行旋转，得到矫正后的图像
# ------------------------------------------------
def main(src_path):
    if not os.path.exists(src_path):
        raise ValueError(print(f"文件夹路径’{src_path}‘不存在"))

    for filename in os.listdir(src_path):
        start_time = time.time()
        filepath = os.path.join(src_path, filename)
        print("----------------------------------------------------")
        print(filepath)
        # 读入原始图像
        image = cv2.imread(filepath)
        cv2.imshow("src", image)

        # 灰度化
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 二值化
        _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


        img_h, img_w = img_gray.shape
        k = max(min(img_h, img_w)//50, 3)
        # print("k is:", k)
        morph_kernel = np.ones((k, k), np.uint8)

        # 开运算
        thresh = open_close(thresh, k_x=k, k_y=k, frequency=3)
        cv2.imshow("thresh", thresh)

        # 膨胀
        img_dilate = cv2.dilate(thresh, morph_kernel, 3)

        # 判断是否需要反转图像
        img_dilate = check_and_reverse_image(img_dilate)

        cv2.imshow("img_dilate", img_dilate)

        # 检测轮廓
        image, centers, boxs = Get_Max_Rect(img_dilate, image)

        # --------------------------------------------
        # 检测结果存储在centers中
        # centers返回值可能是一个 或 多个
        # 若为1个，直接使用box的边缘作为倾斜角度
        # 若为多个，在滤去干扰项后，采用最小二乘法拟合
        # --------------------------------------------
        global slope
        if len(centers) == 1:
            box = boxs[0]
            # k = y2-y1 / x2 -x1
            slope_1 = (box[1][1]-box[0][1]) / (box[1][0]-box[0][0])     # 后续再细化，有点问题，不一定是按左上角第一个
            slope_2 = (box[0][1]-box[3][1]) / (box[0][0]-box[3][0])
            print("slope_1, slope_2 is: ", slope_1, slope_2)
            # ---------------------------------
            #      纵向对应的是较大的斜率
            # ---------------------------------
            if img_h > img_w:
                cv2.line(image, box[0], box[1], (0, 255, 255), 2)       # draw
                slope = max(slope_1, slope_2, key=abs)
                print("当前是纵向，选取矩形长边斜率")
                # return slope_h
            # ---------------------------------
            #      横向对应的是较小的斜率
            # ---------------------------------
            else:
                cv2.line(image, box[0], box[3], (0, 255, 255), 2)       # draw
                slope = min(slope_1, slope_2, key=abs)
                print("当前是横向，选取矩形短边斜率")
                # return slope_v
            # return slope_v, slope_h

        # center返回值为多个
        elif len(centers) > 1:
            # 对一系列矩形中心点可视化
            for center in centers:
                # print("center", center)
                cv2.circle(image, center, 2, (0, 0, 255), 2)       # draw
                pass
            print("当前是最小二乘法拟合结果")
            # centers = np.transpose(centers)
            centers = np.array(centers)

            vx, vy, x, y = cv2.fitLine(centers, cv2.DIST_L2, 0, 0.01, 0.01)

            point1 = (int(x - 100 * vx), int(y - 100 * vy))
            point2 = (int(x + 100 * vx), int(y + 100 * vy))
            cv2.line(image, point1, point2, (0, 255, 255), 2)       # draw
            slope = vy / vx
        else:
            continue

        print("当前用于矫正的直线斜率为:", slope)

        # --------------------------------------------
        #              纠正（旋转）       这部分+90也要分情况讨论
        # --------------------------------------------
        # 计算夹角（弧度）
        angle_rad = np.arctan(slope)

        # 转换为角度（可选）
        angle_degrees = float(np.degrees(angle_rad))
        print("处理前的旋转角度：", angle_degrees)

        # --------------------------------------------
        # 由于cv2.getRotationMatrix2D旋转角度是相对与中心而言的，
        # 但计算的夹角是相对与x轴正方向得到的
        # 因此需要对旋转度数进行平衡，超过45°的部分要减90，
        # 少于0°的部分要补全，目前以下的分类讨论情况已可以满足需求
        # --------------------------------------------
        if img_h > img_w and slope < 0:
            angle_degrees = angle_degrees + 90
        elif img_h > img_w and slope > 0:
            angle_degrees = angle_degrees - 90
        elif img_w > img_h:         # 横向的不矫正，若要矫正，将本段注释掉即可
            angle_degrees = 0
        else:
            angle_degrees = angle_degrees


        print("angle_degrees", angle_degrees)
        # 计算旋转中心
        center_x = img_w // 2
        center_y = img_h // 2

        # 构建旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle_degrees, 1.0)
        # rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), 25.5, 1.0)

        # 进行图像旋转
        rotated_image = cv2.warpAffine(image, rotation_matrix, (img_w, img_h))


        cv2.imshow("src", image)
        cv2.imshow("rotated_image", rotated_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("Used Time :", time.time() - start_time)

    return 0


if __name__ == '__main__':
    # ------------------------------------------------
    #                   数据集路径
    # ------------------------------------------------
    src_path = r"D:\WorkDirectory\school_project\leader\result_crop"


    main(src_path)


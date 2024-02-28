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
import os

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
def open_close(img, k_x=5, k_y=5, open=False):
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
#             delete blue
# ------------------------------------
def Delete_Blue(img):
    hvs_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([150, 255, 255])

    blue_mask = cv2.inRange(hvs_img, lower_blue, upper_blue)
    result = img.copy()
    result[blue_mask != 0] = (255, 255, 255)
    # result = cv2.cvtColor(result, cv2.COLOR_HSV2RGB)
    cv2.imshow("delete blue", result)
    return result

# ------------------------------------
#               segment
# ------------------------------------
def segment(img):
    # 展示原图
    cv2.imshow("img", img)
    # 删除蓝色刻度线，替换为白色，见Delete_Blue
    img = Delete_Blue(img)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化a
    thresh, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imshow("img_binary", img_binary)

    # canny
    img_blur = cv2.GaussianBlur(img_binary, (3, 3), 0)
    # img_blur = cv2.medianBlur(img_binary, 3, 0)
    # img_blur = cv2.blur(img_binary, (3, 3))

    # open or close
    img_binary = open_close(img_blur)
    cv2.imshow("img_open", img_binary)

    # 边缘检测
    img_edge = cv2.Canny(img_binary, 30, 220)
    # img_edge = sobel(img_edge, 0, 1)
    cv2.imshow("img_edge", img_edge)


    return img, img_edge, img_binary


# pipline
def Pipline(img_edge):
    img_cut = img_edge[0:480, 50:250]
    cv2.imshow("img_cut", img_cut)

    morph_kernel = np.ones((5, 5), np.uint8)

    img_dilate = cv2.dilate(img_cut, morph_kernel, 1)
    cv2.imshow("img_dilate", img_dilate)

    # find contour
    contours, _ = cv2.findContours(img_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # get area thresh
    areas_thresh = 200


    # create one black img as boardq
    mask = np.zeros(img_dilate.shape[:2], dtype=np.uint8)

    # draw all contours expect the min area on the mask
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > areas_thresh:
            cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

    # cover the mask on the src img, in order to delete the min area
    result = cv2.bitwise_and(img_dilate, img_dilate, mask=mask)

    cv2.imshow("Pipline", result)

    return result

# -------------------------------------------
#               get his
# -------------------------------------------
def Create_his(img_brainry):
    # transform img to one dismission array
    compressed_img = np.sum(img_brainry, axis=1)

    nonzero_indices = np.where(compressed_img != 0)[0]
    diff = np.diff(nonzero_indices)

    boundaries = np.where(diff > 1)[0]

    nonzero_boundaries = np.split(nonzero_indices, boundaries+1)
    i = 0
    start = []
    end = []
    for boundary in nonzero_boundaries:
        start_index = boundary[0]
        end_index = boundary[-1]
        print("Start", start_index)
        print("End", end_index)
        print("----------")
        cv2.imshow(str(i) + "resul", img_brainry[start_index:end_index, :])
        i+=1
        start.append(start_index)
        end.append(end_index)
    print("-----------***Detectiong Result***----------------")
    print("start",start)
    print("end",end)
    avg = int((start[0]+end[0])/2)
    print("avg",avg)
    print("Now sediment ratio is :", (end[1]-start[1])/(end[1]-avg))
    return (end[1]-start[1])/(end[1]-avg)



src_path = "./data3"
save_path1 = "./data3_result_edge"
save_path2 = "./data3_result"

for imgname in os.listdir(src_path):
    # print(imgname)
    img = cv2.imread(src_path + '/' + imgname)
    img = cv2.resize(img, (540, 480))
    src, img_edge, img_binary = segment(img)
    # cv2.imshow("result", result)
    result = Pipline(img_edge)
    ratio = Create_his(result)
    ratio = round(ratio, 2)
    cv2.putText(img, "Sediment ratio is "+str(ratio), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
    cv2.imshow("img",img)
    cv2.imwrite(save_path1 + '/' + imgname, img_edge)
    cv2.imwrite(save_path2 + '/' + imgname, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




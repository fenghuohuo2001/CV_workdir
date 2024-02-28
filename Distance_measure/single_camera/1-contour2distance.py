"""
@Name: 1-contour2distance.py
@Auth: Huohuo
@Date: 2023/11/27-19:00
@Desc: 
@Ver : 
code_idea
"""
import cv2
import numpy as np

# ----------------------------------------------------
#               使用相似三角形测距
# ----------------------------------------------------
def get_object_contour(img, x1, x2):
    test_x1 = x1
    test_x2 = x2
    cv2.line(img, (test_x1, 0), (test_x1, 480), (255, 0, 0), 1)
    cv2.line(img, (test_x2, 0), (test_x2, 480), (255, 0, 0), 1)
    cv2.imshow("get_object_contour", img)
    cv2.waitKey(1)
    return test_x2 - test_x1

def get_distance(real_width, pixel_width, ratio):
    distance = (real_width / pixel_width) * ratio
    return distance



def main():
    img = cv2.imread("data/distance3-89cm-width50cm.jpg")

    orignal_real_width = 50         # cm

    orignal_pixel_width = get_object_contour(img.copy(), x1=358, x2=420)

    # 焦距
    distance_ratio = (orignal_pixel_width / orignal_real_width) * (3 * 89)

    # 计算2-89的距离
    pixel_width_1 = get_object_contour(cv2.imread("data/2-89.jpg"), x1=332, x2=425)
    distance_2_89 = get_distance(orignal_real_width, pixel_width_1, distance_ratio)
    print("distance_2_89 is: {:.2f}".format(distance_2_89), "should_be:", 2*89)

    # 计算4-89的距离
    pixel_width_2 = get_object_contour(cv2.imread("data/4-89.jpg"), x1=316, x2=360)
    distance_4_89 = get_distance(orignal_real_width, pixel_width_2, distance_ratio)
    print("distance_4_89 is: {:.2f}".format(distance_4_89), "should_be:", 4 * 89)


    cv2.destroyAllWindows()

    return 0


if __name__ == '__main__':
    main()

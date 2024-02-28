"""
@Name: Click2GetPointsValue.py
@Auth: Huohuo
@Date: 2023/7/9-14:08
@Desc: 
@Ver : 
code_idea
"""

import cv2

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Clicked at coordinates (x, y):", x, y)

image = cv2.imread("start.jpg")  # 替换为您的图像路径
img_h, img_w = image.shape[:2]
resize_scale = 1
image = cv2.resize(image, (img_w//resize_scale, img_h//resize_scale))
cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_callback)

while True:
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # 按下 'q' 键退出循环
        break

cv2.destroyAllWindows()


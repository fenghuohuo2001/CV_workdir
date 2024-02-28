"""
@Name: mouse_callback.py
@Auth: Huohuo
@Date: 2023/7/4-14:35
@Desc: 
@Ver : 
code_idea
"""
import cv2
import numpy as np

# -------------------------------
#      定义鼠标点击事件的回调函数
# -------------------------------
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Mouse clicked at (x={}, y={})".format(x, y))

# -------------------------------
#        鼠标点击返回坐标
# -------------------------------
def get_coord(image):
    # 创建窗口并将鼠标回调函数与窗口绑定
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)

    while True:
        cv2.imshow("Image", image)

        # 等待按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 关闭窗口
    cv2.destroyAllWindows()
    return

if __name__ == "__main__":
    img = cv2.imread("../Magnitude Spectrum object.jpg")[500:900, 3500:4200]
    get_coord(img)
    pass

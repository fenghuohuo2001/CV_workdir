"""
@Name: change_piex.py
@Auth: Huohuo
@Date: 2023/3/7-15:40
@Desc: 
@Ver : 
code_idea
"""
import cv2

img = cv2.imread("zhoukaile.jpg")
img_resize = cv2.resize(img, (1000, 563), interpolation=cv2.INTER_LANCZOS4)
cv2.imshow("src", img)
cv2.imshow("resize", img_resize)
cv2.imwrite("resize.jpg", img_resize)
cv2.waitKey(0)
cv2.destroyAllWindows()
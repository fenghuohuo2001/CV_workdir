"""
@Name: 二值化.py
@Auth: Huohuo
@Date: 2023/4/23-19:15
@Desc: 
@Ver : 
code_idea
"""
import cv2

img = cv2.imread(r"D:\WorkDirectory\mywork\myself_OCR\paper_used_data\train\test_en\3848.jpg")
cv2.imshow("img", 255-img)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("img_gray", 255-img_gray)

th, result = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

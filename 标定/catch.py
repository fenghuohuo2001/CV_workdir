"""
@Name: catch.py
@Auth: Huohuo
@Date: 2023/7/13-11:21
@Desc: 
@Ver : 
code_idea
"""
import cv2

camera=cv2.VideoCapture(1)
i = 68
while 1:
    (grabbed, img) = camera.read()
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('j'):  # 按j保存一张图片
        i += 1
        u = str(i)
        firename=str('./data/data-angle=170/img'+u+'.jpg')
        cv2.imwrite(firename, img)
        print('写入：',firename)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

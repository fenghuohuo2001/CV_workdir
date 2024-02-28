# # -*- 练习处理视频 -*-
# """
# 功能：视频的读取及处理
# 作者：fenghuohuo
# 日期：2021年11月22日
# """
# import cv2
#
# cap = cv2.VideoCapture(0)
# # 参数0表示默认为笔记本内置第一个摄像头，可以改为视频所在路径
#
# cap.isOpened()
# # 判断视频对象是否成功读取，成功读取则返回Ture
#
# ret, frame = cap.read()
# # 按帧数读取视频，返回的ret为bool型，正确读取返回Ture，失败或读取视频末尾则返回False
# # frame为每一帧的图像，BRG格式
#
# key = cv2.waitKey(1)
# # 等待键盘输入，参数1表示延时1ms切换到下一帧，参数0表示暂停在当前帧
#
# cv2.VideoWriter(filename, fourcc, fps, frameSize[, isColor])
# # 第一个参数是要保存的文件的路径
# # fourcc 指定编码器
# '''
# # fourcc 本身是一个 32 位的无符号数值，用 4 个字母表示采用的编码器。 常用的有 “DIVX"、”MJPG"、“XVID”、“X264"。可用的列表在这里。
# #
# # 推荐使用 ”XVID"，但一般依据你的电脑环境安装了哪些编码器。
# #
# # 如果 fourcc 采用 -1，系统可能会弹出一个对话框让你进行选择，但是我没有试验成功过。
# '''
# # fps 要保存的视频的帧率
# # frameSize 要保存的文件的画面尺寸
# # isColor 指示是黑白画面还是彩色的画面
# #
# out = cv2.VideoWriter('testwrite.avi',fourcc, 20.0, (1920,1080),True)
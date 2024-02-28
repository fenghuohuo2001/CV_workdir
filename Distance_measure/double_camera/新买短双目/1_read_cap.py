"""
@Name: 1_read_cap.py
@Auth: Huohuo
@Date: 2023/11/30-8:30
@Desc: 
@Ver : 
code_idea
"""
'''
CAP_ANY          = 0             # Auto detect
CAP_VFW          = 200           # Video For Windows (obsolete, removed)
CAP_V4L          = 200           # V4L/V4L2 capturing support
CAP_V4L2         = CAP_V4L       # Same as CAP_V4L
CAP_FIREWIRE     = 300           # IEEE 1394 drivers
CAP_FIREWARE     = CAP_FIREWIRE  # Same value as CAP_FIREWIRE
CAP_IEEE1394     = CAP_FIREWIRE  # Same value as CAP_FIREWIRE
CAP_DC1394       = CAP_FIREWIRE  # Same value as CAP_FIREWIRE
CAP_CMU1394      = CAP_FIREWIRE  # Same value as CAP_FIREWIRE
CAP_QT           = 500           # QuickTime (obsolete, removed)
CAP_UNICAP       = 600           # Unicap drivers (obsolete, removed)
CAP_DSHOW        = 700           # DirectShow (via videoInput)
CAP_PVAPI        = 800           # PvAPI, Prosilica GigE SDK
CAP_OPENNI       = 900           # OpenNI (for Kinect)
CAP_OPENNI_ASUS  = 910           # OpenNI (for Asus Xtion)
CAP_ANDROID      = 1000          # Android - not used
CAP_XIAPI        = 1100          # XIMEA Camera API
CAP_AVFOUNDATION = 1200          # AVFoundation framework for iOS (OS X Lion will have the same API)
CAP_GIGANETIX    = 1300          # Smartek Giganetix GigEVisionSDK
CAP_MSMF         = 1400          # Microsoft Media Foundation (via videoInput)
CAP_WINRT        = 1410          # Microsoft Windows Runtime using Media Foundation
CAP_INTELPERC    = 1500          # RealSense (former Intel Perceptual Computing SDK)
CAP_REALSENSE    = 1500          # Synonym for CAP_INTELPERC
CAP_OPENNI2      = 1600          # OpenNI2 (for Kinect)
CAP_OPENNI2_ASUS = 1610          # OpenNI2 (for Asus Xtion and Occipital Structure sensors)
CAP_GPHOTO2      = 1700          # gPhoto2 connection
CAP_GSTREAMER    = 1800          # GStreamer
CAP_FFMPEG       = 1900          # Open and record video file or stream using the FFMPEG library
CAP_IMAGES       = 2000          # OpenCV Image Sequence (e.g. img_%02d.jpg)
CAP_ARAVIS       = 2100          # Aravis SDK
CAP_OPENCV_MJPEG = 2200          # Built-in OpenCV MotionJPEG codec
CAP_INTEL_MFX    = 2300          # Intel MediaSDK
CAP_XINE         = 2400          # XINE engine (Linux)

'''
# -*- coding: utf-8 -*-
import cv2

cv2.namedWindow("left")
cv2.namedWindow("right")
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# camera = cv2.VideoCapture(0)
# 设置帧率
# camera.set(cv2.CAP_PROP_FPS, 30)
# 设置帧宽
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
# 设置帧高
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

counter = 0

while True:
    ret, frame = camera.read()
    # 裁剪坐标为[y0:y1, x0:x1] HEIGHT*WIDTH
    left_frame = frame[0:720, 0:1280]
    right_frame = frame[0:720, 1280:2560]

    cv2.imshow("left", left_frame)
    cv2.imshow("right", right_frame)


    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("s"):
        cv2.imwrite("save_image/left_image/left_" + str(counter) + ".jpg", left_frame)
        cv2.imwrite("save_image/right_image/right_" + str(counter) + ".jpg", right_frame)
        counter += 1
camera.release()
cv2.destroyWindow("left")
cv2.destroyWindow("right")

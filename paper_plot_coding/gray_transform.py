"""
@Name: gray_transform.py
@Auth: Huohuo
@Date: 2023/3/13-15:17
@Desc: 
@Ver : 
code_idea
"""
import math

import cv2
import numpy as np


def logTransform(c, img):
    # 3通道RGB
    '''h,w,d = img.shape[0],img.shape[1],img.shape[2]
    new_img = np.zeros((h,w,d))
    for i in range(h):
        for j in range(w):
            for k in range(d):
                new_img[i,j,k] = c*(math.log(1.0+img[i,j,k]))'''

    # 灰度图专属
    h, w = img.shape[0], img.shape[1]
    new_img = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            new_img[i, j] = c * (math.log(1.0 + img[i, j]))

    new_img = cv2.normalize(new_img, new_img, 0, 255, cv2.NORM_MINMAX)
    new_img = cv2.convertScaleAbs(new_img)  # 原文中少了这一步
    return new_img

def adjust_gamma(imgs, gamma=10.0):
    new_imgs = np.power(imgs/255, gamma)
    return new_imgs

def mutil_gamma(imgs, gamma_list=[5,20,60]):
    board = np.zeros(imgs.shape)
    for gamma in gamma_list:
        new_imgs = np.power(imgs/255, gamma)
        board += new_imgs/(len(gamma_list))
    return board


def gray_log():
    # 灰度图
    img_path = "../Retinex/data/img_gray.png"
    img = cv2.imread(img_path, 0)
    # cv2.imshow("src", img)
    log_img = logTransform(1, img)
    gam_img = adjust_gamma(img, gamma=14)
    mgam_img = mutil_gamma(img)
    # cv2.imshow('log_img', log_img)
    cv2.imshow('gam_img', gam_img)
    cv2.imshow('mgam_img', mgam_img)
    cv2.waitKey(0)

gray_log()

def color_log():
    # 灰度图
    img_path = "../Retinex/data/img.png"
    img = cv2.imread(img_path)
    cv2.imshow("src", img)
    log_img = np.zeros(img.shape, np.uint8)
    gam_img = np.zeros(img.shape, np.uint8)
    for i in range(3):
        print(i)
        log_img[:, :, i] = logTransform(1, img[:, :, i])
        gam_img[:, :, i] = adjust_gamma(img[:, :, i], gamma=1)
    cv2.imshow('log_img', log_img)
    cv2.imshow('gam_img', gam_img)
    cv2.waitKey(0)

# color_log()
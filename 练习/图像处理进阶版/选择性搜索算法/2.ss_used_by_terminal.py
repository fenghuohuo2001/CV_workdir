# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 1.selective_search.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/9/4 10:37
@Function：估计物体可能出现的区域框
python 2.ss_used_by_terminal.py ./test_img/img.png f
python 2.ss_used_by_terminal.py ./test_img/frame3090.jpg f
python 2.ss_used_by_terminal.py ./test_img/chip.jpg f
\
m  increase the region
l  decrease the region
q  quit the program
"""
import sys
import cv2

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)  # print the top note
        sys.exit(1)
    # speed up using multithreads
    cv2.setUseOptimized(True)       # open program optimization
    cv2.setNumThreads(4)            # set up four threads

    im = cv2.imread(sys.argv[1])
    newHeight = 800
    newWidth = int(im.shape[1]*newHeight/im.shape[0])
    im = cv2.resize(im, (newWidth, newHeight))

    # 使用默认参数开始选择性搜索 initializes a selective search algorithm
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    # add detection picture
    ss.setBaseImage(im)

    # switch to fast but low recall
    if (sys.argv[2] == 'f'):
        ss.switchToSelectiveSearchFast()
    # switch to high recall but slow
    elif (sys.argv[2] == 'q'):
        ss.switchToSelectiveSearchQuality()
    else:
        print(__doc__)

    # run selective search segmentation on input image
    rects = ss.process()   # the 'rects' include frame coordinates
    print('total number of region proposals:{}'.format(len(rects)))

    # number of region proposals to show
    numshowrects = 100
    # increment to increase/decrease total number of region proposals to be shown
    increment = 50

    while True:
        # creat a copy of original image
        imOut = im.copy()

        # itereate over all the region proposals
        for i, rect in enumerate(rects):
            # draw rectangle for region proposal till numshowrects
            if (i < numshowrects):
                x, y, w, h = rect
                cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
            else:
                break
        # show output
        cv2.imshow("output", imOut)

        # record key pressed
        k = cv2.waitKey(0) & 0xFF   # 判断按键是什么，同一按键ASCLL码后8位一定相同

        # press m
        if k == 109:
            # increase total number of rectangles to show by increment
            numshowrects += increment
        # press l
        elif k == 108 and numshowrects > increment:
            # decrease total number of rectangles to show by increment
            numshowrects -= increment
        # press q
        elif k == 113:
            break
    # close image show window
    cv2.destroyAllWindows()
















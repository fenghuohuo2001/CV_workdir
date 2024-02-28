"""
@Name: get_pic_channel.py
@Auth: Huohuo
@Date: 2023/2/22-11:05
@Desc: 
@Ver : 
code_idea
"""
import cv2


def main():
    img = cv2.imread(r"E:\DBNet.pytorch-master\DBNet.pytorch-master\myselfdata_all\test\img\8466.jpg")
    img_msrcr = cv2.imread(r"E:\DBNet.pytorch-master\DBNet.pytorch-master\myselfdata_all_msrcr\test\img\8466.jpg")
    print("img_shape is", img.shape)
    print("img_msrcr_shape is", img_msrcr.shape)


    return 0


if __name__ == '__main__':
    main()

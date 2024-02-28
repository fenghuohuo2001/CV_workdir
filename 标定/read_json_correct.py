import cv2
import numpy as np
import json

def load_camera_parameters(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    ret = data['ret']
    mtx = np.array(data['mtx'], np.float32)
    dist = np.array(data['dist'], np.float32)
    rvecs = np.array(data['rvecs'], np.float32)
    tvecs = np.array(data['tvecs'], np.float32)
    return mtx, dist

def undistort_image(img, mtx, dist):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    undistorted_img = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y + h, x:x + w]
    return undistorted_img

def main():
    camera = cv2.VideoCapture(1)
    json_file = 'camera_params-170.json'  # 替换为你的相机参数文件
    mtx, dist = load_camera_parameters(json_file)

    while True:
        grabbed, img = camera.read()
        if not grabbed:
            break

        undistorted_img = undistort_image(img, mtx, dist)

        cv2.imshow('Undistorted Image', undistorted_img)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):  # 按'q'键保存一张图片
            cv2.imwrite("../result/frame.jpg", undistorted_img)
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

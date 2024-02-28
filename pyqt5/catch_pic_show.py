import sys
import cv2
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Camera App")
        self.setGeometry(100, 100, 640, 480)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()

        self.image_label = QLabel(self)
        self.layout.addWidget(self.image_label)

        self.capture_button = QPushButton("Capture", self)
        self.layout.addWidget(self.capture_button)

        self.capture_button.clicked.connect(self.capture_image)

        self.central_widget.setLayout(self.layout)

        self.capture = cv2.VideoCapture(0)  # 打开默认摄像头
        self.image_count = 0  # 用于跟踪图像数量

    def capture_image(self):
        ret, frame = self.capture.read()  # 从摄像头捕获图像
        if ret:
            image_path = f"captured_image_{self.image_count}.jpg"  # 图像文件名
            cv2.imwrite(image_path, frame)  # 保存图像到文件
            self.display_image(image_path)
            self.image_count += 1  # 增加图像数量

    def display_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)  # 图像居中显示

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())

import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog
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

    def capture_image(self):
        ret, frame = self.capture.read()  # 从摄像头捕获图像
        if ret:
            # 选择保存图像的文件路径
            options = QFileDialog.Options()
            # file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Image Files (*.jpg *.png)", options=options)
            file_path = "save.png"
            if file_path:
                cv2.imwrite(file_path, frame)  # 保存图像到文件
                self.display_image(file_path)

    def display_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)  # 图像居中显示

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QGroupBox, QLabel, QGridLayout, QWidget
from PyQt5.QtWidgets import QApplication, QMainWindow, QGroupBox, QLabel, QGridLayout, QVBoxLayout, QWidget


class LabelGroupApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Label Group App")
        self.setGeometry(100, 100, 400, 300)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QGridLayout()

        # 创建一个标签组
        self.label_group = QGroupBox("Label Group")
        self.label_group.setStyleSheet("QGroupBox { border: 2px solid black; }")

        # 创建标签并添加到标签组中
        for i in range(6):
            label = QLabel(f"Label {i + 1}")
            self.layout.addWidget(label, i // 2, i % 2)  # 放置在左上角三分之一处

        # 设置标签组的布局
        self.label_group.setLayout(self.layout)

        # 将标签组添加到主布局
        self.central_layout = QVBoxLayout()
        self.central_layout.addWidget(self.label_group)
        self.central_widget.setLayout(self.central_layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LabelGroupApp()
    window.show()
    sys.exit(app.exec_())

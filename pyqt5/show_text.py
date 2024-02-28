import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QTextEdit, QVBoxLayout, QWidget

class TextDisplayApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Text Display App")
        self.setGeometry(100, 100, 400, 300)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()

        # 创建一个 QLabel 用于内容说明
        self.label = QLabel("Content Description:", self)
        self.layout.addWidget(self.label)

        # 创建一个文本显示栏
        self.text_display = QTextEdit(self)
        self.layout.addWidget(self.text_display)

        self.central_widget.setLayout(self.layout)

        # 设置初始文本内容
        self.text_display.setPlainText("This is the text content that goes here.\nYou can type and display text in this area.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TextDisplayApp()
    window.show()
    sys.exit(app.exec_())

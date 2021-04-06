import os
import sys

import matplotlib.pyplot as plt
from PySide6.QtCore import Signal, Qt, QObject, QThread, QUrl, QSize
from PySide6.QtGui import QDesktopServices, QPixmap, QMovie
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QFileDialog, QPushButton
import tensorflow as tf

from style_transfer import magenta_v1_256_2
from sizes import *

class Worker(QObject):
    def __init__(self, window, *args, **kwargs):
        super().__init__()
        self.window = window
        self.args = args
        self.kwargs = kwargs
    finished = Signal()

    def run(self):
        save_path = 'outputs/tmp.png'
        image = self.window.model(self.window.content_image.image, self.window.style_image.image,
                                  *self.args, **self.kwargs)
        if not os.path.exists('outputs/'):
            os.mkdir('outputs')
        tf.keras.preprocessing.image.save_img(save_path, image)
        self.window.generated_image.open_image(save_path)
        self.finished.emit()


class ImageLabel(QLabel):
    def __init__(self, window, text):
        super().__init__(window)

        self.setAlignment(Qt.AlignCenter)
        self.setText(text)
        self.setStyleSheet('''
            QLabel{
                border: 1px solid #000
            }
        ''')
        self.setScaledContents(True)

    def setPixmap(self, image):
        super().setPixmap(image)


class Image():
    def __init__(self, window):
        self.window = window
        self.image = None
        self.pixmap = None

    def open_image(self, path=None):
        if not path:
            self.path = QFileDialog.getOpenFileName(self.window, self.window.tr('Open File'), filter=self.window.tr("Image Files (*.png *.jpg *.bmp)"))[0]
        else:
            self.path = path
        self.path = os.path.abspath(self.path).replace('\\', '/')
        self.image = plt.imread(self.path)
        self.original_res = self.image.shape[:-1]
        self.pixmap = QPixmap(self.path)
        self.shape = self.image.shape
        self.label.setPixmap(self.pixmap)

        self.label.mousePressEvent = lambda unnecessary_thing: QDesktopServices.openUrl(QUrl(self.path, QUrl.TolerantMode))

        if self.window.content_image.image is not None and self.window.style_image.image is not None and not self.window.generating:
            self.window.b_generate.setEnabled(True)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Style Transfer')
        self.setGeometry(200, 200, MAIN_W_WIDTH, MAIN_W_HEIGHT)

        self.content_image = Image(self)
        self.style_image = Image(self)
        self.generated_image = Image(self)

        self.b_load_content = QPushButton(self)
        self.b_load_content.setText('Load Content')
        self.b_load_content.clicked.connect(self.content_image.open_image)

        self.b_load_style = QPushButton(self)
        self.b_load_style.setText('Load Style')
        self.b_load_style.clicked.connect(self.style_image.open_image)

        self.b_generate = QPushButton(self)
        self.b_generate.setText('Generate')
        self.b_generate.clicked.connect(self.generate)
        self.b_generate.setEnabled(False)

        self.l_content = ImageLabel(self, 'Content Image')
        self.l_style = ImageLabel(self, 'Style Image')
        self.l_generated = ImageLabel(self, '')

        self.content_image.label = self.l_content
        self.style_image.label = self.l_style
        self.generated_image.label = self.l_generated

        self.l_loading = QLabel(self)
        self.loading_animation = QMovie("resources/loading.gif")
        self.l_loading.setMovie(self.loading_animation)

        self.generating = False

        self.resizeEvent(None)

        self.model = magenta_v1_256_2

    def startAnimation(self):
        self.l_loading.show()
        self.loading_animation.start()

    def stopAnimation(self):
        self.l_loading.hide()
        self.loading_animation.stop()

    def handle_start_generating(self):
        self.generating = True
        self.b_generate.setEnabled(False)

    def handle_stop_generating(self):
        self.stopAnimation()
        self.generating = False
        self.b_generate.setEnabled(True)

    def generate(self):
        self.handle_start_generating()

        self.thread = QThread()
        self.worker = Worker(self)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.startAnimation)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

        self.thread.finished.connect(self.handle_stop_generating)


    def resize_content_input(self, x, y, img_width, img_height):
        self.b_load_content.setGeometry(x, y, B_WIDTH, B_HEIGHT)

        tmp_y = y + B_HEIGHT + B_IMG_height
        self.l_content.setGeometry(x, tmp_y, img_width, img_height)
        if self.content_image.pixmap:
            self.l_content.setPixmap(self.content_image.pixmap)
        return x + img_width, tmp_y + img_height

    def resize_style_input(self, x, y, img_width, img_height):
        self.b_load_style.setGeometry(x, y, B_WIDTH, B_HEIGHT)

        tmp_y = y + B_HEIGHT + B_IMG_height
        self.l_style.setGeometry(x, tmp_y, img_width, img_height)
        if self.style_image.pixmap:
            self.l_style.setPixmap(self.content_image.pixmap)
        return x + img_width, tmp_y + img_height

    def resize_inputs_V(self, x, y, width, height):
        img_size = min((height - 2 * B_IMG_height - IMG_B_height - 2 * B_HEIGHT) // 2,
                        width)
        _, y = self.resize_content_input(x, y, img_size, img_size)
        self.resize_style_input(x, y + IMG_B_height, img_size, img_size)

    def resize_inputs_H(self, x, y, width, height):
        img_size = min((width - IMG_IMG_WIDTH) // 2,
                        height)
        x, _ = self.resize_content_input(x, y, img_size, img_size)
        self.resize_style_input(x + IMG_IMG_WIDTH, y, img_size, img_size)

    def resize_output(self, x, y, width, height):
        output_img_size = min((height - B_IMG_height - B_HEIGHT),
                                  width)
        self.b_generate.setGeometry(x, y, B_WIDTH, B_HEIGHT)
        tmp_x = x + B_WIDTH + B_B_width
        self.l_loading.setGeometry(tmp_x, y, B_HEIGHT, B_HEIGHT)
        self.loading_animation.setScaledSize(QSize(B_HEIGHT, B_HEIGHT))

        tmp_y = y + B_HEIGHT + B_B_height
        self.l_generated.setGeometry(x, tmp_y, output_img_size, output_img_size)
        if self.style_image.pixmap:
            self.l_style.setPixmap(self.style_image.pixmap)

    def resizeEvent(self, event):
        main_w_width = self.width()
        main_w_height = self.height()

        main_width_margin = min(main_w_width // 70, MAIN_WIDTH_MARGIN)
        main_height_margin = min(main_w_width // 70, MAIN_HEIGHT_MARGIN)

        if main_w_width >= main_w_height:
            inputs_width = main_w_width // 4
            inputs_height = main_w_height - 2 * main_height_margin
            self.resize_inputs_V(main_width_margin,
                                 main_height_margin,
                                 inputs_width,
                                 inputs_height)
            inputs_output_width = main_w_width // 5 - 2 * main_width_margin
            output_width = main_w_width - inputs_width - inputs_output_width - 2 * main_width_margin
            self.resize_output(main_width_margin + inputs_width + inputs_output_width,
                               main_height_margin,
                               output_width,
                               main_w_height - 2 * main_height_margin)

        else:
            inputs_height = main_w_height // 4
            inputs_width = main_w_width - 2 * main_width_margin
            self.resize_inputs_H(main_width_margin,
                                 main_height_margin,
                                 inputs_width,
                                 inputs_height)
            inputs_output_height = main_w_height // 8 - 2 * main_height_margin
            output_height = main_w_height - inputs_height - inputs_output_height - 2 * main_height_margin
            self.resize_output(main_width_margin,
                               main_height_margin + inputs_height + inputs_output_height,
                               main_w_width - 2 * main_width_margin,
                               output_height)


app = QApplication(sys.argv)
main_window = MainWindow()
main_window.show()
sys.exit(app.exec_())
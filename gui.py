import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtCore import QLine, Signal, Qt, QObject, QThread, QUrl, QSize
from PySide6.QtGui import QDesktopServices, QPixmap, QMovie
from PySide6.QtWidgets import QApplication, QFormLayout, QLabel, QLineEdit, QMainWindow, QFileDialog, QPushButton, QWidget
import tensorflow as tf

from style_transfer import magenta_v1_256_2
from config import *
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
        image = self.window.model(*self.args, **self.kwargs)
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
    def __init__(self, window, img_type):
        self.window = window
        self.type = img_type
        self.image = None
        self.pixmap = None

    def open_image(self, path=None):
        if not path:
            self.path = QFileDialog.getOpenFileName(self.window, self.window.tr('Open File'), filter=self.window.tr("Image Files (*.png *.jpg *.bmp)"))[0]
            if self.path == '':
                return
        else:
            self.path = path

        self.path = os.path.abspath(self.path).replace('\\', '/')
        self.image = plt.imread(self.path)
        self.original_res = self.image.shape[:-1]
        self.image = self.image.astype(np.float32)[np.newaxis, ...] / 255.
        if self.type == 'content':
            self.res = DEFAULT_CONTENT_RESOLUTION
        elif self.type == 'style':
            self.res = DEFAULT_STYLE_RESOLUTION
        if self.type != 'generated':
            self.resize_image(*self.res)
            self.b_resize.clicked.connect(self.open_resizing_window)
            self.b_resize.show()

        self.pixmap = QPixmap(self.path)
        self.shape = self.image.shape
        self.l_img.setPixmap(self.pixmap)

        self.l_img.mousePressEvent = lambda _: QDesktopServices.openUrl(QUrl(self.path, QUrl.TolerantMode))

        if self.window.content_image.image is not None and self.window.style_image.image is not None and not self.window.generating:
            self.window.b_generate.setEnabled(True)

    def resize_image(self, x, y):
        self.image = tf.image.resize(self.image, (x, y))
        self.res = (x, y)
        self.l_res.setText(f'{x}x{y}')

    def open_resizing_window(self):
        self.resizing_window = ResizeImageWindow(self)
        self.resizing_window.show()


class ResizeImageWindow(QWidget):
    def __init__(self, image):
        super().__init__()
        self.image = image
        self.setWindowTitle(f'Resize {self.image.type}')
        self.setGeometry(300, 300, 250, 120)

        self.layout = QFormLayout()
        self.x_value = QLineEdit()
        self.x_value.setText(f'{self.image.res[0]}')
        self.y_value = QLineEdit()
        self.y_value.setText(f'{self.image.res[1]}')

        self.b_confirm = QPushButton()
        self.b_confirm.setText('Confirm')
        self.b_confirm.clicked.connect(self.resize_image)

        self.layout.addRow(QLabel(f'Original size: {self.image.original_res[0]}x{self.image.original_res[1]}'))
        self.layout.addRow(QLabel('X:'), self.x_value)
        self.layout.addRow(QLabel('Y:'), self.y_value)
        self.layout.addRow(self.b_confirm)
        self.setLayout(self.layout)

    def resize_image(self):
        self.image.res = (int(self.x_value.text()), int(self.y_value.text()))
        self.image.resize_image(*self.image.res)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Style Transfer')
        self.setGeometry(100, 100, MAIN_W_WIDTH, MAIN_W_HEIGHT)

        self.content_image = Image(self, 'content')
        self.style_image = Image(self, 'style')
        self.generated_image = Image(self, 'generated')

        self.b_load_content = QPushButton(self)
        self.b_load_content.setText('Load Content')
        self.b_load_content.clicked.connect(self.content_image.open_image)

        self.l_content_res = QLabel(self)

        self.b_resize_content = QPushButton(self)
        self.b_resize_content.setText('Change')
        self.b_resize_content.hide()

        self.b_load_style = QPushButton(self)
        self.b_load_style.setText('Load Style')
        self.b_load_style.clicked.connect(self.style_image.open_image)

        self.l_style_res = QLabel(self)

        self.b_resize_style = QPushButton(self)
        self.b_resize_style.setText('Change')
        self.b_resize_style.hide()

        self.b_generate = QPushButton(self)
        self.b_generate.setText('Generate')
        self.b_generate.clicked.connect(self.generate)
        self.b_generate.setEnabled(False)

        self.l_content = ImageLabel(self, 'Content Image')
        self.l_style = ImageLabel(self, 'Style Image')
        self.l_generated = ImageLabel(self, '')

        self.content_image.l_img = self.l_content
        self.content_image.l_res = self.l_content_res
        self.content_image.b_resize = self.b_resize_content
        self.style_image.l_img = self.l_style
        self.style_image.l_res = self.l_style_res
        self.style_image.b_resize = self.b_resize_style
        self.generated_image.l_img = self.l_generated

        self.l_loading = QLabel(self)
        self.loading_animation = QMovie("resources/loading.gif")
        self.l_loading.setMovie(self.loading_animation)

        self.generating = False

        self.resizeEvent(None)

        self.model = magenta_v1_256_2

    def start_animation(self):
        self.l_loading.show()
        self.loading_animation.start()

    def stop_animation(self):
        self.l_loading.hide()
        self.loading_animation.stop()

    def handle_start_generating(self):
        self.start_animation()
        self.generating = True
        self.b_generate.setEnabled(False)

    def handle_stop_generating(self):
        self.stop_animation()
        self.generating = False
        self.b_generate.setEnabled(True)

    def generate(self):
        self.handle_start_generating()

        self.thread = QThread()
        self.worker = Worker(self, self.content_image.image, self.style_image.image)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

        self.thread.started.connect(self.handle_stop_generating)


    def resize_content_input(self, x, y, img_width, img_height):
        self.b_load_content.setGeometry(x, y, B_WIDTH, B_HEIGHT)

        tmp_x = x + B_WIDTH + B_B_width
        self.l_content_res.setGeometry(tmp_x, y, L_RES_WIDTH, B_HEIGHT)

        tmp_x += L_RES_WIDTH + L_RES_B_WIDTH
        self.b_resize_content.setGeometry(tmp_x, y, B_WIDTH, B_HEIGHT)

        tmp_y = y + B_HEIGHT + B_IMG_height
        self.l_content.setGeometry(x, tmp_y, img_width, img_height)
        if self.content_image.pixmap:
            self.l_content.setPixmap(self.content_image.pixmap)
        return x + img_width, tmp_y + img_height

    def resize_style_input(self, x, y, img_width, img_height):
        self.b_load_style.setGeometry(x, y, B_WIDTH, B_HEIGHT)

        tmp_x = x + B_WIDTH + B_B_width
        self.l_style_res.setGeometry(tmp_x, y, L_RES_WIDTH, B_HEIGHT)

        tmp_x += L_RES_WIDTH + L_RES_B_WIDTH
        self.b_resize_style.setGeometry(tmp_x, y, B_WIDTH, B_HEIGHT)

        tmp_y = y + B_HEIGHT + B_IMG_height
        self.l_style.setGeometry(x, tmp_y, img_width, img_height)
        if self.style_image.pixmap:
            self.l_style.setPixmap(self.style_image.pixmap)
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
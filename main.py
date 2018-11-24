#!/usr/bin/env python

"""
@author: Jafar Jabr <jafaronly@yahoo.com>
=======================================
All what is related to Ai, Pytorch and image processing are from
https://github.com/pytorch/examples/tree/master/fast_neural_style
"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QDialog, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, \
    QApplication, QFileDialog, QButtonGroup, QRadioButton, QMessageBox, QGroupBox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import re
import torch
from torchvision import transforms
import torch.onnx
import models.utils as utils
from models.transformer_net import TransformerNet


class Dialog(QDialog):
    def __init__(self, parent=None):
        super(Dialog, self).__init__(parent)
        self.show_file_browser = True
        self.img_url = ""
        self.trained_models = {1: "trained_models/candy.pth", 2: "trained_models/mosaic.pth", 3: "trained_models/rain_princess.pth", 4:"trained_models/udnie.pth"}
        self.selected_model = 1
        layout = QVBoxLayout(self)
        label0 = QLabel("Styles: ")
        layout.addWidget(label0)

        selection_line = QHBoxLayout()

        self.style_group = QButtonGroup()  # Number group

        r1 = QRadioButton("")
        r1.setChecked(True)
        r1.toggled.connect(lambda: self.set_model(r1, 1))
        r2 = QRadioButton("")
        r2.toggled.connect(lambda: self.set_model(r2, 2))
        r3 = QRadioButton("")
        r3.toggled.connect(lambda: self.set_model(r3, 3))
        r4 = QRadioButton("")
        r4.toggled.connect(lambda: self.set_model(r4, 4))

        self.style_group.addButton(r1)
        self.style_group.addButton(r2)
        self.style_group.addButton(r3)
        self.style_group.addButton(r4)

        groupBox1 = QGroupBox("Candy")
        label1 = QLabel()
        pixmap1 = QPixmap('images/styles/candy.jpg')
        pixmap_resized1 = pixmap1.scaled(100, 105, Qt.KeepAspectRatio)
        label1.setPixmap(pixmap_resized1)

        vbox1 = QVBoxLayout()
        vbox1.addWidget(r1)
        vbox1.addWidget(label1)
        groupBox1.setLayout(vbox1)

        groupBox2 = QGroupBox("Mosaic")
        label2 = QLabel()
        pixmap2 = QPixmap('images/styles/mosaic.jpg')
        pixmap_resized2 = pixmap2.scaled(100, 105, Qt.KeepAspectRatio)
        label2.setPixmap(pixmap_resized2)

        vbox2 = QVBoxLayout()
        vbox2.addWidget(r2)
        vbox2.addWidget(label2)
        groupBox2.setLayout(vbox2)

        groupBox3 = QGroupBox("Rain Princess")
        label3 = QLabel()
        pixmap3 = QPixmap('images/styles/rain-princess.jpg')
        pixmap_resized3 = pixmap3.scaled(100, 105, Qt.KeepAspectRatio)
        label3.setPixmap(pixmap_resized3)

        vbox3 = QVBoxLayout()
        vbox3.addWidget(r3)
        vbox3.addWidget(label3)
        groupBox3.setLayout(vbox3)

        groupBox4 = QGroupBox("udnie")
        label4 = QLabel()
        pixmap4 = QPixmap('images/styles/udnie.jpg')
        pixmap_resized4 = pixmap4.scaled(100, 105, Qt.KeepAspectRatio)
        label4.setPixmap(pixmap_resized4)

        vbox4 = QVBoxLayout()
        vbox4.addWidget(r4)
        vbox4.addWidget(label4)
        groupBox4.setLayout(vbox4)

        selection_line.addWidget(groupBox1)
        selection_line.addWidget(groupBox2)
        selection_line.addWidget(groupBox3)
        selection_line.addWidget(groupBox4)
        layout.addLayout(selection_line)

        self.figure = plt.figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        btn1 = QPushButton('Input Image')
        btn1.clicked.connect(lambda me: self.handle_main_btn())
        layout.addWidget(btn1)

        btn2 = QPushButton('Apply Style')
        btn2.clicked.connect(lambda me: self.stylize())
        layout.addWidget(btn2)

        self.resize(1000, 800)
        self.show_image()

    def set_model(self, instance, model_index):
        if instance.isChecked():
            self.selected_model = model_index

    def handle_main_btn(self):
        if self.show_file_browser:
            self.open_file()
        else:
            self.show_image(self.img_url)
            self.show_file_browser = True

    def stylize(self):
        if len(self.img_url) < 2:
            QMessageBox.warning(self, 'خطأ', 'لم يتم اختيار اي صورة')
        else:
            self.show_file_browser = False
            content_image = self.img_url
            model = self.trained_models[self.selected_model]
            output_image = 'images/output.png'
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            content_image = utils.load_image(content_image, scale=None)
            content_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.mul(255))
            ])
            content_image = content_transform(content_image)
            content_image = content_image.unsqueeze(0).to(device)
            with torch.no_grad():
                style_model = TransformerNet()
                state_dict = torch.load(model)
                # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
                for k in list(state_dict.keys()):
                    if re.search(r'in\d+\.running_(mean|var)$', k):
                        del state_dict[k]
                style_model.load_state_dict(state_dict)
                style_model.to(device)
                output = style_model(content_image).cpu()
            utils.save_image(output_image, output[0])
            self.show_image(output_image)

    def show_image(self, img_path="images/no_image.png"):
        plt.close('all')
        self.figure.clear()
        # create an axis
        ax = self.figure.add_subplot(111)
        ax.set_title('Original Image')
        # discards the old graph
        ax.clear()
        dd = plt.imread(img_path)
        # plot data
        ax.imshow(dd)
        # refresh canvas
        self.canvas.draw()

    def open_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        img_path, _ = QFileDialog.getOpenFileName(None, "choose an image", "", "Image Files (*.jpg *.png)", options=options)
        if img_path:
            self.img_url = img_path
            self.show_image(img_path)


if __name__ == '__main__':
    app = QApplication([])
    my_dialog = Dialog(None)
    my_dialog.exec_()

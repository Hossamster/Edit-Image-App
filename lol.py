from PyQt5 import uic
from PyQt5.QtWidgets import QApplication,QComboBox, QColorDialog,QLineEdit, QLabel, QPushButton, QVBoxLayout, QWidget,QRadioButton, QFileDialog, QGridLayout, QMainWindow,QDialog,QSpinBox,QFormLayout,QHBoxLayout
from PyQt5.QtGui import QPixmap,QCursor,QColor
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QLabel, QComboBox, QSpinBox, QVBoxLayout, QDialogButtonBox
from PyQt5.QtWidgets import QSlider, QVBoxLayout, QHBoxLayout, QLabel, QWidget


# Load the UI files for the button window and main window
button_ui_file = 'mybutton.ui'
main_ui_file = 'main.ui'
class MainApp(QMainWindow):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)

        # Load the UI file for the main window
        uic.loadUi(main_ui_file, self)

        # Connect resize button to before_resize_function
        self.brightness.clicked.connect(self.brightness_function)
        
    def brightness_function(self):
            # Get the path of the image file
            img_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
            if not img_path:
                return

            # Load the image
            img = cv2.imread(img_path).astype(np.float32)/255
            img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            
            gamma = 0.4 # less than 1 means brighter.
            result = np.power(img_rgb,gamma)
            plt.figure(figsize = (12,8))
            plt.subplot(1,2,1)
            plt.imshow(img_rgb)
            plt.title('Original image',fontsize = 17)
            plt.subplot(1,2,2)
            plt.imshow(result)
            plt.title('Result',fontsize = 17)
            plt.suptitle(f'Gamma Correction with value {gamma}',fontsize = 25)
            plt.show()
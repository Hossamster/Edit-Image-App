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
import seaborn as sns

# Load the UI files for the button window and main window
button_ui_file = 'mybutton.ui'
main_ui_file = 'main.ui'

class Histogram_equalizationDialog(QDialog):
    def __init__(self, parent=None):
        super(Histogram_equalizationDialog, self).__init__(parent)
        self.setWindowTitle("Choose :")
        self.setFixedSize(200, 150)  # Set a fixed size for the dialog

        # Create the UI elements
        self.comboBox = QComboBox()
        self.comboBox.addItems(["Gray","RGB"])
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.close)  # Connect the rejected signal to the close slot of the dialog

        # Lay out the UI elements
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Select :"))
        layout.addWidget(self.comboBox)
        layout.addWidget(self.buttonBox)
        self.setLayout(layout)

    def getSelectedTechnique(self):
        # Return the selected technique
        return self.comboBox.currentText()
class HistogramDialog(QDialog):
    def __init__(self, parent=None):
        super(HistogramDialog, self).__init__(parent)
        self.setWindowTitle("Choose : ")
        self.setFixedSize(200, 110)  # Set a fixed size for the dialog

        # Create the UI elements
        self.comboBox = QComboBox()
        self.comboBox.addItems(["Calculate Histogram","Histogram Equalization"])
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.close)  # Connect the rejected signal to the close slot of the dialog

        # Lay out the UI elements
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Select : "))
        layout.addWidget(self.comboBox)
        layout.addWidget(self.buttonBox)
        self.setLayout(layout)

    def getSelectedTechnique(self):
        # Return the selected technique
        return self.comboBox.currentText()
class MainApp(QMainWindow):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)

        # Load the UI file for the main window
        uic.loadUi(main_ui_file, self)        
        self.histogram.clicked.connect(self.histogram_function)
    

    def histogram_function(self):
        # Get the path of the image file
        img_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if not img_path:
            return

        # Load the image
        img = cv2.imread(img_path)
        # convert the image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # convert the image to grayscale
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        

        # Show the Gradient technique dialog
        dialog = HistogramDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            # Apply the selected 
            technique = dialog.getSelectedTechnique()
            if technique == "Calculate Histogram":
                # flatten the image
                flattened_img_gray = img_gray.ravel()

                # create the histogram using seaborn
                plt.figure(figsize = (16,10))
                plt.subplot(2,2,1)
                plt.imshow(img_rgb)
                plt.title('RGB image')
                plt.subplot(2,2,2)
                plt.title('Gray Scale')
                plt.imshow(img_gray,'gray')
                plt.subplot(2,2,3)
                plt.title('Histogram of RGB image')
                # define the list of channel colors
                colors = ['blue', 'green', 'red']
                # plot the kde plot for each color channel
                for i, col in enumerate(colors):
                    channel = img[:, :, i].ravel()  # flatten the channel values into a 1D array
                    sns.kdeplot(channel, color=col, shade=True, alpha=0.5, label=f'{col.upper()} channel')
                
                plt.subplot(2,2,4)
                plt.title('Histogram of Gray image')
                sns.distplot(flattened_img_gray, kde=False ,rug=False)
                plt.tight_layout()
                plt.show()
            
            
            elif technique == "Histogram Equalization":
                dialog = Histogram_equalizationDialog(self)
                if dialog.exec_() == QDialog.Accepted:
                    # Apply the selected  technique
                    technique = dialog.getSelectedTechnique()
                    if technique == "Gray":
                        # flatten the image
                        flattened_img = img_gray.ravel()

                        # create the histogram using seaborn
                        plt.figure(12,8)
                        plt.subplot(1,4,1)
                        plt.imshow(img_gray)
                        plt.title('Gray Image')
                        plt.subplot(1,4,2)
                        sns.distplot(flattened_img, kde=False ,rug=False)
                        plt.title('Histogram of Gray image')
                        
                        hist_equalization_img = cv2.equalizeHist(img_gray)
                        plt.subplot(1,4,3)
                        plt.title('Histogram equalization result')
                        plt.imshow(hist_equalization_img,'gray')
                        plt.subplot(1,4,4)
                        plt.title('Histogram after histogram equalization')
                        flattened_img = hist_equalization_img.ravel()
                        sns.distplot(flattened_img, kde=False ,rug=False)
                        plt.show()
                    elif technique == "RGB":
                        # first we should convert it to hsv color space 
                        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                        # Grab V channel
                        hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
                        hist_equalization_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                        
                        plt.figure(12,8)
                        plt.subplot(2,2,1)
                        plt.imshow(img_rgb)
                        plt.title('RGB image')
                        
                        plt.subplot(2,2,2)
                        # define the list of channel colors
                        colors = ['blue', 'green', 'red']
                        # plot the kde plot for each color channel
                        for i, col in enumerate(colors):
                            channel = img[:, :, i].ravel()  # flatten the channel values into a 1D array
                            sns.kdeplot(channel, color=col, shade=True, alpha=0.5, label=f'{col.upper()} channel')
                            
                        plt.title('Histogram of RGB')
                        
                        plt.subplot(2,2,3)
                        plt.imshow(hist_equalization_img)
                        plt.title('Image after histogram equalization')
                        
                        plt.subplot(2,2,4)
                        # define the list of channel colors
                        colors = ['blue', 'green', 'red']
                        # plot the kde plot for each color channel
                        for i, col in enumerate(colors):
                            channel = hist_equalization_img[:, :, i].ravel()  # flatten the channel values into a 1D array
                            sns.kdeplot(channel, color=col, shade=True, alpha=0.5, label=f'{col.upper()} channel')
                        plt.title('Histogram of the new image')
                        plt.tight_layout()
                        plt.show()
                    else: return
        else:
            # User pressed "Cancel", don't perform any blurring
            return
        
        # Display the result
        plt.figure(figsize = (10,5))
        plt.subplot(1,2,1)
        plt.imshow(img_rgb)
        plt.title('Original image',fontsize = 17)
        plt.subplot(1,2,2)
        plt.imshow(img)
        plt.title('Result',fontsize = 17)
        plt.show()





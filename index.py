from PyQt5 import uic
from PyQt5.QtWidgets import QApplication,QDoubleSpinBox,QComboBox, QColorDialog,QLineEdit, QLabel, QPushButton, QVBoxLayout, QWidget,QRadioButton, QFileDialog, QGridLayout, QMainWindow,QDialog,QSpinBox,QFormLayout,QHBoxLayout
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


def display_img(img):
    fig = plt.figure(figsize = (15,15))
    ax = fig.add_subplot(111)
    ax.imshow(img)
    plt.show()
    
class ButtonWindow(QMainWindow):
    def __init__(self, parent=None):
        super(ButtonWindow, self).__init__(parent)

        # Load the UI file for the button window
        uic.loadUi(button_ui_file, self)

        self.pushButton.clicked.connect(self.show_main_window)

    def show_main_window(self):
        self.hide()
        self.main_window = MainApp()
        self.main_window.setWindowTitle("Main Window")
        self.main_window.show()

class ResizeDialog(QDialog):
    def __init__(self, parent=None):
        super(ResizeDialog, self).__init__(parent)

        # Create the spin boxes
        self.width_spinbox = QSpinBox()
        self.width_spinbox.setRange(1, 10000)
        self.width_spinbox.setValue(640)
        self.height_spinbox = QSpinBox()
        self.height_spinbox.setRange(1, 10000)
        self.height_spinbox.setValue(480)

        # Create the label
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)

        # Create the layout
        layout = QFormLayout()
        layout.addRow("Width:", self.width_spinbox)
        layout.addRow("Height:", self.height_spinbox)
        layout.addRow(self.label)

        # Create the buttons
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        # Create the button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        # Add the layouts to the dialog
        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

    def get_width(self):
        return self.width_spinbox.value()

    def get_height(self):
        return self.height_spinbox.value()

    def set_label_text(self, text):
        self.label.setText(text)


class ThresholdDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Set up the widgets
        self.threshold_label = QLabel("Select threshold type:")
        self.threshold_combo = QComboBox()
        self.threshold_combo.addItems(["cv2.THRESH_BINARY", "cv2.THRESH_BINARY_INV", "cv2.THRESH_TRUNC", "cv2.THRESH_TOZERO", "cv2.THRESH_TOZERO_INV", "Adaptive threshold","Compare all"])
        self.threshold_value_label = QLabel("Enter threshold value:")
        self.threshold_value_spinbox = QSpinBox()
        self.threshold_value_spinbox.setMinimum(0)
        self.threshold_value_spinbox.setMaximum(255)
        self.threshold_value_spinbox.setValue(128)

        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(self.threshold_label)
        layout.addWidget(self.threshold_combo)
        layout.addWidget(self.threshold_value_label)
        layout.addWidget(self.threshold_value_spinbox)

        # Set up the button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def get_threshold_type(self):
        return self.threshold_combo.currentText()

    def get_threshold_value(self):
        return self.threshold_value_spinbox.value()
    
class MainApp(QMainWindow):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)

        # Load the UI file for the main window
        uic.loadUi(main_ui_file, self)

        # Connect resize button to before_resize_function
        self.resize.clicked.connect(self.before_resize_function)
        self.flip.clicked.connect(self.before_flip_function)
        self.draw_rectangle.clicked.connect(self.before_draw_rectangle)
        self.gray_scale.clicked.connect(self.gray_scale_function)
        self.hsv.clicked.connect(self.hsv_function)
        self.blend2imgs.clicked.connect(self.blend2imgs_function)
        self.blending_diff_size.clicked.connect(self.blending_diff_size_function)
        self.threshold.clicked.connect(self.threshold_function)
        self.brightness.clicked.connect(self.brightness_function)
        
    def before_resize_function(self):
        img_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if not img_path:
            return
        # Get the size of the selected image
        img = cv2.imread(img_path)
        size = "{}x{}".format(img.shape[1], img.shape[0])
        # Create the resize dialog
        dialog = ResizeDialog(self)
        dialog.setWindowTitle("Resize Image")
        dialog.setFixedSize(200, 160)
        dialog.set_label_text("Selected image size: {}".format(size))
        result = dialog.exec_()


        # If the user clicked OK, get the width and height and call resize_function
        if result == QDialog.Accepted:
            width = dialog.get_width()
            height = dialog.get_height()
            self.resize_function(width, height,img_path)

    def resize_function(self, width, height,img_path):
        # Get the path of the selected image
        # img_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        # if not img_path:
        #     return

        # Load the image and resize it
        img = cv2.imread(img_path)
        img = (cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        resized_img = cv2.resize(img, (width, height))

        # Display the original and resized images
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(img)
        ax1.set_title('Original Image')
        ax2.imshow(resized_img)
        ax2.set_title('Resized Image')
        ax1.axis('off')
        ax2.axis('off')
        plt.tight_layout()
        plt.show()

    def before_flip_function(self):
        # Create a QDialog to get input from the user
        flip_dialog = QDialog()
        flip_dialog.setWindowTitle("Choose flip direction")
        flip_dialog.resize(250, 100)

        # Create radio buttons for vertical and horizontal flip
        v_flip_radio = QRadioButton("Vertical")
        h_flip_radio = QRadioButton("Horizontal")
        v_flip_radio.setChecked(True)

        # Create OK button to confirm flip direction
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(lambda: (self.flip_function(v_flip_radio.isChecked()), flip_dialog.accept()))

        # Add radio buttons and OK button to a layout
        layout = QVBoxLayout()
        layout.addWidget(v_flip_radio)
        layout.addWidget(h_flip_radio)
        layout.addWidget(ok_button)

        # Set the layout and show the dialog
        flip_dialog.setLayout(layout)
        flip_dialog.exec_()

    
    def flip_function(self, is_vertical):
        flip_code = 0 if is_vertical else 1
        
        # Set flip code based on radio button selection
        img_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if not img_path:
            return

        # Load the image 
        img = cv2.imread(img_path)
        img = (cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


        # Flip the image
        flipped_img = cv2.flip(img, flip_code)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(img)
        ax1.set_title('Original Image')
        ax2.imshow(flipped_img)
        ax2.set_title('flipped Image')
        ax1.axis('off')
        ax2.axis('off')
        plt.tight_layout()
        plt.show()
        


    def before_draw_rectangle(self):
        # Get the path of the image file
        img_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if not img_path:
            return

        # Load the image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get the size of the image
        img_size = "{} x {}".format(img.shape[1], img.shape[0])

        # Create a QDialog to get input from the user
        rect_dialog = QDialog()
        rect_dialog.setWindowTitle("Enter coordinates")
        rect_dialog.resize(250, 250)

        # Create line edits for the coordinates
        x1_edit = QLineEdit()
        y1_edit = QLineEdit()
        x2_edit = QLineEdit()
        y2_edit = QLineEdit()

        # Create a combo box for color selection
        color_combo = QComboBox()
        color_combo.addItems(["Blue", "Red", "White", "Green", "Custom"])

        # Create OK button to confirm coordinates
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(lambda: self.draw_rectangle_function(img, x1_edit.text(), y1_edit.text(), x2_edit.text(), y2_edit.text(), color_combo.currentText()))
        ok_button.clicked.connect(rect_dialog.accept)  # Add this line to close the dialog on OK button click

        # Add labels, line edits, color combo box, and OK button to a layout
        layout = QGridLayout()
        layout.addWidget(QLabel("Point 1 (x,y)"), 0, 0)
        layout.addWidget(x1_edit, 0, 1)
        layout.addWidget(y1_edit, 0, 2)
        layout.addWidget(QLabel("Point 2 (x,y)"), 1, 0)
        layout.addWidget(x2_edit, 1, 1)
        layout.addWidget(y2_edit, 1, 2)
        layout.addWidget(color_combo, 2, 0, 1, 3)
        layout.addWidget(QLabel("Image Size: {}".format(img_size)), 3, 0, 1, 3)
        layout.addWidget(ok_button, 4, 1)

        # Set the layout and show the dialog
        rect_dialog.setLayout(layout)
        rect_dialog.exec_()

    def draw_rectangle_function(self, img, x1, y1, x2, y2, color):
        # Convert string inputs to integer values
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Get the color for the rectangle
        if color == "Blue":
            rect_color = (255, 0, 0)  # Blue
        elif color == "Red":
            rect_color = (0, 0, 255)  # Red
        elif color == "White":
            rect_color = (255, 255, 255)  # White
        elif color == "Green":
            rect_color = (0, 255, 0)  # Green
        elif color == "Custom":
            # Open a color picker dialog to choose a custom color
            custom_color = QColorDialog.getColor()
            if custom_color.isValid():
                
                rect_color = custom_color.getRgb()[:3]
            else:
            # If the user cancels the color picker dialog, use white as default
                rect_color = (255, 255, 255)

        # Draw a rectangle on the image
        rect_img = img.copy()
        cv2.rectangle(rect_img, (x1, y1), (x2, y2), rect_color, 2)

        # Show the image
        plt.imshow(rect_img)
        plt.axis('off')
        plt.show()

    def gray_scale_function(self):
        # Get the path of the image file
        img_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if not img_path:
            return

        # Load the image
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(img_rgb)
        ax1.set_title('Original Image')
        ax2.imshow(img_gray,'gray')
        ax2.set_title('GrayImage')
        ax1.axis('off')
        ax2.axis('off')
        plt.tight_layout()
        plt.show()
    
    
    def hsv_function(self):
        # Get the path of the image file
        img_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if not img_path:
            return

        # Load the image
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(img_rgb)
        ax1.set_title('Original Image')
        ax2.imshow(img_hsv)
        ax2.set_title('HSV Image')
        ax1.axis('off')
        ax2.axis('off')
        plt.tight_layout()
        plt.show()
        
    def blend2imgs_function(self):
        # Get the path of the image file
        img1_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if not img1_path:
            return

        # Load the image
        img1 = cv2.imread(img1_path)
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
        
        # Get the path of the image file
        img2_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if not img2_path:
            return

        # Load the image
        img2 = cv2.imread(img2_path)
        img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
        
        # shapes = [(img1.shape[0],img1.shape[1]),(img2.shape[0],img2.shape[1])]
        # max_size = max(shapes, key=lambda x: x[0]*x[1])
        
        # easy to blend if they are equal sizes otherwise it will be a bit trickier
        img1 = cv2.resize(img1,(1200,1200))
        img2 = cv2.resize(img2,(1200,1200))
        blended = cv2.addWeighted(img1,0.91,img2,0.1,0)   
        
        fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(10, 6))
        ax1.imshow(img1)
        ax1.set_title('Original Image')
        ax2.imshow(img2)
        ax2.set_title('HSV Image')
        ax3.imshow(blended)
        ax3.set_title('HSV Image')
        ax1.axis('off')
        ax2.axis('off')
        plt.tight_layout()
        plt.show()
    
    def blending_diff_size_function(self):
        # Get the path of the image file
        img1_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if not img1_path:
            return

        # Load the image
        img1 = cv2.imread(img1_path)
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
        
        # Get the path of the image file
        img2_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if not img2_path:
            return

        # Load the image
        img2 = cv2.imread(img2_path)
        img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
        # size = (int(img2.shape[0] / 2),int(img2.shape[1] / 2))
        img2 = cv2.resize(img2,(600,600))
        x_offset = img1.shape[1] - img2.shape[1]
        y_offset = img1.shape[0] - img2.shape[0]
        rows,cols,channels = img2.shape
        roi = img1[y_offset:img1.shape[0],x_offset:img1.shape[1]]
        
        img2gray = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)
        mask_inv = cv2.bitwise_not(img2gray)
        fg = (cv2.bitwise_or(img2,img2,mask=mask_inv))
        
        final_roi = cv2.bitwise_or(roi,fg)
        large_img = img1
        small_img = final_roi
        img1[y_offset:img1.shape[0],x_offset:img1.shape[1]] = small_img
        # fig, (ax1, ax2,ax3,ax4,ax5,ax6,ax7) = plt.subplots(nrows=2,ncols=4 , figsize=(10, 5))
        plt.figure(figsize=(10, 5))
        plt.subplot(2,4,1)
        plt.imshow(img1)
        plt.title('Image 1')
        plt.subplot(2,4,2)
        plt.imshow(img2)
        plt.title('Image 2')
        plt.subplot(2,4,3)
        plt.imshow(img2gray,'gray')
        plt.title('Image 2 gray')
        
        plt.subplot(2,4,4)
        plt.imshow(mask_inv,cmap = 'gray')
        plt.title('Image 2 after bitwise not')
        plt.subplot(2,4,5)
        plt.imshow(fg,cmap = 'gray')
        plt.title('Image 2 after bitwise or (with not masking)')
        plt.subplot(2,4,6)
        plt.imshow(roi)
        plt.title('Image 1 ROI')
        plt.subplot(2,4,7)
        plt.imshow(img1)
        plt.title('Final ROI')
        plt.tight_layout()
        plt.show()

    

    def threshold_function(self):
        # Get the path of the image file
        img_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if not img_path:
            return

        # Load the image
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Create a QDialog to get input from the user
        dialog = ThresholdDialog(self)
        if dialog.exec_():
            threshold_type = dialog.get_threshold_type()
            threshold_value = dialog.get_threshold_value()

            threshold_types = {
                "cv2.THRESH_BINARY": cv2.THRESH_BINARY,
                "cv2.THRESH_BINARY_INV": cv2.THRESH_BINARY_INV,
                "cv2.THRESH_TRUNC": cv2.THRESH_TRUNC,
                "cv2.THRESH_TOZERO": cv2.THRESH_TOZERO,
                "cv2.THRESH_TOZERO_INV": cv2.THRESH_TOZERO_INV
            }

            # Create a dictionary with the keys and values swapped
            threshold_types_reverse = {v: k for k, v in threshold_types.items()}

            if threshold_type == "Adaptive threshold":
                img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                # Display the thresholded image
                plt.figure(figsize=(10,6))
                plt.subplot(1,3,1)
                plt.imshow(img_rgb)
                plt.title("Original Image")
                plt.subplot(1,3,2)
                plt.imshow(img_gray,'gray')
                plt.title("Gray Image")
                plt.subplot(1,3,3)
                plt.imshow(img_thresh,'gray')
                plt.title(f"Thresholded Image ({threshold_types_reverse[threshold_type]})")
                plt.tight_layout()
                plt.show()
            elif  threshold_type == "Compare all":
                img_thresh_adaptive = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                _, img_thresh_binary = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)
                _, THRESH_BINARY_INV = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
                _, THRESH_TRUNC = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_TRUNC)
                _, THRESH_TOZERO = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_TOZERO)
                _, THRESH_TOZERO_INV = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_TOZERO_INV)
                # Display the thresholded image
                plt.figure(figsize=(16,12))
                plt.subplot(3,3,1)
                plt.imshow(img_rgb)
                plt.title("Original Image")
                plt.subplot(3,3,2)
                plt.imshow(img_gray,'gray')
                plt.title("Gray Image")
                plt.subplot(3,3,3)
                plt.imshow(img_thresh_adaptive,'gray')
                plt.title(f"Thresholded cv2.adaptiveThreshold ")
                plt.subplot(3,3,4)
                plt.imshow(img_thresh_binary,'gray')
                plt.title(f"Thresholded cv2.THRESH_BINARY ")
                plt.subplot(3,3,5)
                plt.imshow(THRESH_BINARY_INV,'gray')
                plt.title(f"Thresholded cv2.THRESH_BINARY_INV ")
                plt.subplot(3,3,6)
                plt.imshow(THRESH_TRUNC,'gray')
                plt.title(f"Thresholded cv2.THRESH_TRUNC ")
                plt.subplot(3,3,7)
                plt.imshow(THRESH_TOZERO,'gray')
                plt.title(f"Thresholded cv2.THRESH_TOZERO ")
                plt.subplot(3,3,8)
                plt.imshow(THRESH_TOZERO_INV,'gray')
                plt.title(f"Thresholded cv2.THRESH_TOZERO_INV ")
                
                plt.tight_layout()
                plt.show()
            else:
                threshold_type = threshold_types[threshold_type]
                _, img_thresh = cv2.threshold(img_gray, threshold_value, 255, threshold_type)

                # Display the thresholded image
                plt.figure(figsize=(10,6))
                plt.subplot(1,3,1)
                plt.imshow(img_rgb)
                plt.title("Original Image")
                plt.subplot(1,3,2)
                plt.imshow(img_gray,'gray')
                plt.title("Gray Image")
                plt.subplot(1,3,3)
                plt.imshow(img_thresh,'gray')
                plt.title(f"Thresholded Image ({threshold_types_reverse[threshold_type]})")
                plt.tight_layout()
                plt.show()

    def brightness_function(self):
        # Get the path of the image file
        img_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if not img_path:
            return

        # Load the image
        img = cv2.imread(img_path).astype(np.float32)/255
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        # Create the slider for gamma adjustment
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setMinimum(1)
        self.gamma_slider.setMaximum(1000)
        self.gamma_slider.setValue(40)
        self.gamma_slider.setTickPosition(QSlider.TicksBelow)
        self.gamma_slider.setTickInterval(10)

        # Create the label for displaying the current value of the slider
        self.gamma_label = QLabel(f"Gamma Value: {self.gamma_slider.value()/100:.2f}")
        self.gamma_label.setAlignment(Qt.AlignCenter)

        # Connect the slider to the function for updating the label
        self.gamma_slider.valueChanged.connect(self.update_gamma_label)

        # Create a spinbox for manual input of gamma value
        self.gamma_spinbox = QDoubleSpinBox()
        self.gamma_spinbox.setMinimum(0.01)
        self.gamma_spinbox.setMaximum(10.00)
        self.gamma_spinbox.setSingleStep(0.01)
        self.gamma_spinbox.setValue(0.40)

        # Connect the spinbox to the function for updating the slider value
        self.gamma_spinbox.valueChanged.connect(self.update_gamma_slider)

        # Create the layout for the slider, label, and spinbox
        slider_layout = QVBoxLayout()
        slider_layout.addWidget(self.gamma_slider)
        slider_layout.addWidget(self.gamma_label)
        slider_layout.addWidget(QLabel("OR"))
        slider_layout.addWidget(QLabel("Manually Enter Gamma Value:"))
        slider_layout.addWidget(self.gamma_spinbox)

        # Create the dialog for displaying the slider, label, and spinbox
        dialog = QDialog(self)
        dialog.setWindowTitle("Adjust Gamma Value")
        dialog.setLayout(slider_layout)

        # Create a horizontal layout for the OK and Cancel buttons
        button_layout = QHBoxLayout()

        # Create an OK button
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(dialog.accept)
        button_layout.addWidget(ok_button)

        # Create a Cancel button
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_button)

        # Add the button layout to the main layout
        slider_layout.addLayout(button_layout)

        # Display the dialog and wait for it to be closed
        if dialog.exec_() == QDialog.Accepted:
            gamma = self.gamma_slider.value() / 100 if self.gamma_slider.isEnabled() else self.gamma_spinbox.value()
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

    def update_gamma_label(self, value):
        self.gamma_label.setText(f"Gamma Value: {value/100:.2f}")

    def update_gamma_slider(self, value):
        self.gamma_spinbox.setValue(value )
        self.gamma_slider.setValue(int(value * 100))
        self.update_gamma_label(value *100)  # Update the gamma label with the current slider value



if __name__ == '__main__':
    app = QApplication(sys.argv)
    button_window = ButtonWindow()
    button_window.show()
    sys.exit(app.exec_())

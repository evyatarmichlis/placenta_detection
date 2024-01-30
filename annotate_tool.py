from datetime import datetime
from tkinter import messagebox
import cv2
import numpy as np
import uploader.uploader
from consts import Consts, Folders


class ImageAnnotation:
    def __init__(self, image,date = None):
        self.img = cv2.resize(image, (640, 480))
        self.mask = np.zeros((self.img.shape[0], self.img.shape[1]), np.uint8)
        self.mask = cv2.resize(self.mask, (640, 480))
        self.date = date
        if not self.date:
            self.date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.up = uploader.uploader.GoogleDriveUploader()
        self.setup_window()

    def setup_window(self):
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.draw_mask)

    def draw_mask(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.img, (x, y), 10, 255, -1)
            cv2.circle(self.mask, (x, y), 10, 255, -1)
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            cv2.circle(self.img, (x, y), 10, 255, -1)
            cv2.circle(self.mask, (x, y), 10, 255, -1)

    def upload_mask(self):
        mask_image_path = f"Images/color_images/mask-image_{self.date}.jpg"
        cv2.imwrite(mask_image_path, self.mask)
        self.up.upload_to_drive(mask_image_path,Folders.mask_folder)

    def display_image_with_mask(self):
        while True:
            center_coordinates = (self.img.shape[1] // 2, self.img.shape[0] // 2)
            text_line1 = "Draw here"
            org_line1 = (center_coordinates[0] - 150, center_coordinates[1] - 200)
            cv2.putText(self.img, text_line1, org_line1, Consts.font, Consts.font_scale, Consts.font_color,
                        Consts.font_thickness, cv2.LINE_AA)
            cv2.imshow('image', self.img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # esc key
                break
            if key == 13:
                cv2.destroyWindow('image')
                cv2.imshow("image", self.img)
                result = messagebox.askyesno("Confirmation", "Are you sure you want to upload this mask to the database?")
                if result:
                    self.upload_mask()
                    break
                else:
                    break

if __name__ == "__main__":
    image_path = r'C:\Users\Evyatar\PycharmProjects\placenta\placenta_example4.jpg'
    image = cv2.imread(image_path)
    annotation_tool = ImageAnnotation(image)
    annotation_tool.display_image_with_mask()
    cv2.destroyAllWindows()

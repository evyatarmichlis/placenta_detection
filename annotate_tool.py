import cv2
import copy
import numpy as np
from datetime import datetime
from tkinter import messagebox
from consts import Folders
from uploader.dropbox_uploder import DropboxUploader


class ImageAnnotation:
    def __init__(self, image, date=None,upload_online=False):
        self.img = cv2.resize(image, (640, 480))
        self.img_copy = copy.deepcopy(self.img)
        self.mask = np.zeros((self.img.shape[0], self.img.shape[1]), np.uint8)
        self.date = date
        self.upload_online = upload_online
        if not self.date:
            self.date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.up = None
        if self.upload_online:
            token_file = r'C:\Users\evyat\PycharmProjects\placenta_detection\uploader\dropbox_token.txt'
            refresh_token = "ZDhC-N3mwtMAAAAAAAAAAXKuN-TzjTfa_kvRtnVSrcNJ0CHZTQUSol62CXEBRu31"
            client_id = "prrl5bsxyc65kxw"
            client_secret = "hxw3omlgoj9uulz"
            self.up = DropboxUploader(token_file, refresh_token, client_id, client_secret)
        self.draw_enabled = True
        self.setup_window()

    def setup_window(self):
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.draw_mask)

    def draw_mask(self, event, x, y, flags, param):
        if self.draw_enabled:
            if event == cv2.EVENT_LBUTTONDOWN:
                cv2.circle(self.img, (x, y), 10, 255, -1)
                cv2.circle(self.mask, (x, y), 10, 255, -1)
            elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
                cv2.circle(self.img, (x, y), 10, 255, -1)
                cv2.circle(self.mask, (x, y), 10, 255, -1)

    def upload_mask(self,real_defect):
        mask_image_path = f"C:/Users/evyat/PycharmProjects/placenta_detection/Images/gt/{real_defect}mask-image_{self.date}.jpg"
        cv2.imwrite(mask_image_path, self.mask)
        if self.upload_online:
            self.up.upload_to_dropbox(mask_image_path, '/gt')

    def display_image_with_mask(self,real_defect):
        done = False
        while not done:
            center_coordinates = (self.img.shape[1] // 2, self.img.shape[0] // 2)
            text_line1 = "Draw here"
            org_line1 = (center_coordinates[0] - 150, center_coordinates[1] - 200)
            cv2.putText(self.img, text_line1, org_line1, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.imshow('image', self.img)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # esc key
                break
            elif key == 8:  # backspace
                self.img = copy.deepcopy(self.img_copy)
                self.mask = np.zeros((self.img.shape[0], self.img.shape[1]), np.uint8)

            elif key == 13:
                cv2.destroyWindow('image')
                cv2.imshow("image", self.img)
                result = messagebox.askyesno("Confirmation",
                                             "Are you sure you want to upload this mask to the database?")
                if result:
                    self.upload_mask(real_defect)
                    done = True
                else:
                    break

        cv2.destroyAllWindows()  # Close window after loop

if __name__ == "__main__":
    image_path = r'C:\Users\Evyatar\PycharmProjects\placenta\placenta_example4.jpg'
    image = cv2.imread(image_path)
    annotation_tool = ImageAnnotation(image)
    annotation_tool.display_image_with_mask()

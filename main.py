import cv2
from tkinter import messagebox

import pandas as pd

from annotate_tool import ImageAnnotation
from consts import Consts, Folders
from realsense_depth import *
from uploader.dropbox_uploder import DropboxUploader
from uploader.uploader import GoogleDriveUploader

from datetime import datetime
import requests
import tkinter as tk
from tkinter import messagebox

class CustomDialog(tk.Toplevel):
    def __init__(self, parent, options):
        super().__init__(parent)
        self.options = options
        self.result = None
        self.create_widgets()
        self.position_center(parent)
    def position_center(self, parent):
        # Get the dimensions of the parent window

        screen_width = parent.winfo_screenwidth()
        screen_height = parent.winfo_screenheight()
        dialog_width = self.winfo_reqwidth()
        dialog_height = self.winfo_reqheight()
        x = (screen_width - dialog_width) // 2
        y = (screen_height - dialog_height) // 2
        self.geometry("+{}+{}".format(x, y))
    def create_widgets(self):
        label = tk.Label(self, text="How would you classify this image?")
        label.pack()

        for option in self.options:
            button = tk.Button(self, text=option, command=lambda o=option: self.set_result(o))
            button.pack()

    def set_result(self, value):
        self.result = value
        self.destroy()

class PlacentaImageUploader:
    def __init__(self, upload_online=False):
        self.upload_online = upload_online
        self.up = None
        if self.upload_online:
            token_file = r'C:\Users\evyat\PycharmProjects\placenta_detection\uploader\dropbox_token.txt'
            refresh_token = "ZDhC-N3mwtMAAAAAAAAAAXKuN-TzjTfa_kvRtnVSrcNJ0CHZTQUSol62CXEBRu31"
            client_id = "prrl5bsxyc65kxw"
            client_secret = "hxw3omlgoj9uulz"
            self.up = DropboxUploader(token_file, refresh_token, client_id, client_secret)
        self.dc = DepthCamera()
        self.date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def display_live_preview(self,root):
        options = ["Normal","Real defect", "Twins", "Bipartite", "Preterm" ]
        dialog = CustomDialog(root, options)
        root.wait_window(dialog)
        type_msg = dialog.result
        print("Selected option:", type_msg)

        type = f'{type_msg} ' if type_msg else ""
        for side in ["maternal"]:
            while True:
                ret, depth_frame, color_frame, depth_color_image = self.dc.get_frame()
                display_frame = color_frame.copy()
                center_coordinates = (display_frame.shape[1] // 2, display_frame.shape[0] // 2)
                cv2.circle(display_frame, center_coordinates, Consts.circle_radius, Consts.circle_color,
                           Consts.circle_thickness)
                text_line1 = f"Place the {side} of the placenta in the circle"
                text_line2 = "Press Enter to continue"
                org_line1 = (center_coordinates[0] - 170, center_coordinates[1] - 30)
                org_line2 = (center_coordinates[0] - 150, center_coordinates[1] + 10)

                cv2.putText(display_frame, text_line1, org_line1, Consts.font, Consts.font_scale, Consts.font_color,
                            Consts.font_thickness, cv2.LINE_AA)
                cv2.putText(display_frame, text_line2, org_line2, Consts.font, Consts.font_scale, Consts.font_color,
                            Consts.font_thickness, cv2.LINE_AA)
                cv2.imshow("Live Preview", display_frame)

                key = cv2.waitKey(1)
                if key == 27:  # esc key
                    break

                if key == 13:  # Enter key

                    cv2.destroyWindow('Live Preview')
                    cv2.imshow("Image", color_frame)

                    result = messagebox.askyesno("Confirmation",
                                                 "Are you sure you want to upload this image to the database?")

                    if result:
                        cv2.imshow("Image", color_frame)

                        self.upload_images(depth_color_image, color_frame, side, type)
                        self.save_and_upload_csv(depth_frame, side, type)

                        if side == "maternal":
                            annotate_result = messagebox.askyesno("Annotate",
                                                                  "Do you want to annotate the image?")
                            if annotate_result:
                                annotation_tool = ImageAnnotation(color_frame, date=self.date,
                                                                  upload_online=self.upload_online)
                                annotation_tool.display_image_with_mask(type)
                            break
                    break

            cv2.destroyAllWindows()
        self.send_message(f"{type} new sample has been added. date:{self.date}")

    def upload_images(self, depth_frame, color_frame, side, real_defect=""):
        depth_image_path = f"C:/Users/evyat/PycharmProjects/placenta_detection/Images/depth_images/{real_defect}{side}_depth-image_{self.date}.jpg"
        color_image_path = f"C:/Users/evyat/PycharmProjects/placenta_detection/Images/color_images/{real_defect}{side}_color-image_{self.date}.jpg"
        cv2.imwrite(depth_image_path, depth_frame)
        cv2.imwrite(color_image_path, color_frame)
        if self.upload_online:
                self.up.upload_to_dropbox(depth_image_path,'/depth_images')
                self.up.upload_to_dropbox(color_image_path,'/images')

            #google drive try
            # self.up.upload_to_drive(depth_image_path, Folders.depth_folder)
            # self.up.upload_to_drive(color_image_path, Folders.color_folder)

    def save_and_upload_csv(self, depth_frame, side, real_defect=""):
        depth_frame_pd = pd.DataFrame(depth_frame)
        csv_path = rf"C:/Users/evyat/PycharmProjects/placenta_detection/Images/csv_data/{real_defect}raw_depth_{side}_data_{self.date}.csv"
        depth_frame_pd.to_csv(csv_path, index=False)
        if self.upload_online:
            self.up.upload_csv_to_dropbox(csv_path,'/csv_files')
            #google drive try
            # self.up.upload_csv_to_drive(csv_path, Folders.depth_csv_folder)

    def send_message(self, message):
        url = f"https://api.telegram.org/bot" \
              f"{Consts.TOKEN}/sendMessage?chat_id={Consts.chat_id}&text={message}"
        response = requests.get(url)
        if response.status_code == 200:
            print("Message sent successfully")
        else:
            print(f"Failed to send message. Status code: {response.status_code}")

    def main(self):
        i = 1
        root = tk.Tk()

        while True:
            print(f"This is image number {i}")
            i += 1
            self.date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.display_live_preview(root)
            response = messagebox.askyesno("Take Another Image", "Do you want to take another image?")
            if not response:
                print("Stopping the process.")
                break


if __name__ == "__main__":
    clipping_distance_in_meters = 0.8
    uploader = PlacentaImageUploader(upload_online=True)
    uploader.main()

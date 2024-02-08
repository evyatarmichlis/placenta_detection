import cv2
from tkinter import messagebox

import pandas as pd

from annotate_tool import ImageAnnotation
from consts import Consts, Folders
from realsense_depth import *
from uploader.uploader import GoogleDriveUploader
from datetime import datetime
import requests


class PlacentaImageUploader:
    def __init__(self):
        self.dc = DepthCamera()
        self.up = GoogleDriveUploader()
        self.date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def display_live_preview(self):
        for side in ["maternal"]:
            while True:
                ret, depth_frame, color_frame,depth_color_image = self.dc.get_frame()
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

                        self.upload_images(depth_color_image, color_frame,side)
                        self.save_and_upload_csv(depth_frame,side)

                        if side == "maternal":
                            annotate_result = messagebox.askyesno("Annotate",
                                                                  "Do you want to annotate the image?")
                            if annotate_result:
                                annotation_tool = ImageAnnotation(color_frame, date=self.date)
                                annotation_tool.display_image_with_mask()
                            break
                    break

            cv2.destroyAllWindows()
        self.send_message("new sample has been added")

    def upload_images(self, depth_frame, color_frame,side):
        depth_image_path = f"Images/depth_images/{side}_depth-image_{self.date}.jpg"
        color_image_path = f"Images/color_images/{side}_color-image_{self.date}.jpg"
        cv2.imwrite(depth_image_path, depth_frame)
        cv2.imwrite(color_image_path, color_frame)
        self.up.upload_to_drive(depth_image_path, Folders.depth_folder)
        self.up.upload_to_drive(color_image_path, Folders.color_folder)

    def save_and_upload_csv(self, depth_frame, side):
        depth_frame_pd = pd.DataFrame(depth_frame)
        csv_path = f"Images/csv_data/raw_depth_{side}_data_{self.date}.csv"
        depth_frame_pd.to_csv(csv_path, index=False)
        self.up.upload_csv_to_drive(csv_path, Folders.depth_csv_folder)


    def send_message(self, message):
        url = f"https://api.telegram.org/bot" \
              f"{Consts.TOKEN}/sendMessage?chat_id={Consts.chat_id}&text={message}"
        response = requests.get(url)
        if response.status_code == 200:
            print("Message sent successfully")
        else:
            print(f"Failed to send message. Status code: {response.status_code}")


if __name__ == "__main__":
    uploader = PlacentaImageUploader()
    uploader.display_live_preview()

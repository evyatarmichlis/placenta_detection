import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import pandas as pd
import os
import skimage as ski
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from PIL import Image
from skimage.morphology import convex_hull_image

from find_local_max import FindLocalMax


class DefectsFinder:

    def __init__(self,date_of_image=None):
        sys.path.append("..")
        sam_checkpoint = r"/cs/usr/evyatar613/josko_lab/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        self.device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(sam)

    def get_placenta_segments(self, image, plot_img=False):
        anns = self.mask_generator.generate(image)
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0
        possibole_segment = []
        for ann in sorted_anns[:4]:
            masd_dict = {"segmentation": ann['segmentation'], "area": ann['area']}
            m = ann['segmentation']
            if not (m[0:50, :].any()) and not (m[-40:, :].any()) and not (m[:, 0:40].any()) \
                    and not (m[:, -40:].any()) and (m[:, m.shape[1] // 2 - 40:m.shape[1] // 2 + 40].any()):
                possibole_segment.append(masd_dict)
        if len(possibole_segment) > 0:
            final_mask = sorted(possibole_segment, key=lambda x: x['area'])[0]
            if plot_img:
                color_mask = np.concatenate([np.random.random(3), [0.35]])
                img[final_mask["segmentation"]] = color_mask
                ax = plt.gca()
                ax.set_autoscale_on(False)
                ax.imshow(img)
            return final_mask["segmentation"]
        else:
            print("no segmentation for the placenta")

    def read_images_from_dict(self, directory):
        files = []
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            files.append(file_path)

        return files

    def display_segmant(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes[0].imshow(image)
        axes[0].axis('off')
        axes[0].set_title('Original Image')
        axes[1].imshow(image)
        seg = self.get_placenta_segments(image, True)
        print(type(seg))
        axes[1].axis('off')
        axes[1].set_title('Segmentation Result')
        plt.show()

    def find_placenta_contours(self, image):
        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_contours = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 10)
        kernel_size = (5, 5)  # Adjust the kernel size as needed
        img_contours = cv2.GaussianBlur(img_contours, kernel_size, 0)
        return img_contours, contours[0]

    def find_convexity_defects(self, contour):
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)
        return hull, defects

    def plot_cnt(self, cnt, defects, img):
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            cv2.line(img, start, end, [0, 255, 0], 2)
            cv2.circle(img, far, 5, [0, 0, 255], -1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title('Final Image')
        plt.show()

    def placenta_convexity_defects(self, image):
        segment = self.get_placenta_segments(image, False)
        background_color = [255, 255, 255]
        new_img = np.full_like(image, background_color)
        new_img[segment] = image[segment]
        segment = np.where(segment, 1, 0).astype("uint8")
        chull = convex_hull_image(segment)
        defects = chull - segment
        return defects

    def order_connected_components(self, binary_image, original_image, k=3):
        labeled_image, count = ski.measure.label(binary_image, connectivity=2, return_num=True)
        properties = ski.measure.regionprops(labeled_image)
        sorted_regions = sorted(properties, key=lambda x: x.area, reverse=True)
        largest_labels = [region.label for region in sorted_regions[:k]]
        k_largest_cc_masks = [labeled_image == label for label in largest_labels]
        k_largest_cc_binary = np.any(k_largest_cc_masks, axis=0)
        k_largest_cc_binary, count = ski.measure.label(k_largest_cc_binary, connectivity=2, return_num=True)
        return k_largest_cc_binary

    def contour_detection(self, image_path,plot=True):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        defect = self.placenta_convexity_defects(image)
        k_largest_cc_binary = self.order_connected_components(defect, image)
        colored_label_image = ski.color.label2rgb(k_largest_cc_binary, bg_label=0)
        if plot:
            fig, ax = plt.subplots()
            ax.imshow(image)
            ax.imshow(colored_label_image, alpha=0.4)  # Adjust alpha for transparency
            ax.axis("off")
            plt.show()
        return

    def segment_images(self, image_path, color_map_path, depth_csv_path):
        # Read the images and CSV file
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        color_map = cv2.imread(color_map_path)
        color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)

        segment = self.get_placenta_segments(image, False)

        segment_3d = np.dstack((segment, segment, segment))

        depth_data = pd.read_csv(depth_csv_path)
        cropped_image = np.where(segment_3d == True, image, 0)
        cropped_color_map = np.where(segment_3d == True, color_map, 0)
        cropped_depth_data = np.where(segment == True, depth_data, 0)
        return cropped_image, cropped_color_map, cropped_depth_data,segment

    def combine_mask_with_color_map(self,color_map, mask_path, alpha=0.5):
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        mask_alpha = cv2.merge((mask, mask, mask))
        color_map_with_mask = cv2.addWeighted(color_map, 1 - alpha, mask_alpha[:, :, :3], alpha, 0)
        return color_map_with_mask

    def save_cropped_data(self,folder_name,date_of_image,segment,cropped_image,cropped_color_map,cropped_depth_data):
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        cropped_segment_path = rf"{folder_name}/cropped_segment_{date_of_image}.png"
        cropped_image_pil = Image.fromarray(segment)
        cropped_image_pil.save(cropped_segment_path)
        cropped_image_path = rf"{folder_name}/cropped_image_{date_of_image}.png"
        cropped_image_pil = Image.fromarray(cropped_image)
        cropped_image_pil.save(cropped_image_path)
        cropped_color_map_path = rf"{folder_name}/cropped_color_map_{date_of_image}.png"
        cropped_color_map_pil = Image.fromarray(cropped_color_map)
        cropped_color_map_pil.save(cropped_color_map_path)
        cropped_depth_data_path = rf"{folder_name}/cropped_depth_data_{date_of_image}.csv"
        cropped_depth_df = pd.DataFrame(cropped_depth_data)
        cropped_depth_df.to_csv(cropped_depth_data_path, index=False)
        plt.imsave(rf"{folder_name}/cropped_depth_data_image_{date_of_image}.png", cropped_depth_data.data, cmap='gray')
        print("Saved the cropped data")
        return cropped_depth_data_path,cropped_segment_path

if __name__ == '__main__':

    date_of_image = "2024-02-08_13-23-37"
    defect = DefectsFinder()

    image_path=fr"/cs/usr/evyatar613/PycharmProjects/placenta_detection/samples 8_2/color images/maternal_color-image_{date_of_image}.jpg"
    defect.contour_detection(image_path)
    depth_path=fr"/cs/usr/evyatar613/PycharmProjects/placenta_detection/samples 8_2/color map/maternal_depth-image_{date_of_image}.jpg"
    depth_csv = fr"/cs/usr/evyatar613/PycharmProjects/placenta_detection/samples 8_2/depth matirx/raw_depth_maternal_data_{date_of_image}.csv"
    cropped_image, cropped_color_map, cropped_depth_data,segment = defect.segment_images(image_path,depth_path,depth_csv)

    folder_name = "cropped_data"
    cropped_depth_data_path,cropped_segment_path = defect.save_cropped_data(folder_name,date_of_image,segment,cropped_image,cropped_color_map,cropped_depth_data)
    "/cs/usr/evyatar613/PycharmProjects/placenta_detection/samples 8_2/mask/mask-image_2024-02-08_12-56-50.jpg"
    mask = cv2.imread(f"samples 8_2/mask/mask-image_{date_of_image}.jpg")
    mask = cv2.cvtColor(mask, cv2.IMREAD_GRAYSCALE)
    local_max = FindLocalMax(cropped_depth_data_path,cropped_segment_path,threshold=100,ground_truth=mask)
    maxima_coords, magnitude, orientation = local_max.detect_local_maxima(plot=True,gt=True)

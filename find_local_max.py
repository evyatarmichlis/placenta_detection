import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import cv2
from PIL import Image

import os
import numpy as np
import cv2
from pathlib import Path
from PIL import Image


class FindLocalMax:
    def __init__(self, csv_path, rgb_image, ground_truth=None, threshold=None,real_depth = True):
        """
        Initialize the FindLocalMax object.

        Args:
            csv_path (str): Path to the CSV file containing depth data.
            rgb_image (str): Path to the RGB image.
            ground_truth (str): Path to the ground truth file (optional).
            threshold (float): Optional threshold for processing.
        """
        self.csv_path = csv_path
        self.processed_path = str(
            Path(csv_path).parent.parent / 'png_files' / Path(csv_path).name.replace('.csv', '.png')
        )
        os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)
        self.threshold = threshold
        self.data = None
        self.data = self.apply_mask(rgb_image)
        self.data = self.read_csv_and_norm(save=True,real_depth=real_depth)
        self.data = self.apply_mask(rgb_image)
        self.maxima_coords = None
        self.ground_truth = ground_truth
        self.contour = None

    def apply_mask(self, rgb_image_path):
        """
        Convert the RGB image to grayscale and mask out the black (0-value) regions from the CSV data.

        Args:
            rgb_image_path (str): Path to the RGB image.
        """
        # Convert RGB image to grayscale
        rgb_image = cv2.imread(rgb_image_path)
        if rgb_image is None:
            raise FileNotFoundError(f"RGB image not found at {rgb_image_path}")
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        # Create a binary mask from the grayscale image
        _, binary_mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

        # Read CSV data and mask out the black regions
        if self.data is None:

            data = np.genfromtxt(self.csv_path, delimiter=',')

            if data.shape == (481,640):
                data =   data[1:, :]
            if data.shape != (480, 640):
                data = data.reshape(480, 640)
            # data = np.genfromtxt(self.csv_path, delimiter=',')
            if data.shape[:2] != binary_mask.shape:
                raise ValueError("Mismatch between CSV data and RGB image dimensions.")
            self.data = np.where(binary_mask > 0, data, 0)  # Apply mask
        else:
            self.data = np.where(binary_mask > 0, self.data, 0)

        return self.data

    import numpy as np
    import cv2
    from PIL import Image

    def read_csv_and_norm(self, save=False, real_depth = False):
        """
        Enhance local differences in depth data while preserving local max visibility.

        Steps:
        1. Ignore zero values (background).
        2. Apply CLAHE (Local Histogram Equalization).
        3. Apply Gamma Correction to reduce brightness (if needed).
        4. Apply High-Pass Filter (Laplacian) to emphasize edges.
        5. Adaptive Sharpening for high-contrast areas.
        6. Merge enhanced data back into the masked region.

        Args:
            save (bool): Whether to save the processed image.

        Returns:
            np.ndarray: Processed depth image.
        """
        # Load CSV depth data
        data = np.genfromtxt(self.csv_path, delimiter=',')
        if data.shape == (481, 640):
            data = data[1:, :]

        # Mask for valid depth values (ignore background)
        mask = (data > 0).astype(np.uint8)  # 1 for valid pixels, 0 for background

        # Extract non-zero values for processing
        non_zero_pixels = data[mask == 1]  # Extract only valid depth values
        if non_zero_pixels.size > 0 and real_depth:
            ## Step 1: Normalize within valid regions (ignore background)
            min_val = np.percentile(non_zero_pixels, 2)
            max_val = np.percentile(non_zero_pixels, 98)
            stretched = (non_zero_pixels - min_val) / (max_val - min_val)
            stretched = np.clip(stretched, 0, 1) * 255

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            contrast_enhanced = clahe.apply(stretched.astype(np.uint8))

            gamma = 1.2 # Tune this value for your dataset
            gamma_corrected = np.power(contrast_enhanced / 255.0, gamma) * 255
            gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)

            laplacian = cv2.Laplacian(gamma_corrected, cv2.CV_64F, ksize=3)
            enhanced_edges = gamma_corrected - 0.5 * laplacian  # Boosts local differences

            blurred = cv2.GaussianBlur(enhanced_edges, (3, 3), 0)
            sharpened = cv2.addWeighted(enhanced_edges, 1.5, blurred, -0.5, 0)

            processed = np.zeros_like(data)
            processed[data > 0] = sharpened.reshape(-1)

        else:
            processed = data  # If all zeros, keep as is

        # Convert to uint8 for saving
        processed = np.clip(processed, 0, 255).astype(np.uint8)

        if save:
            Image.fromarray(processed).save(self.processed_path)

        return processed

    def adaptive_normalization(self, image, clip_low=5, clip_high=95):
        """Normalize depth values using percentile-based min-max scaling."""
        min_val = np.percentile(image, clip_low)
        max_val = np.percentile(image, clip_high)
        normalized = (image - min_val) / (max_val - min_val) * 255
        return np.clip(normalized, 0, 255).astype(np.uint8)

    def log_transform(self, image):
        """Apply logarithmic transformation to reduce extreme brightness."""
        log_image = np.log1p(image)
        log_image = (log_image / np.max(log_image)) * 255
        return log_image.astype(np.uint8)

    def gamma_correction(self, image, gamma=0.7):
        """Apply gamma correction to adjust brightness."""
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def apply_clahe(self, image, mask):
        """Apply CLAHE only on valid regions."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        valid_pixels = image[mask > 0]
        if valid_pixels.size > 0:
            equalized = clahe.apply(valid_pixels)
            result = np.zeros_like(image)
            result[mask > 0] = equalized.reshape(-1)
            return result
        return image

    def stretch_contrast(self, image):
        """
        Stretch the contrast of the image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Image with stretched contrast.
        """
        min_val = np.min(image)
        max_val = np.max(image)
        stretched_image = (image - min_val) / (max_val - min_val) * 255
        return stretched_image



    def otsu_threshold(self,image, min_var=True):
        hist, bins = np.histogram(image, bins=256, range=(0, 256))
        hist = hist.astype(float) / hist.sum()
        variance = []
        thresholds = []
        for threshold in range(256):
            w0 = hist[:threshold].sum()
            w1 = hist[threshold:].sum()
            if w0 == 0 or w1 == 0:
                continue
            mean0 = (hist[:threshold] * np.arange(threshold)).sum() / w0
            mean1 = (hist[threshold:] * np.arange(threshold, 256)).sum() / w1
            var = w0 * w1 * (mean0 - mean1) ** 2
            variance.append(var)
            thresholds.append(threshold)

        optimal_threshold = thresholds[np.argmax(variance)]
        if min_var:
            optimal_threshold = thresholds[np.argmin(variance)]
        return optimal_threshold

    def plotting(self, maxima_coords, slices, magnitude, orientation, gt):
        if gt:
            fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(15, 4))
        else:
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

        # Plot original image with local maxima
        axs[0].imshow(self.data, cmap="gray")
        axs[0].autoscale(False)
        axs[0].plot(maxima_coords[:, 1], maxima_coords[:, 0], 'ro')
        axs[0].set_title("Original Image with Local Maxima")

        # Plot gradient magnitude
        img1 = axs[1].imshow(magnitude, cmap="jet")
        axs[1].set_title("Gradient Magnitude")
        plt.colorbar(img1, ax=axs[1], orientation='vertical', fraction=0.05)

        # Plot gradient orientation
        img2 = axs[2].imshow(orientation, cmap="jet")
        axs[2].set_title("Gradient Orientation [0, 180]")
        plt.colorbar(img2, ax=axs[2], orientation='vertical', fraction=0.05)

        # Exclude points near the edges of the segmented object
        mask = np.zeros_like(self.data, dtype=bool)
        for dy, dx in slices:
            mask[dy.start:dy.stop, dx.start:dx.stop] = True
        mask = np.invert(mask)
        for ax in axs[:3]:
            ax.set_facecolor('black')
            ax.imshow(np.ma.masked_array(ax.images[0].get_array(), mask))

        # Turn off ticks for each axis
        for ax in axs[:3]:
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

        if gt:
            # Plot ground truth
            axs[3].imshow(self.ground_truth, cmap="gray")
            axs[3].set_title("Ground Truth")
            axs[3].get_xaxis().set_ticks([])
            axs[3].get_yaxis().set_ticks([])

            # Adjust spacing for ground truth plot
            plt.subplots_adjust(wspace=0.3)

        plt.tight_layout()
        plt.show()

    def crop_image_around_maxima(self, original_image, maxima_coords_df, neighborhood_size=50, plot=False):
        cropped_images = []
        patch_indices = []
        for idx, row in maxima_coords_df.iterrows():
            for jdx, value in row.items():
                if value == 1:
                    y, x = idx, jdx
                    # Define bounding box around the local maximum
                    top = max(0, y - neighborhood_size)
                    bottom = min(original_image.shape[0], y + neighborhood_size)
                    left = max(0, x - neighborhood_size)
                    right = min(original_image.shape[1], x + neighborhood_size)
                    # Crop the region from the original image
                    cropped_img = original_image[top:bottom, left:right]
                    patch_indices.append((top, left, bottom, right))
                    cropped_images.append(cropped_img)
        combined_image = np.zeros_like(original_image)
        for patch, (top, left, bottom, right) in zip(cropped_images, patch_indices):
            combined_image[top:bottom, left:right] = patch

        if plot:
            plt.imshow(combined_image)
            plt.title('Combined Cropped Images')
            plt.show()

        return combined_image

    def detect_local_maxima(self, neighborhood_size=10, plot=True, gt=False):
        original_threshold = self.threshold
        current_threshold = original_threshold
        max_attempts = 3  # Number of threshold reduction attempts
        reduction_factor = 0.5  # How much to reduce threshold each time

        for attempt in range(max_attempts):
            if self.threshold is None:
                current_threshold = self.otsu_threshold(self.data, True)
            else:
                current_threshold = original_threshold * (reduction_factor ** attempt)

            print(f"Attempting detection with threshold: {current_threshold}")

            data_max = ndimage.maximum_filter(self.data, neighborhood_size)
            maxima = (self.data == data_max)
            data_min = ndimage.minimum_filter(self.data, neighborhood_size)
            diff = ((data_max - data_min) > current_threshold)
            maxima[diff == 0] = 0

            labeled, num_objects = ndimage.label(maxima)
            slices = ndimage.find_objects(labeled)

            if slices:  # If we found maxima
                maxima_coords = np.array([[(dy.start + dy.stop - 1) / 2, (dx.start + dx.stop - 1) / 2]
                                          for dy, dx in slices])
                maxima_coords = self.close_to_contour(maxima_coords)

                if len(maxima_coords) > 0:  # If we still have points after contour filtering
                    gray = np.uint8(self.data * 255)
                    gX = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
                    gY = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
                    magnitude = np.sqrt((gX ** 2) + (gY ** 2))
                    orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180

                    if plot:
                        self.plotting(maxima_coords, slices, magnitude, orientation, gt)

                    maxima_coords_int = [(int(i), int(j)) for i, j in maxima_coords]
                    df = pd.DataFrame(np.zeros((640, 480), dtype=int))
                    x_coords, y_coords = zip(*maxima_coords_int)
                    df.iloc[y_coords, x_coords] = 1
                    return df, pd.DataFrame(magnitude), pd.DataFrame(orientation)

            print(f"No valid maxima found with threshold {current_threshold}, reducing threshold...")

        # If we still haven't found any maxima after all attempts
        print(f"Warning: No local maxima found for file {self.csv_path} after {max_attempts} attempts")
        df = pd.DataFrame(np.zeros((640, 480), dtype=int))
        gray = np.uint8(self.data * 255)
        gX = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        gY = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        magnitude = np.sqrt((gX ** 2) + (gY ** 2))
        orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180
        return df, pd.DataFrame(magnitude), pd.DataFrame(orientation)

    def convert_csv_to_png(self,csv_path):
        df = pd.read_csv(csv_path, header=None)
        data_array = df.values.astype(np.uint8)  # Convert to uint8
        image = Image.fromarray(data_array[1:, :])
        image.save(csv_path.replace("csv", "png"))



if __name__ == '__main__':

    #
    date = '13-02-11'
    csv_path = f"/cs/usr/evyatar613/PycharmProjects/placenta_detection/model/train_data/cropped_depth_data02-08_{date}.csv"
    segment_path = fr"/cs/usr/evyatar613/PycharmProjects/placenta_detection/model/train_data/RGB/cropped_image_02-08_{date}.png"
    ground_truth_path = fr"model/train_data/GT/mask-image_2024-02-08_{date}.jpg"
    ground_truth = cv2.imread(ground_truth_path)
    ground_truth = cv2.cvtColor(ground_truth, cv2.IMREAD_GRAYSCALE)
    local_max = FindLocalMax(csv_path,segment_path,ground_truth=ground_truth,threshold=2500)
    maxima_coords,magnitude,orientation =local_max.detect_local_maxima(plot=True,gt=True)
    image = cv2.imread(segment_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    crop_local_max = local_max.crop_image_around_maxima(image,maxima_coords,plot=True)
    cropped_csv_path = \
        f"/cs/usr/evyatar613/PycharmProjects/placenta_detection/model/train_data/local_max_cropped_depth_data02-08_{date}.png"
    crop_local_max.save( cropped_csv_path)
    image_csv = cv2.imread(csv_path.replace("csv","png"))
    image_csv = cv2.cvtColor(image_csv , cv2.IMREAD_GRAYSCALE)
    image_csv[crop_local_max[:,:,0] == 0] = 0
    plt.imshow(image_csv)
    plt.show()
    cropped_csv_path= \
        f"/cs/usr/evyatar613/PycharmProjects/placenta_detection/model/train_data/local_max_cropped_depth_data02-08_{date}.png"

    cropped_image_csv_image = Image.fromarray(image_csv)

    cropped_image_csv_image.save( cropped_csv_path)

    # cropped_image_csv_image = Image.fromarray(cropped_image_csv[1:, :])
    # cropped_image_csv_image.save( csv_path.replace("csv", "png"))
    # maxima_coords.to_csv(f"local_maxima_{date}.csv")
    # magnitude.to_csv(f"magnitude_{date}.csv")
    # orientation.to_csv(f"orientation_{date}.csv")
    #
    #
    #

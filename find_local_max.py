import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import cv2
class FindLocalMax:
    def __init__(self, csv_path,segment_path,threshold=None):
        self.csv_path = csv_path
        self.threshold = threshold
        self.data = self.read_csv_and_norm()
        self.maxima_coords = None
        self.contour=None
        if not self.contour:
            self.get_mask_contour(segment_path)

    def get_mask_contour(self,segment_path):
        binary_image = cv2.imread(segment_path, cv2.IMREAD_GRAYSCALE)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(binary_image)
        cv2.drawContours(mask, contours, -1, (255), thickness=1)
        self.contour = [(x, y) for x, y in zip(np.where(mask == 255)[0], np.where(mask == 255)[1])]

    def distance(self,point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def close_to_contour(self,maxima_coords,threshold_distance=40):
        filtered_coords = []
        for point in maxima_coords:
            point_close_to_contour = False
            for cord in self.contour:
                dist = self.distance(cord,point)
                if 0 <= dist <= threshold_distance:
                    point_close_to_contour = True
                    break
            if not point_close_to_contour:
                filtered_coords.append(point)
        filtered_coords = np.array(filtered_coords)
        return filtered_coords

    def read_csv_and_norm(self, p=2):
        my_data = np.genfromtxt(self.csv_path, delimiter=',')[1:, :]
        image = my_data
        min_non_zero = np.min(image[image != 0])
        # Subtract the minimum non-zero value from all non-zero elements
        image[image != 0] -= min_non_zero - 1
        # increase the derivatives
        data = np.power(image, p)
        return data
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


    def plotting(self,maxima_coords,slices,magnitude,orientation):
        # Create subplots
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

        # Plot original image with local maxima
        axs[0].imshow(self.data, cmap="gray")
        axs[0].autoscale(False)
        axs[0].plot(maxima_coords[:, 1], maxima_coords[:, 0], 'ro')
        axs[0].set_title("Original Image with Local Maxima")

        # Plot gradient magnitude
        axs[1].imshow(magnitude, cmap="jet")
        axs[1].set_title("Gradient Magnitude")

        # Plot gradient orientation
        axs[2].imshow(orientation, cmap="jet")
        axs[2].set_title("Gradient Orientation [0, 180]")

        # Exclude points near the edges of the segmented object
        mask = np.zeros_like(self.data, dtype=bool)
        for dy, dx in slices:
            mask[dy.start:dy.stop, dx.start:dx.stop] = True
        mask = np.invert(mask)
        for ax in axs:
            ax.set_facecolor('black')
            ax.imshow(np.ma.masked_array(ax.images[0].get_array(), mask))

        # Turn off ticks for each axis
        for ax in axs:
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

        plt.tight_layout()
        plt.show()

    def detect_local_maxima(self, neighborhood_size=10, plot=True):
        if self.threshold is None:
            self.threshold = self.otsu_threshold(self.data, True)
            print(f"Threshold: {self.threshold}")

        # Finding local maxima
        data_max = ndimage.maximum_filter(self.data, neighborhood_size)
        maxima = (self.data == data_max)
        data_min = ndimage.minimum_filter(self.data, neighborhood_size)
        diff = ((data_max - data_min) > self.threshold)
        maxima[diff == 0] = 0

        # Labeling and finding coordinates of local maxima
        labeled, num_objects = ndimage.label(maxima)
        slices = ndimage.find_objects(labeled)
        maxima_coords = np.array([[(dy.start + dy.stop - 1) / 2, (dx.start + dx.stop - 1) / 2] for dy, dx in slices])
        maxima_coords = self.close_to_contour(maxima_coords)
        gray = np.uint8(self.data * 255)

        gX = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        gY = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

        magnitude = np.sqrt((gX ** 2) + (gY ** 2))
        orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180

        if plot:
            self.plotting(maxima_coords,slices,magnitude,orientation)

        return pd.DataFrame(maxima_coords),pd.DataFrame(magnitude),pd.DataFrame(orientation)



if __name__ == '__main__':
    date = '13-23-37'
    image_path = fr"/cs/usr/evyatar613/PycharmProjects/placenta_detection/cropped_image_02-08_{date}.png"
    csv_path = fr"/cs/usr/evyatar613/PycharmProjects/placenta_detection/cropped_depth_data02-08_{date}.csv"
    segment_path = fr"/cs/usr/evyatar613/PycharmProjects/placenta_detection/cropped_segment_02-08_{date}.png"
    local_max = FindLocalMax(csv_path,segment_path,threshold=150)
    maxima_coords,magnitude,orientation =local_max.detect_local_maxima(plot=True)
    maxima_coords.to_csv(f"local_maxima_{date}.csv")
    maxima_coords.to_csv(f"magnitude_{date}.csv")
    maxima_coords.to_csv(f"orientation_{date}.csv")




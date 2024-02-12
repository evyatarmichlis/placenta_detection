import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage

class FindLocalMax:
    def __init__(self, csv_path,threshold=None):
        self.csv_path = csv_path
        self.threshold = threshold
        self.data = self.read_csv_and_norm()
        self.maxima_coords = None

    def read_csv_and_norm(self,pow=2):
        my_data = np.genfromtxt(self.csv_path, delimiter=',')[1:, :]
        image = my_data
        min_non_zero = np.min(image[image != 0])
        # Subtract the minimum non-zero value from all non-zero elements
        image[image != 0] -= min_non_zero - 1
        # increase the derivatives
        data = np.power(image, pow)

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



    def detect_local_maxima(self, neighborhood_size=10, plot=True):
        if self.threshold is None:
            self.threshold = self.otsu_threshold(self.data,True)
            print(f"Threshold:{self.threshold}")
        data_max = ndimage.maximum_filter(self.data, neighborhood_size)
        maxima = (self.data == data_max)
        data_min = ndimage.minimum_filter(self.data, neighborhood_size)
        diff = ((data_max - data_min) > self.threshold)
        maxima[diff == 0] = 0
        labeled, num_objects = ndimage.label(maxima)
        slices = ndimage.find_objects(labeled)
        maxima_coords = np.array([[(dy.start + dy.stop - 1) / 2, (dx.start + dx.stop - 1) / 2] for dy, dx in slices])
        if plot:
            plt.imshow(self.data)
            plt.autoscale(False)
            plt.plot(maxima_coords[:, 1], maxima_coords[:, 0], 'ro')
            plt.show()
        return pd.DataFrame(maxima_coords)


if __name__ == '__main__':
    date = '12-56-50'
    csv_path = fr"/cs/usr/evyatar613/PycharmProjects/placenta_detection/cropped_depth_data02-08_{date}.csv"
    local_max = FindLocalMax(csv_path)
    result =local_max.detect_local_maxima(plot=True)
    result.to_csv(f"local_maxima_{date}.csv")
import os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import scipy.ndimage as ndimage
import pandas as pd
import matplotlib.pyplot as plt

class FindLocalMax:
    def __init__(self, csv_path, rgb_image_path, ground_truth=None, threshold=None, real_depth=True):
        """
        Locate local maxima in depth maps after masking and normalization.
        """
        self.csv_path = csv_path
        self.rgb_image_path = rgb_image_path
        self.ground_truth = ground_truth
        self.threshold = threshold
        self.real_depth = real_depth

        # Prepare output path
        base = Path(csv_path).parent.parent / 'png_files'
        self.processed_path = str((base / Path(csv_path).name).with_suffix('.png'))
        os.makedirs(base, exist_ok=True)

    def process(self, save=False, clip_low=2, clip_high=98, clip_limit=2.0, tile_size=(8,8), gamma=None,clahe = True):
        """
        Load CSV, mask with RGB, normalize/enhance, and return processed depth.
        """
        # Load depth
        data = np.genfromtxt(self.csv_path, delimiter=',')
        if data.shape == (481, 640):
            data = data[1:, :]

        # Mask using RGB
        mask = self._load_mask(self.rgb_image_path)
        data = np.where(mask, data, 0)

        # Adaptive percentile normalization
        nonzero = data[data > 0]
        if nonzero.size:
            low, high = np.percentile(nonzero, (clip_low, clip_high))
            norm = (data - low) / (high - low) * 255
        else:
            norm = data
        proc = norm

        if self.real_depth:
            if gamma:
                proc = self._gamma_correction(proc, gamma)
            if clahe:
                proc = self._apply_clahe(proc, mask, clip_limit, tile_size)

        proc = np.clip(proc, 0, 255).astype(np.uint8)

        if save:
            Image.fromarray(proc).save(self.processed_path)
        return proc

    def detect_local_maxima(self, data, neighborhood_size=10, plot=True, include_gt=False):
        """
        Given a processed depth array, find and return maxima coords, magnitude, and orientation.
        """
        thresh = self.threshold or self._otsu_threshold(data)
        # detect peaks
        mx = ndimage.maximum_filter(data, neighborhood_size)
        mn = ndimage.minimum_filter(data, neighborhood_size)
        peaks = (data == mx) & ((mx - mn) > thresh)
        labeled, num = ndimage.label(peaks)
        slices = ndimage.find_objects(labeled)
        coords = np.array([[ (dy.start+dy.stop-1)/2, (dx.start+dx.stop-1)/2 ]
                           for dy,dx in slices])
        coords = coords.astype(int)

        # gradients
        gX = cv2.Sobel(data, cv2.CV_64F, 1, 0)
        gY = cv2.Sobel(data, cv2.CV_64F, 0, 1)
        magnitude = np.sqrt(gX**2 + gY**2)
        orientation = (np.arctan2(gY, gX)*180/np.pi)%180

        # Optional plotting
        if plot:
            self._plot_results(data, coords, magnitude, orientation, include_gt)

        # DataFrames if needed
        df_coords = pd.DataFrame(np.zeros_like(data, dtype=int))
        df_coords.values[coords[:,0], coords[:,1]] = 1
        return df_coords, pd.DataFrame(magnitude), pd.DataFrame(orientation)

    def _load_mask(self, rgb_path):
        img = cv2.imread(rgb_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 1, cv2.THRESH_BINARY)
        return mask.astype(bool)

    def _apply_clahe(self, image, mask, clip_limit, tile_size):
        """
        Apply CLAHE across the full image, then keep only masked regions.
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        eq = clahe.apply(image.astype(np.uint8))
        out = np.zeros_like(image, dtype=np.uint8)
        out[mask] = eq[mask]
        return out

    def _gamma_correction(self, image, gamma):
        inv = 1.0/gamma
        table = np.array([(i/255.0)**inv*255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(image.astype(np.uint8), table)

    def _otsu_threshold(self, image):
        hist = cv2.calcHist([image.astype(np.uint8)], [0], None, [256], [0,256]).ravel()
        total = hist.sum()
        sumB = 0; wB = 0; maximum=0; sum1 = np.dot(np.arange(256), hist)
        for i in range(256):
            wB += hist[i];
            if wB==0: continue
            wF = total - wB
            if wF==0: break
            sumB += i*hist[i]
            mB = sumB/wB; mF=(sum1-sumB)/wF
            varBetween = wB*wF*(mB-mF)**2
            if varBetween>maximum:
                threshold=i; maximum=varBetween
        return threshold

    def _plot_results(self, data, coords, mag, ori, include_gt):
        cols = 4 if include_gt else 3
        fig, axs = plt.subplots(1, cols, figsize=(4*cols,4))
        axs[0].imshow(data, cmap='gray'); axs[0].plot(coords[:,1], coords[:,0],'ro'); axs[0].set_title('Data+Maxima')
        im1 = axs[1].imshow(mag, cmap='jet'); axs[1].set_title('Magnitude'); fig.colorbar(im1, ax=axs[1])
        im2 = axs[2].imshow(ori, cmap='jet'); axs[2].set_title('Orientation'); fig.colorbar(im2, ax=axs[2])
        if include_gt and isinstance(self.ground_truth, np.ndarray):
            axs[3].imshow(self.ground_truth, cmap='gray'); axs[3].set_title('GT')
        plt.tight_layout()
        plt.show()

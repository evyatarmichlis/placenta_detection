import matplotlib.pyplot as plt
import numpy as np
import cv2

def apply_canny_to_csv(csv_path, gradient_threshold=20):
    my_data = np.genfromtxt(csv_path, delimiter=',')
    my_data_scaled = (my_data - np.min(my_data)) / (np.max(my_data) - np.min(my_data)) * 255
    my_data_scaled = my_data_scaled.astype(np.uint8)

    image = my_data_scaled

    smoothed_image = cv2.blur(image, (3, 3))  # You can adjust the kernel size as needed

    sobelx = cv2.Sobel(smoothed_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(smoothed_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

    gradient_thresholded = (gradient_magnitude > gradient_threshold) * 255

    return smoothed_image, gradient_thresholded

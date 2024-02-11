import matplotlib.pyplot as plt
import numpy as np
import cv2


def apply_canny_to_csv(csv_path, gradient_threshold=20, ignore_threshold=5):
    # Load data from CSV
    my_data = np.genfromtxt(csv_path, delimiter=',')[1:,:]
    my_data = np.where(my_data > np.mean(my_data), my_data, 0)
    # Scale data to [0, 255]
    # my_data_scaled = (my_data - np.min(my_data)) / (np.max(my_data) - np.min(my_data)) * 255
    # my_data_scaled = my_data_scaled.astype(np.uint8)

    # Convert to image
    image = my_data

    # Smooth the image
    smoothed_image = cv2.boxFilter(image, -1, (1, 1))

    # Compute gradients
    sobelx = cv2.Sobel(smoothed_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(smoothed_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Threshold the gradients
    gradient_thresholded = (gradient_magnitude > gradient_threshold) * 255


    return gradient_magnitude, gradient_thresholded




date = '12-56-50'

csv_path = fr"/cs/usr/evyatar613/PycharmProjects/placenta_detection/cropped_depth_data02-08_{date}.csv"

gradient_magnitude, gradient_thresholded = apply_canny_to_csv(csv_path)

plt.imshow(gradient_thresholded)
plt.show()

plt.imshow(gradient_magnitude)
plt.show()
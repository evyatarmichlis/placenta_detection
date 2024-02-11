import cv2
import numpy as np


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # Get the image and csv data
        image, csv_data = param

        # Get the value at the corresponding location in the csv_data
        csv_value = csv_data[y, x]

        print("CSV Value at ({},{}): {}".format(x, y, csv_value))


def display_image_with_csv(image, csv_data):
    # Create a window and set mouse callback function
    cv2.namedWindow('Image with CSV')
    cv2.setMouseCallback('Image with CSV', mouse_callback, (image, csv_data))

    # Display the image
    while True:
        cv2.imshow('Image with CSV', image)
        key = cv2.waitKey(1)
        if key == 27:  # Escape key
            break

    cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    date = '12-56-50'

    csv_path = fr"/cs/usr/evyatar613/PycharmProjects/placenta_detection/cropped_depth_data02-08_{date}.csv"

    image_path = fr"/cs/usr/evyatar613/PycharmProjects/placenta_detection/cropped_image_02-08_{date}.png"

    image = cv2.imread(image_path)
    csv_data = np.genfromtxt(csv_path, delimiter=',')[1:, :]

    display_image_with_csv(image, csv_data)

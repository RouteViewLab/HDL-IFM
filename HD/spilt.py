import cv2
import numpy as np
import os


input_image_path = r"E:\picture\split\0705.png"
save_dir = r"E:\picture\split\1"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
image = cv2.imread(input_image_path)
if image is not None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        if contour.dtype == np.int32 or contour.dtype == np.float32:
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            segmented_image = np.zeros_like(image)
            cv2.drawContours(segmented_image, [contour], -1, (255, 255, 255), -1)
            cv2.imwrite(os.path.join(save_dir, f'segmented_image_{i}.png'), segmented_image)
        else:
            print(f"Contour {i} has incorrect data type: {contour.dtype}")
else:
    print(f"Image not found or unable to read: {input_image_path}")

print("Split finished:", save_dir)
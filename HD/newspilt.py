import os
import cv2
import time
import numpy as np


def extract_contours_from_image(img_path, output_folder):
    start_time = time.time()
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Image {img_path} could not be read.")
        return 0
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        # Extract original contour region
        extracted_region = image[y:y + h, x:x + w]

        # Create canvas (height=413, width=258)
        canvas = np.zeros((413, 413), dtype=image.dtype)  # Note: (rows, columns) = (height, width)

        # Calculate scaling ratio - use canvas width (258) and height (413)
        scale = min(413 / w, 413 / h)  # width first, then height
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize image
        resized_region = cv2.resize(extracted_region, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Calculate paste position (centered)
        x_offset = (413 - new_w) // 2  # width offset
        y_offset = (413 - new_h) // 2  # height offset

        # Paste the resized contour region into the center of the canvas
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_region

        # Generate output path
        output_filename = f"{os.path.splitext(os.path.basename(img_path))[0]}_contour{i}.png"
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, canvas)
        print(f"Extracted contour saved to {output_path}")

    duration = time.time() - start_time
    print(f"Processed {img_path} in {duration:.2f} seconds")
    return len(contours)


def extract_contours_from_folder(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    total_start = time.time()
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    total_contours = 0
    for img_file in image_files:
        print("\n" + "=" * 50)
        print(f"Processing {os.path.basename(img_file)}...")
        contours_count = extract_contours_from_image(img_file, output_folder)
        total_contours += contours_count

    total_duration = time.time() - total_start
    print("\n" + "=" * 50)
    print(f"Total images processed: {len(image_files)}")
    print(f"Total contours extracted: {total_contours}")
    print(f"Total running time: {total_duration:.2f} seconds")
    print(
        f"Average time per image: {total_duration / len(image_files):.2f} seconds" if image_files else "No images processed")


if __name__ == "__main__":
    folder_path = r"E:\picture\0705split\1\1"
    output_folder = r"E:\picture\0705split\1\1"
    start_time = time.time()
    extract_contours_from_folder(folder_path, output_folder)
    print(f"\nTotal program execution time: {time.time() - start_time:.2f} seconds")

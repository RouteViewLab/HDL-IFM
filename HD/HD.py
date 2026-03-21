import os
import cv2
import numpy as np
import pandas as pd
import re
import time
def extract_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    return contours
def save_contours_to_file(contours, file_path):
    points = np.empty((0, 2), dtype=np.float32)
    for contour in contours:
        for point in contour:
           points = np.vstack((points, point[0]))
    np.savetxt(file_path, points, fmt='%f', delimiter=',')
def load_contours_from_file(file_path):
    points = np.loadtxt(file_path, delimiter=',', dtype=np.float32)
    return points.reshape(-1, 1, 2)
def calculate_hausdorff_distance(points1, points2):
    hausdorff_sd = cv2.createHausdorffDistanceExtractor()
    distance = hausdorff_sd.computeDistance(points1, points2)
    return distance
def process_images_and_save_contours(folder_path1, folder_path2):
    for filename in os.listdir(folder_path1):
        if filename.endswith(('.png')):
            image_path1 = os.path.join(folder_path1, filename)
            image1 = cv2.imread(image_path1)
            contours1 = extract_contours(image1)
            contour_file_path1 = os.path.join(os.path.dirname(image_path1), f"{os.path.splitext(filename)[0]}_contours.txt")
            save_contours_to_file(contours1, contour_file_path1)
    for filename in os.listdir(folder_path2):
        if filename.endswith(('.png')):
            image_path2 = os.path.join(folder_path2, filename)
            image2 = cv2.imread(image_path2)
            contours2 = extract_contours(image2)
            contour_file_path2 = os.path.join(os.path.dirname(image_path2), f"{os.path.splitext(filename)[0]}_contours.txt")
            save_contours_to_file(contours2, contour_file_path2)


def compare_folders_and_save_distances_to_excel(folder_path1, folder_path2, excel_file_path):
    distances = []
    contour_files1 = [f for f in os.listdir(folder_path1) if f.endswith('_contours.txt')]
    contour_files2 = [f for f in os.listdir(folder_path2) if f.endswith('_contours.txt')]

    if not contour_files1 or not contour_files2:
        print("No contour files found in one of the folders.")
        return

    for file1 in contour_files1:
        file_path1 = os.path.join(folder_path1, file1)
        points1 = load_contours_from_file(file_path1)

        for file2 in contour_files2:
            file_path2 = os.path.join(folder_path2, file2)
            points2 = load_contours_from_file(file_path2)
            distance = calculate_hausdorff_distance(points1, points2)
            distances.append((os.path.basename(file_path1), os.path.basename(file_path2), distance))

    df = pd.DataFrame(distances, columns=['File1', 'File2', 'Hausdorff Distance'])

    def extract_angle(file_name):
        match = re.search(r'_rotated_(\d+)', file_name)  # 修改后的正则表达式
        if match:
            return int(match.group(1))
        return 0

    df['Angle'] = df['File2'].apply(extract_angle)
    df_sorted = df.sort_values(by='Angle')
    df_sorted.to_excel(excel_file_path, index=False)
    print(f"Distances saved to {excel_file_path}")
if __name__ == "__main__":
    folder_path2 = r'C:\Users\Administrator\PycharmProjects\pythonProject\NEW\try\0705spilt'
    folder_path1 = r'C:\Users\Administrator\PycharmProjects\pythonProject\NEW\try\bspilt'
    excel_file_path = r'C:\Users\Administrator\PycharmProjects\pythonProject\NEW\try\test hausdorff\b12hausdorff distances.xlsx'
    start_time = time.time()
    try:
        process_images_and_save_contours(folder_path1, folder_path2)
        compare_folders_and_save_distances_to_excel(folder_path1, folder_path2, excel_file_path)
    except Exception as e:
        print(f"An error occurred: {e}")
    end_time = time.time()
    print(f"Program execution time: {end_time - start_time:.2f} seconds")
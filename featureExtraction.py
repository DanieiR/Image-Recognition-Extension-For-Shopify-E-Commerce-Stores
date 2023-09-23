import cv2
import numpy as np
import os
import csv
from sklearn.cluster import KMeans

def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def extract_features(image, feature_extractor):
    gray_image = convert_to_gray(image)
    return feature_extractor.detectAndCompute(gray_image, None)


def perform_kmeans_clustering(image, num_clusters):
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(pixels)
    return kmeans.cluster_centers_[kmeans.labels_].reshape(image.shape)


def load_image(image_path, img_size):
    image = cv2.imread(image_path)
    if image is None:
        return None
    return cv2.resize(image, img_size)


def process_images(folder_path, img_size, feature_extractor, num_clusters, csv_file):
    features_list = []
    file_names = []
    dominant_colors = []

    for file_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file_name)
        image = load_image(image_path, img_size)

        if image is None:
            print(f"Error reading image: {file_name}")
            continue

        keypoints, descriptors = extract_features(image, feature_extractor)
        features_list.append(descriptors)
        file_names.append(file_name)

        dominant_color = perform_kmeans_clustering(image, num_clusters)
        dominant_colors.append(dominant_color)

    export_to_csv(file_names, features_list, dominant_colors, csv_file)


def export_to_csv(file_names, features_list, dominant_colors, csv_file):
    with open(csv_file, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Image", "Features", "Dominant_Colors"])
        for i in range(len(file_names)):
            csv_writer.writerow([file_names[i], features_list[i].tobytes().hex(), dominant_colors[i].tolist()])

# Define the folder containing the images
folder_path = "Images"

# Define the desired size of the images
img_size = (255, 255)

# Define the feature extractor
feature_extractor = cv2.ORB_create()

# Define the number of clusters for K-means
num_clusters = 5

# Define the CSV file path
csv_file = "imagefeatures.csv"

# Process the images and export features to CSV
process_images(folder_path, img_size, feature_extractor, num_clusters, csv_file)
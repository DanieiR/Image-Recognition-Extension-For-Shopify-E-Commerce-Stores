import cv2
import numpy as np
import os
import csv
from sklearn.cluster import KMeans
import sys
csv.field_size_limit(2**30)  # 1 GB

import requests

def download_image(url, save_as):
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure we got a valid response

    with open(save_as, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192): 
            file.write(chunk)

# Use the function
image_url = "https://th.bing.com/th/id/R.6a829c1170f902032e391c4339cfd0ae?rik=kJ0UVohLUITuHA&riu=http%3a%2f%2fwww.zazzle.com%2frlv%2fsvc%2fview%3frlvnet%3d1%26realview%3d113745894146800642%26design%3dc25591fa-5c89-4716-9dda-e105940beb52%26style%3dhanes_mens_crew_tshirt_5250%26size%3da_l%26color%3dwhite%26max_dim%3d325%26bg%3d0xffffff&ehk=BrPwvN2OQR51VKFIxxxe0DTTdBFyoT3tnXpdF2bKF58%3d&risl=&pid=ImgRaw&r=0"
save_as = "local_image.jpg"
download_image(image_url, save_as)


def load_image(image_path, img_size):
    image = cv2.imread(image_path)
    if image is None:
        return None
    return cv2.resize(image, img_size)


def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def extract_features(image, feature_extractor):
    gray_image = convert_to_gray(image)
    return feature_extractor.detectAndCompute(gray_image, None)


def perform_kmeans_clustering(image, num_clusters):
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(pixels)
    return kmeans.cluster_centers_[kmeans.labels_].reshape(image.shape)


def process_images(folder_path, img_size, feature_extractor, num_clusters):
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

    return file_names, features_list, dominant_colors


def export_to_csv(file_names, features_list, dominant_colors, csv_file):
    with open(csv_file, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Image", "Features", "Dominant_Colors"])
        for i in range(len(file_names)):
            csv_writer.writerow([file_names[i], features_list[i].tobytes().hex(), dominant_colors[i].tolist()])


def read_features_from_csv(csv_file):
    with open(csv_file, "r", newline="") as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip header row

        file_names = []
        features_list = []
        dominant_colors = []

        for row in csv_reader:
            file_names.append(row[0])
            features = np.frombuffer(bytes.fromhex(row[1]), dtype=np.uint8)
            features_list.append(features.reshape(-1, 32))

            dominant_color = np.array(eval(row[2]))
            dominant_colors.append(dominant_color)

        return file_names, features_list, dominant_colors


def calculate_similarities(input_descriptors, features_list, input_dominant_color, dominant_colors):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    similarities = {}

    for i in range(len(features_list)):
        matches = bf.match(input_descriptors, features_list[i])
        ratio = len(matches) / len(input_keypoints)
        similarity = ratio * 100

        dominant_color_diff = np.sum(np.abs(input_dominant_color - dominant_colors[i]))
        dominant_color_diff_scaled = dominant_color_diff / 100005
        score = similarity - dominant_color_diff_scaled

        similarities[file_names[i]] = score

    return similarities


def normalize_score(score, min_score, max_score):
    return (score - min_score) / (max_score - min_score) * 100

def get_top_similar_images(similarities, folder_path, num_similar_images=5):
    similarities_sorted = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    min_score = similarities_sorted[-1][1]
    max_score = similarities_sorted[0][1]

    similar_images = []
    similarity_scores = []

    for i in range(num_similar_images):
        similar_image_path = os.path.join(folder_path, similarities_sorted[i][0])
        similar_image = load_image(similar_image_path, img_size)
        similar_images.append(similar_image)
        similarity_scores.append(normalize_score(similarities_sorted[i][1], min_score, max_score))

    return similar_images, similarity_scores

def display_image(window_name, images, scores, index):
    image = images[index].copy()
    text = f"Similarity Score: {scores[index]:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x = 10
    text_y = img_size[1] - 16
    cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
    cv2.imshow(window_name, image)

def handle_key_event(images, index, key):
    if key == ord("n"):
        return (index + 1) % len(images)
    elif key == ord("p"):
        return (index - 1) % len(images)
    else:
        return index

import os.path

# Define the CSV file path
csv_file = "image_features.csv"

# Define the folder containing the images
folder_path = "Images"

# Define the desired size of the images
img_size = (255, 255)

# Define the feature extractor
feature_extractor = cv2.ORB_create()

# Define the number of clusters for K-means
num_clusters = 5

# Check if the CSV file already exists
if not os.path.isfile(csv_file):
    # Process the images and export features to CSV
    file_names, features_list, dominant_colors = process_images(folder_path, img_size, feature_extractor, num_clusters)
    export_to_csv(file_names, features_list, dominant_colors, csv_file)
else:
    print(f"{csv_file} already exists. Skipping feature extraction and CSV creation.")

# Read the features from the CSV file
file_names, features_list, dominant_colors = read_features_from_csv(csv_file)

# Provide the path to input image
input_image_path = save_as

# Read the input image and extract its features
input_image = load_image(input_image_path, img_size)
input_keypoints, input_descriptors = extract_features(input_image, feature_extractor)
input_dominant_color = perform_kmeans_clustering(input_image, num_clusters)

# Calculate the similarity between the input image and all images in the folder
similarities = calculate_similarities(input_descriptors, features_list, input_dominant_color, dominant_colors)

# Get the top 5 most similar images
similar_images, similarity_scores = get_top_similar_images(similarities, folder_path)

# Display the similar images
window_name = "Similar Images"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

index = 0
display_image(window_name, similar_images, similarity_scores, index)
while True:
    key = cv2.waitKey(0) 
    if key == 27:  # Escape key
        break
    new_index = handle_key_event(similar_images, index, key)
    if new_index != index:
        index = new_index
        display_image(window_name, similar_images, similarity_scores, index)

cv2.destroyAllWindows()

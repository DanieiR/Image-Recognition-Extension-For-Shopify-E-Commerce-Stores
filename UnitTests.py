import unittest
import tempfile
import shutil
import os
import numpy as np
import cv2
from skimage import data
from InitialModel import (load_image, convert_to_gray, extract_features, perform_kmeans_clustering,
                         process_images, export_to_csv, read_features_from_csv, calculate_similarities,
                         normalize_score, get_top_similar_images)

class TestImageSimilarity(unittest.TestCase):
    def setUp(self):
        self.temp_folder = tempfile.mkdtemp()
        self.feature_extractor = cv2.ORB_create()
        self.num_clusters = 5
        self.img_size = (255, 255)

        # Save images to the temporary folder
        for i in range(5):
            image = data.astronaut() 
            temp_file = os.path.join(self.temp_folder, f"image{i}.jpg")
            cv2.imwrite(temp_file, image)

    def tearDown(self):
        shutil.rmtree(self.temp_folder)

    def test_load_image(self):
        img = load_image(os.path.join(self.temp_folder, "image0.jpg"), self.img_size)
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (*self.img_size, 3))

    def test_convert_to_gray(self):
        img = cv2.imread(os.path.join(self.temp_folder, "image0.jpg"))
        gray_img = convert_to_gray(img)
        self.assertEqual(gray_img.shape, img.shape[:2])

    def test_extract_features(self):
        img = cv2.imread(os.path.join(self.temp_folder, "image0.jpg"))
        keypoints, descriptors = extract_features(img, self.feature_extractor)
        self.assertIsNotNone(keypoints)
        self.assertIsNotNone(descriptors)

    def test_perform_kmeans_clustering(self):
        img = cv2.imread(os.path.join(self.temp_folder, "image0.jpg"))
        clustered_image = perform_kmeans_clustering(img, self.num_clusters)
        self.assertIsNotNone(clustered_image)
        self.assertEqual(clustered_image.shape, img.shape)

    def test_process_images(self):
        file_names, features_list, dominant_colors = process_images(self.temp_folder, self.img_size, self.feature_extractor, self.num_clusters)
        self.assertEqual(len(file_names), 5)
        self.assertEqual(len(features_list), 5)
        self.assertEqual(len(dominant_colors), 5)

    def test_export_and_read_csv(self):
        file_names, features_list, dominant_colors = process_images(self.temp_folder, self.img_size, self.feature_extractor, self.num_clusters)
        csv_file = os.path.join(self.temp_folder, "image_features.csv")
        export_to_csv(file_names, features_list, dominant_colors, csv_file)

        read_file_names, read_features_list = read_features_from_csv(csv_file)
        self.assertEqual(len(read_file_names), len(file_names))
        self.assertEqual(len(read_features_list), len(features_list))

    def test_normalize_score(self):
        score = 50
        min_score = 0
        max_score = 100
        normalized_score = normalize_score(score, min_score, max_score)
        self.assertEqual(normalized_score, 50)


if __name__ == "__main__":
    unittest.main()

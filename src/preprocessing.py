# src/preprocessing.py
import cv2
import numpy as np

def preprocess_image(image_path):
    # Read image
    image = cv2.imread(image_path)
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blur
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # Histogram Equalization for contrast improvement
    equalized_image = cv2.equalizeHist(blurred_image)
    return equalized_image

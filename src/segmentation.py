# src/segmentation.py
import cv2
import numpy as np

def canny_edge_detection(image, low_threshold=30, high_threshold=100):
    """Apply Canny edge detection to highlight edges in the image."""
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges

def threshold_segmentation(image, threshold_value=127):
    """Apply binary thresholding to segment the image."""
    _, thresholded = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return thresholded

def watershed_segmentation(image):
    """Apply watershed segmentation for more complex segmentation tasks."""
    # Convert image to grayscale and apply Gaussian blur
    gray = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Apply thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Noise removal using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Marker labeling
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels to differentiate sure background
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply watershed
    markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), markers)
    segmented_image = np.copy(image)
    segmented_image[markers == -1] = [255]  # Mark boundaries in white

    return segmented_image

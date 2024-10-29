# src/main.py
from preprocessing import preprocess_image
from segmentation import canny_edge_detection, threshold_segmentation, watershed_segmentation
import matplotlib.pyplot as plt
import cv2

# Path to a sample image
image_path = "../data/raw/sample_image.png"  # Make sure to have this file

# Process the image
processed_image = preprocess_image(image_path)

# Apply various segmentation techniques
edges = canny_edge_detection(processed_image)
thresholded = threshold_segmentation(processed_image)
watershed_segmented = watershed_segmentation(processed_image)

# Display segmentation results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection")

plt.subplot(1, 3, 2)
plt.imshow(thresholded, cmap='gray')
plt.title("Threshold Segmentation")

plt.subplot(1, 3, 3)
plt.imshow(watershed_segmented, cmap='gray')
plt.title("Watershed Segmentation")

plt.show()

# Save results (optional)
cv2.imwrite("../data/processed/edges.png", edges)
cv2.imwrite("../data/processed/thresholded.png", thresholded)
cv2.imwrite("../data/processed/watershed.png", watershed_segmented)
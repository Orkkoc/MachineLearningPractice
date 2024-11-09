import os
from preprocessing import preprocess_image
import cv2

# Set the working directory to the project root
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")

# Define the directories
input_dir = "data/raw/"
output_dir = "data/processed/"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each image in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):  # Check for common image file types
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Preprocess the image
        processed_image = preprocess_image(input_path)

        # Save the processed image
        cv2.imwrite(output_path, processed_image)

        print(f"Processed and saved: {filename}")

print("All images have been processed.")

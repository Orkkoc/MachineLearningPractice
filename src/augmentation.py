# src/augmentation.py
import cv2
import numpy as np
import os

def rotate_image(image, angle):
    """Rotate the image by a specific angle."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def flip_image(image, flip_code):
    """Flip the image horizontally or vertically."""
    return cv2.flip(image, flip_code)

def add_noise(image, noise_level=10):
    """Add random noise to the image."""
    noise = np.random.randint(-noise_level, noise_level, image.shape, dtype='int16')
    noisy_image = cv2.add(image, noise, dtype=cv2.CV_8U)
    return noisy_image

def save_image(image, path, filename):
    """Save an image to a specified path."""
    os.makedirs(path, exist_ok=True)
    cv2.imwrite(os.path.join(path, filename), image)
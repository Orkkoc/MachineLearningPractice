# src/model.py
import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape=(850, 619, 1)):
    """Creates a simple CNN model for image classification."""
    model = models.Sequential()

    # First convolutional layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Second convolutional layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Third convolutional layer
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten and fully connected layer
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification (0 or 1)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
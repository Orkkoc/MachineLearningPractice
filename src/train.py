# src/train.py
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_model
from preprocessing import preprocess_image

# Paths
data_dir = "../data/processed/"
batch_size = 16
img_height, img_width = 256, 256

# Image Data Generator for loading and augmenting images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Load training and validation data
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="binary",
    subset="training"
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="binary",
    subset="validation"
)

# Create and train the model
model = create_model(input_shape=(img_height, img_width, 1))
model.summary()

# Training the model
epochs = 10
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs
)

# Save the trained model
model.save("wagner_syndrome_cnn_model.h5")
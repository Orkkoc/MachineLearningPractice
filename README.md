
# Wagner Syndrome Detection from Eye Tomography Images using AI

This project aims to develop an AI-based image processing application for diagnosing Wagner Syndrome from eye tomography data. The application uses machine learning and image processing techniques to preprocess, augment, segment, and analyze tomography images, ultimately training a deep learning model to detect signs of the syndrome.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Setup Instructions](#setup-instructions)
- [Usage Guide](#usage-guide)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

**Wagner Syndrome** is a rare hereditary eye disease that affects the vitreous body and retina, potentially leading to vision impairment. This project leverages AI and machine learning to assist in early detection of the syndrome from eye tomography images. By analyzing images, the model aims to identify markers that indicate the presence of Wagner Syndrome, aiding in the diagnosis process.

## Features

- **Image Preprocessing**: Converts raw tomography images into formats suitable for machine learning.
- **Data Augmentation**: Enhances the training data by generating variations of images.
- **Image Segmentation**: Identifies key areas in tomography images for analysis.
- **CNN Model Training**: Trains a Convolutional Neural Network to classify images.
- **Prediction and Evaluation**: Evaluates model accuracy and makes predictions on new images.

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/WagnerSyndromeDetection.git
   cd WagnerSyndromeDetection
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment**:
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Guide

1. **Add Data**:
   - Place raw tomography images in `data/raw/`.
   - For model training, organize images in `data/processed/` with two subfolders:
     - `class_0/` for images without Wagner Syndrome
     - `class_1/` for images with Wagner Syndrome

2. **Run Preprocessing**:
   ```bash
   python src/main.py
   ```

3. **Train the Model**:
   ```bash
   python src/train.py
   ```

4. **Evaluate and Predict**:
   - Once trained, you can use the model on new images to make predictions.

## Project Structure

```plaintext
WagnerSyndromeDetection/
├── data/               # Data storage
│   ├── raw/            # Raw eye tomography images
│   └── processed/      # Processed and segmented images
├── src/                # Source code
│   ├── preprocessing.py   # Image preprocessing functions
│   ├── augmentation.py    # Data augmentation functions
│   ├── segmentation.py    # Image segmentation methods
│   ├── model.py           # CNN model architecture
│   ├── train.py           # Training script
│   └── utils.py           # Utility functions
├── notebooks/          # Jupyter notebooks for analysis
├── output/             # Model output and predictions
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Future Work

- **Hyperparameter Tuning**: Enhance model performance through parameter optimization.
- **Advanced Segmentation**: Implement more sophisticated segmentation techniques.
- **Deployment**: Build a web or mobile application for real-time diagnosis.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with any improvements or new features.

## License

This project is licensed under the MIT License 

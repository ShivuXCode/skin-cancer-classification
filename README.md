# Skin Cancer Classification Web App

This project is a deep learning-based web application that classifies skin lesions as either **Benign** or **Malignant** using a Convolutional Neural Network (EfficientNetB0).

## Overview

Skin cancer is one of the most common forms of cancer. Early detection is critical. This tool provides a simple, user-friendly interface to upload images of skin lesions and instantly receive a preliminary diagnosis.

The application uses a custom-trained **EfficientNetB0** model, fine-tuned on the HAM10000 dataset, to classify images into two primary categories.

## Features

- **Drag and Drop Interface:** Easily upload images for prediction.
- **Fast Inference:** Built using TensorFlow and Flask, providing near-instant results.
- **Confidence Scores:** The model outputs a probability score for its prediction.

## Project Structure

- `/app`: Contains the Flask web application (`app.py`), HTML templates, and CSS/JS assets.
- `/models`: Stores the trained `.h5` deep learning models.
- `/notebooks`: Contains the Jupyter notebook used for exploratory data analysis (EDA) and model training.
- `/scripts`: Contains utility scripts for training or comparing models locally.
- `/data`: Used for storing the dataset locally (ignored in git).

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- Flask
- Pillow, Pandas, Numpy

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/ShivuXCode/skin-cancer-classification.git
   cd skin-cancer-classification
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Ensure you have TensorFlow and Flask installed)*

3. Run the Flask Application:
   ```bash
   python app/app.py
   ```

4. Open your web browser and navigate to `http://127.0.0.1:5001`.

## Disclaimer
This application is for educational and research purposes only. It is not intended to replace professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

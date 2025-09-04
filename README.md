# Waste Classification System

## Overview

The Waste Classification System classifies waste images into categories such as cardboard, glass, metal, paper, plastic, and trash using deep learning models. The project provides a web-based demo for easy image upload and prediction.

## Features

* Multi-model support: CNN, ResNet, Autoencoder, Multimodal model, YOLOv8, GAN.
* Web interface built with Flask for uploading images and viewing results.
* GAN image generation for selected waste categories.
* YOLO object detection for identifying multiple objects in images.
* Automatic download of models if they do not exist locally.

## Technologies

* Python 3
* Flask
* TensorFlow / Keras
* PyTorch (for YOLOv8)
* TensorFlow Addons
* OpenCV, Pillow, NumPy
* gdown (for downloading models)

## Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/malak128/waste-classification.git
cd waste-classification
```

2. **Create a virtual environment**

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the web app**

```bash
python app.py
```

Access the app locally at the link.

## Model Storage & Automatic Download

* Models are stored in the `saved_models/` folder.
* When you run the web app for the first time, if the models are **not already present**, they will automatically **download from Google Drive** to your device.
* This ensures you don’t need to manually download or place any model files.
* Drive link for reference (manual download if needed): [Google Drive Models](https://drive.google.com/drive/folders/17NT5-jTKiYdlFrP62o4Z2v1tjYMgjOJx?usp=sharing)

## Usage

* Upload an image of waste to classify using CNN, ResNet, YOLO, or multimodal models.
* Select a category and generate an image using the GAN model.
* View predictions and generated images directly on the web page.



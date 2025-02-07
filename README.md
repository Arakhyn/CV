Card Detection with OpenCV and TensorFlow

This project implements a card detection system using OpenCV, TensorFlow, and real-time screen capture with mss. The goal is to identify cards on the screen and classify them correctly using a deep learning model.

Features

Real-time screen capture.

Image preprocessing with OpenCV.

Contour segmentation to identify cards.

Classification using a convolutional neural network model.

Model training with a custom dataset.

Requirements

Make sure you have the following dependencies installed:

pip install opencv-python numpy mss tensorflow

Usage

Run the main script to start real-time detection:

python script.py

Press q to exit the program.

Project Structure

project-folder/
│── train/  # Training images
│── test/   # Test images
│── valid/  # Validation images
│── script.py  # Main script
│── modelo_cartas.h5  # Trained model (if exists)

Model Training

If no pre-existing model is found, the script will train a new one using the images in the train, test, and valid folders. Make sure to structure the folders with subdirectories for each card class.

Contribution

If you want to contribute, you can clone the repository and submit improvements via a Pull Request.

License

This project is under the MIT license.

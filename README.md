# PyTorch-Gradio-MNIST-Digit-Classification
"An interactive PyTorch-based solution for MNIST digit classification, featuring Gradio for real-time predictions and customizable model training."

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Usage](#usage)
- [Installation](#installation)



## Introduction

Handwritten digit classification is a fundamental problem in the field of machine learning and computer vision. The MNIST dataset, comprising 28x28 pixel grayscale images of handwritten digits (0-9), serves as a standard benchmark for developing and testing machine learning models.

This project utilizes PyTorch, a popular deep learning framework, to train a simple neural network for accurately recognizing handwritten digits. Moreover, it leverages Gradio, a Python library for creating customizable UI components, to provide an interactive interface for model training and prediction.

## Features

- **Model Training**: Train a neural network with customizable hyperparameters such as hidden units, learning rate, and epochs.
- **Interactive Visualization**: Visualize the training process with plots showing accuracy over epochs.
- **Real-time Prediction**: Draw a digit on the sketchpad and get real-time predictions from the trained model.

- ### Training Accuracy Plot
![Screenshot 2024-06-12 181116](https://github.com/Abhirajraushan/PyTorch-Gradio-MNIST-Digit-Classification/assets/84666932/358412b1-fa72-4e05-bcfe-a48fe4a6a474)
This plot shows the accuracy of the model on the training data over epochs. It helps visualize the training progress and identify any trends or patterns.

### Predicted Digit Example
![Screenshot 2024-06-12 180934](https://github.com/Abhirajraushan/PyTorch-Gradio-MNIST-Digit-Classification/assets/84666932/0d91049c-e483-49d2-9e11-d64e6b784762)
![Screenshot 2024-06-12 181032](https://github.com/Abhirajraushan/PyTorch-Gradio-MNIST-Digit-Classification/assets/84666932/24e8436a-92df-4779-bea0-dce6183e3bce)
This image illustrates an example of a handwritten digit from the MNIST dataset along with the model's prediction. It demonstrates how the model performs in real-world scenarios.

## Usage
1.Clone the repository:
git clone https://github.com/Abhirajraushan/PyTorch-Gradio-MNIST-Digit-Classification.git

2.Navigate to the project directory:
cd PyTorch-Gradio-MNIST-Digit-Classification

3.Train the model and visualize accuracy:
python train_and_visualize.py

4.To predict digits interactively, run:
python predict_digit.py



## Installation

Ensure you have the following dependencies installed:

- Python 3
- PyTorch
- Gradio
- Matplotlib
- NumPy
- PIL
  

You can install the dependencies via pip:
```bash
pip install -r requirements.txt
nds or patterns.





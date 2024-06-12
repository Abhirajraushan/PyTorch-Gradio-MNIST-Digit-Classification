# PyTorch-Gradio-MNIST-Digit-Classification
"An interactive PyTorch-based solution for MNIST digit classification, featuring Gradio for real-time predictions and customizable model training."

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Configuration](#configuration)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Introduction

Handwritten digit classification is a fundamental problem in the field of machine learning and computer vision. The MNIST dataset, comprising 28x28 pixel grayscale images of handwritten digits (0-9), serves as a standard benchmark for developing and testing machine learning models.

This project utilizes PyTorch, a popular deep learning framework, to train a simple neural network for accurately recognizing handwritten digits. Moreover, it leverages Gradio, a Python library for creating customizable UI components, to provide an interactive interface for model training and prediction.

## Features

- **Model Training**: Train a neural network with customizable hyperparameters such as hidden units, learning rate, and epochs.
- **Interactive Visualization**: Visualize the training process with plots showing accuracy over epochs.
- **Real-time Prediction**: Draw a digit on the sketchpad and get real-time predictions from the trained model.

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

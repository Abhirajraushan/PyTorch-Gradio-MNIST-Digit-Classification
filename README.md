# PyTorch-Gradio-MNIST-Digit-Classification
"An interactive PyTorch-based solution for MNIST digit classification, featuring Gradio for real-time predictions and customizable model training."
This repository contains code for training a simple neural network using PyTorch to classify handwritten digits from the MNIST dataset. Additionally, it provides an interactive interface for both training the model and predicting digits using Gradio.

Features
Model Training: Train a neural network with customizable hyperparameters such as hidden units, learning rate, and epochs.
Interactive Visualization: Visualize the training process with plots showing accuracy over epochs.
Real-time Prediction: Draw a digit on the sketchpad and get real-time predictions from the trained model.
Requirements
Python 3
PyTorch
Gradio
Matplotlib
NumPy
PIL
Usage
Clone the repository:
bash
Copy code
git clone https://github.com/your-username/mnist-digit-classification.git
Install the dependencies:
bash
Copy code
pip install -r requirements.txt
Train the model and visualize accuracy:
bash
Copy code
python train_and_visualize.py
To predict digits interactively, run:
bash
Copy code
python predict_digit.py
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
The code is adapted from various sources, including PyTorch documentation and Gradio examples.

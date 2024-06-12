import torch
import torch.nn as nn
from torch import optim  
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def reshape_to_28x28(image):
    # Open the image
    image = Image.fromarray(image)
    # Resize the image to 28x28 while maintaining aspect ratio
    resized_image = image.resize((28, 28))
    return resized_image

# Define the neural network
class SimpleNN(nn.Module):
  def __init__(self, hidden_units=128):
    super(SimpleNN, self).__init__()
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(28 * 28, hidden_units)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_units, 10)  # 10 output units for 10 digits

  def forward(self, x):
    x = self.flatten(x)
    x = self.relu(self.fc1(x))
    x = self.fc2(x)
    return F.softmax(x, dim=1)  # Softmax for probability distribution

model = SimpleNN(16)

# Function to train the model
def train_model(hidden_units, learning_rate, epochs):
  global model
  # Define data loaders
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
  train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
  test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

  # Create model and optimizer
  model = SimpleNN(hidden_units)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  # Training loop
  train_losses, test_losses, test_accuracies = [], [], []
  for epoch in range(epochs):
    for images, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      train_losses.append(loss.item())

    # Calculate test accuracy and loss
    test_loss, test_correct, total = 0, 0, 0
    correct = 0
    model.eval()
    with torch.no_grad():
      for images, labels in test_loader:
        outputs = model(images)
        test_loss += criterion(outputs, labels).item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    test_accuracies.append(100 * correct / total)
    test_losses.append(test_loss / total)
      # Print final test results
    print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")
    print(f"Final Test Loss: {test_losses[-1]:.4f}")

  return model, train_losses, test_losses, test_accuracies

# Function to predict digit and its probability from a 28x28 NumPy array
def predict(sketch):
  global model
  # Preprocess the image (assuming grayscale)
  image = sketch["composite"]
  grayscale_image = np.mean(image, axis=2)  # Average across RGB channels
  # Reshape the image to a 28x28x1 format (assuming a grayscale model)
  image = np.asarray(reshape_to_28x28(grayscale_image))

  plt.imsave("digit.png", image)


  image = np.expand_dims(image, axis=0)  # Add batch dimension
  image = image.astype(np.float32)  # Convert to float32
  # image /= 255.0  # Normalize to 0-1 range

  # Create a PyTorch tensor
  image_tensor = torch.from_numpy(image)

  # Load a trained model (replace with your model loading logic)
  model.eval()  # Set model to evaluation mode

  # Get predictions
  with torch.no_grad():
    output = model(image_tensor)

  predicted = torch.argmax(output)
  print(predicted)
  print(output, predicted)
  # Return a dictionary with digit and probability
  return {
      str(int(predicted)): output[0][int(predicted)]
  }

# Gradio interface elements
hidden_units_input = gr.Slider(minimum=32, maximum=256, step=32, value=128, label="Hidden Units")
learning_rate_input = gr.Slider(minimum=0.001, maximum=0.1, step=0.001, value=0.01, label="Learning Rate")
epochs_input = gr.Slider(minimum=1, maximum=20, step=1, value=5, label="Epochs")

def plt_plot(x, y):
    # prepare some data
    # create a new plot
    plt.rcParams['figure.figsize'] = 6,4
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.plot(x, y)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    return fig

outputs = gr.Plot(label="Training Accuracy")
# Define the Gradio interface
def train_and_visualize(hidden_units, learning_rate, epochs):
  # Train the model
  model, train_losses, test_losses, test_accuracies = train_model(hidden_units, learning_rate, epochs)

  # Update loss and accuracy plots
  # loss_plot.update(np.vstack([train_losses, test_losses]))
  # accuracy_plot.update(test_accuracies)
  # loss_plot = plt_plot([i for i in range(len(train_losses))], train_losses)
  accuracy_plot = plt_plot([i for i in range(len(test_accuracies))], test_accuracies)
  return accuracy_plot

# Launch the Gradio interface
interface_train = gr.Interface(
    fn=train_and_visualize,
    inputs=[hidden_units_input, learning_rate_input, epochs_input],
    outputs=outputs,
    title="MNIST Digit Classification with Gradio",
    description="Train a simple neural network for MNIST digit classification and visualize loss/accuracy. You can also input a handwritten digit image for prediction."
)

label = gr.Label(num_top_classes=1)
interface_predict = gr.Interface(fn=predict, inputs="sketchpad", outputs=label, live=True)
demo = gr.TabbedInterface([interface_train, interface_predict], ["Train Model", "Predict Digit"])
demo.launch(share=True)

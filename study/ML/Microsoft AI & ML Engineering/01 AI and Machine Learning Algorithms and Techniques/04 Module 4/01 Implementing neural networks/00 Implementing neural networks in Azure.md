# Implementing neural networks in Azure

## Introduction
Azure offers tools such as Azure Machine Learning that facilitate the deployment and scaling of machine learning models, including neural networks.

A neural network is composed of layers of neurons, where each neuron processes inputs and passes the output to the next layer. In this guide, we will focus on building and training a simple feedforward neural network using the Azure Machine Learning SDK and PyTorch, a widely used deep learning library.

By the end of this lesson, you will be able to:

- Describe the steps needed for the basic implementation of a neural network using Python and Microsoft Azure. 

## Step-by-step guide to implementing neural networks

### Step 1: Set up an Azure Machine Learning workspace
Before implementing a neural network, we need to set up an Azure Machine Learning workspace. You can do this by following these steps:

```python
# Install the Azure Machine Learning SDK
pip install azureml-core
```

Next, create a new workspace or use an existing one:

```python
from azureml.core import Workspace

# Create or retrieve an existing Azure ML workspace
#ws = Workspace.get(name='myworkspace',
                      subscription_id='your-subscription-id',
                      resource_group='myresourcegroup',
                      location='eastus')

ws = Workspace.create(name='myworkspace',
                      subscription_id='your-subscription-id',
                      resource_group='myresourcegroup',
                      location='eastus')

# Write configuration to the workspace config file
ws.write_config(path='.azureml')
```

### Step 2: Build a simple neural network using PyTorch
Now that we have our workspace set up, let's define a simple neural network. PyTorch is well-suited for building neural networks, and it integrates seamlessly with Azure.

Here's a basic implementation of a feedforward neural network:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network with one hidden layer
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # Input layer (784 input features)
        self.fc2 = nn.Linear(128, 10)   # Output layer (10 classes)
        self.relu = nn.ReLU()           # Activation function

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Apply ReLU after the first layer
        x = self.fc2(x)             # Output layer
        return x

# Initialize the neural network
model = SimpleNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # For classification tasks
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent
```

In the above code, we:

- Defined a simple neural network with an input layer, one hidden layer, and an output layer.
- Used the ReLU (rectified linear unit) activation function, which is commonly used in neural networks.
- Defined a CrossEntropyLoss for multi-class classification and SGD (stochastic gradient descent) as the optimizer.

### Step 3: Train the neural network
After defining the model, we need to train it on a dataset. We will simulate training using the MNIST dataset (a set of handwritten digits) to demonstrate the process. First, we load the data and then train the model.

```python
from torchvision import datasets, transforms

# Load MNIST dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=32, shuffle=True)

# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Reset gradients

        # Forward pass
        outputs = model(inputs.view(-1, 784))  # Flatten input
        loss = criterion(outputs, labels)      # Compute loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
```

In this code:

- The MNIST dataset is loaded using torchvision.
- The network is trained for five epochs using stochastic gradient descent to update the weights.
- The input is flattened (from 28 × 28 images to 784 features) before being fed into the network.

### Step 4: Deploy the model on Azure
After training the model, you can deploy it to Azure Machine Learning for inference at scale. Here's how to register the model and deploy it as a web service:

```python
from azureml.core import Model

# Save the trained model
torch.save(model.state_dict(), 'simple_nn.pth')

# Register the model in Azure
model = Model.register(workspace=ws, model_path='simple_nn.pth', model_name='simple_nn')

# Deploying as a web service requires more steps such as creating a scoring script and environment.
```

At this point, the model can be registered and further steps can be taken to deploy it as an Azure web service.

## Guide summary
- **Workspace setup**: Using Azure ML SDK to create or retrieve a workspace.
- **Model building**: Creating a simple feedforward network with PyTorch, defining the architecture and activation function.
- **Training**: Using PyTorch's data loader and optimizer to train the neural network with backpropagation.
- **Deployment**: Registering the model for deployment on Azure.

## Conclusion
This reading has covered how to implement a neural network from scratch using PyTorch and deploy it with Azure Machine Learning. While this is a simple example, neural networks can be scaled to solve complex problems, and Azure provides the tools to do so efficiently.

Next, we'll move into hands-on activities where you'll build and deploy your neural network on Azure.

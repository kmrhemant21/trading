# Practice Activity: Implementing and Comparing Neural Network Architectures

## Introduction

In this activity, you will implement two different neural network architectures using **TensorFlow** and **PyTorch**. The objective is to compare the performance, ease of use, and key features of these two frameworks.

By the end of this activity, you will be able to:

- **Build and train** a neural network using both TensorFlow (Keras) and PyTorch.
- **Compare performance metrics** (accuracy and loss) of the models.
- **Identify differences** in implementation, ease of debugging, and flexibility between the two frameworks.

***

## Step-by-step Instructions

### Step 1: Set Up the Environment

Before starting, ensure that you have both TensorFlow and PyTorch installed in your environment. Install them using the following commands:

```bash
pip install tensorflow
pip install torch torchvision
```

Once installed, you’re ready to start implementing the models.

***

### Step 2: Load the Dataset

For this activity, you will use the **CIFAR-10 dataset**, which consists of 60,000 32 × 32 color images across 10 different classes. You will implement the same neural network architecture using both TensorFlow and PyTorch.

#### Load the CIFAR-10 Dataset in TensorFlow (Keras)
```python
import tensorflow as tf

# Load CIFAR-10 dataset
(train_images, train_activityels), (test_images, test_activityels) = tf.keras.datasets.cifar10.load_data()

# Normalize the images to a range of 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0
```

#### Load the CIFAR-10 Dataset in PyTorch
```python
import torch
import torchvision
import torchvision.transforms as transforms

# Define a transformation to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
```

**Explanation:**  
In both TensorFlow and PyTorch, the images are normalized so that pixel values range from 0 to 1 in TensorFlow and from –1 to 1 in PyTorch (using `transforms.Normalize`).  
In PyTorch, a DataLoader should be defined to handle batching and shuffling.

***

### Step 3: Define the Neural Network Architecture

For both frameworks, we will define the same neural network architecture—a simple **convolutional neural network (CNN)** consisting of two convolutional layers followed by two fully connected layers.

#### Define the CNN in TensorFlow (Keras)
```python
from tensorflow.keras import layers, models

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

#### Define the CNN in PyTorch
```python
import torch.nn as nn
import torch.nn.functional as F

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

**Explanation:**  
Both architectures consist of two convolutional layers followed by pooling, flattening, and fully connected layers.  
In TensorFlow, the input shape is explicitly defined, whereas PyTorch calculates the shape dynamically during the forward pass.

***

### Step 4: Compile the Model in TensorFlow and Define the Optimizer in PyTorch

#### Compile the Model in TensorFlow
```python
# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

#### Define the Optimizer in PyTorch
```python
import torch.optim as optim
import torch.nn as nn

# Make sure to define the model using the PyTorch-defined CNN
model = SimpleCNN()

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
```

***

### Step 5: Training the Neural Network

#### Train the Model in TensorFlow
```python
# Train the model for 10 epochs
model.fit(train_images, train_activityels, epochs=10, batch_size=32, validation_data=(test_images, test_activityels))
```

#### Train the Model in PyTorch
```python
# Training loop for PyTorch
for epoch in range(10):
    running_loss = 0.0
    for inputs, activityels in trainloader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, activityels)  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Optimize
        running_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}')
```

**Explanation:**  
TensorFlow simplifies the training process using the `.fit()` method, while PyTorch requires manual loops to handle the forward pass, backpropagation, and optimization steps.

***

### Step 6: Evaluate the Model

#### Evaluate the Model in TensorFlow
```python
# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_activityels)
print(f'Test accuracy: {test_acc}')
```

#### Evaluate the Model in PyTorch
```python
correct = 0
total = 0
with torch.no_grad():
    for inputs, activityels in testloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += activityels.size(0)
        correct += (predicted == activityels).sum().item()
print(f'Test accuracy: {100 * correct / total}%')
```

***

### Step 7: Compare and Analyze

After training both models, compare their performance:

- **Accuracy:** Compare the test accuracy of both models.
- **Ease of use:** Which framework was easier to implement? Did TensorFlow’s higher-level API make implementation smoother, or did PyTorch's flexibility offer better control?
- **Debugging:** Reflect on the debugging process. Did PyTorch’s dynamic graph provide better insight, or did TensorFlow’s simplified process make it easier?

***

## Deliverables

At the end of this activity, you should have:

- Your **complete code** for implementing and training the neural networks in both TensorFlow and PyTorch.
- A **comparison report** discussing:
    - The test accuracy of both models.
    - The ease of implementation in both frameworks.
    - Key differences, including which framework you preferred and why.

***

## Conclusion

By completing this activity, you’ve gained practical experience with two of the most popular deep learning frameworks. You’ve also compared how they differ in terms of implementation, ease of use, and performance. This will help you make informed decisions about which framework to use for your future projects.
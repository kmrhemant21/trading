# Practice Activity: Implementing a Model

## Introduction
In this activity, you will apply the concepts learned in the previous lessons by implementing an ML model from scratch. This hands-on exercise will guide you through the process of building, training, and evaluating a model using your choice of ML frameworks—TensorFlow, PyTorch, or Scikit-learn. The goal is to reinforce your understanding of model implementation. By the end of this activity, you will gain practical experience in deploying a working model.

By the end of this activity, you will:
- Build an ML model using a specified framework.
- Train the model on a provided dataset.
- Evaluate the model’s performance.
- Save and potentially reload the model for future use.

## Dataset
You will use the CIFAR-10 dataset, a well-known dataset consisting of 60,000 32x32 color images in 10 different classes. The dataset is already split into 50,000 training images and 10,000 test images. You can download the dataset directly through the TensorFlow or PyTorch libraries, or if using Scikit-learn, you may download it separately. <https://www.cs.toronto.edu/~kriz/cifar.html>

---

## Step-by-Step Instructions

### Step 1: Set Up Your Environment
#### Choose Your Framework
Decide whether you’ll be using TensorFlow, PyTorch, or Scikit-learn for this lab.

If you haven’t installed the framework yet, do so now using the following commands:

**TensorFlow example**
```bash
pip install tensorflow
```

**PyTorch example**
```bash
pip install torch torchvision
```

**Scikit-learn example**
```bash
pip install scikit-learn
```

#### Set Up Your Development Environment
Open your preferred coding environment (e.g., Jupyter Notebook, VSCode, or any Python IDE).

---

### Step 2: Import Libraries and Load the Dataset
#### Import the Required Libraries
Import the necessary libraries for your chosen framework.

**TensorFlow example**
```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
```

**PyTorch example**
```python
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
```

**Scikit-learn example**
```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
```

#### Load the CIFAR-10 Dataset
**TensorFlow example**
```python
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
```

**PyTorch example**
```python
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform

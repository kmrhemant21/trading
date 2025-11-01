# Walkthrough: Implementing a model (Optional)

## Introduction
In this walkthrough, we’ll go over the steps needed to successfully complete the lab assignment where you built, trained, and evaluated an ML model using either TensorFlow, PyTorch, or Scikit-learn. 

This guide will provide you with the correct implementation, along with explanations for each step, so you can ensure your solution aligns with best practices.

By the end of this reading, you will be able to: 

Set up your environment with the appropriate ML framework installed.

Import necessary libraries and load datasets using TensorFlow, PyTorch, or Scikit-learn.

Build, train, and evaluate an ML model.

Save and load models for future use.

## 1. Setting up your environment
The first step is to ensure that you have the necessary environment set up with the appropriate ML framework installed.

TensorFlow installation

```
pip install tensorflow
```

PyTorch installation

```
pip install torch torchvision
```

Scikit-learn installation

```
pip install scikit-learn
```

Once installed, you should have opened your preferred coding environment (e.g., Jupyter Notebook) and verified the installation by importing the relevant libraries.

## 2. Importing libraries and loading the dataset
Depending on your chosen framework, the necessary libraries and dataset should be loaded as follows:

TensorFlow

```
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
```

PyTorch

```
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
```

Scikit-learn: Since Scikit-learn does not directly support CIFAR-10, if you used a different dataset like MNIST:

```
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 3. Building your model
Your model’s architecture will vary depending on the framework, but the general structure for each should look something like this:

TensorFlow

```
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])
```

PyTorch

```
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
```

Scikit-learn: (example using RandomForest on the MNIST dataset)

```
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
```

## 4. Training your model
Training the model involves feeding the training data into the model and optimizing it to improve its performance. Here’s how you could have implemented the training process:

TensorFlow

```
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
```

PyTorch

```
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()  # zero the parameter gradients
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
```

Scikit-learn

```
clf.fit(X_train, y_train)
```

## 5. Evaluating your model
Evaluating your model is crucial to understanding how well it performs on unseen data. Below is how this can be done in each framework:

TensorFlow

```
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')
```

PyTorch

```
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
```

Scikit-learn

```
from sklearn.metrics import accuracy_score


y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test accuracy: {accuracy}')
```

## 6. Saving and loading the model
Finally, you were asked to optionally save your model so that it can be loaded and used later without retraining.

TensorFlow

```
model.save('my_cifar10_model.h5')
# Loading the model
loaded_model = tf.keras.models.load_model('my_cifar10_model.h5')
```

PyTorch

```
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)


# Loading the model
net = Net()
net.load_state_dict(torch.load(PATH))
```

Scikit-learn:

```
import joblib
joblib.dump(clf, 'random_forest_model.pkl')
# Loading the model
clf = joblib.load('random_forest_model.pkl')
```

## Conclusion
By following these steps, you should have successfully implemented an ML model using your chosen framework. The lab exercise was designed to give you hands-on experience with the full cycle of model development, from setting up the environment and data to training, evaluating, and saving the model. If you encounter any issues or deviations, use this walkthrough to troubleshoot and refine your approach.

This exercise reinforces the core concepts of model implementation, and you’re now better prepared to tackle more complex ML challenges in your future projects.
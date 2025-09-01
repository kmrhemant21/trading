# Practice activity: Implementing deep learning techniques

## Introduction

In this lab, you will implement three different deep learning techniques: feedforward neural networks (FNNs), convolutional neural networks (CNNs), and recurrent neural networks (RNNs). 

By the end of this activity, you will:

- Implement and train models using FNN, CNN, and RNN architectures.
- Compare the performance of each architecture on different types of data.
- Gain hands-on experience using TensorFlow's Keras API.

## Step-by-step instructions:

### Step 1: Set up the environment

Before starting, ensure you have TensorFlow installed. You can install it using the following command:

```python
pip install tensorflow
```

After installation, import the necessary libraries to build and train the neural networks:

```python
import tensorflow as tf
from tensorflow.keras import layers, models
```

### Step 2: Implement a feedforward neural network (FNN)

**Objective**  
Implement a simple FNN to perform classification on the Iris dataset.

**Steps**  
1. Load the Iris dataset
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# One-hot encode labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

2. Define the FNN architecture
```python
# Build the FNN model
model_fnn = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 output classes for the Iris dataset
])
```

3. Compile and train the model
```python
# Compile the model
model_fnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model_fnn.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
```

### Step 3: Implement a convolutional neural network (CNN)

**Objective**  
Implement a CNN to classify images from the CIFAR-10 dataset.

**Steps**  
1. Load the CIFAR-10 dataset
```python
# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
```

2. Define the CNN architecture
```python
# Build the CNN model
model_cnn = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 output classes for CIFAR-10
])
```

3. Compile and train the model
```python
# Compile the model
model_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model_cnn.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(test_images, test_labels))
```

### Step 4: Implement a recurrent neural network (RNN)

**Objective**  
Implement an RNN for time-series prediction using synthetic sequential data.

**Steps**  
1. Create synthetic sequential data for a sine wave
```python
import numpy as np

# Generate synthetic sine wave data
t = np.linspace(0, 100, 10000)
X = np.sin(t).reshape(-1, 1)

# Prepare sequences
def create_sequences(data, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(data) - seq_length):
        X_seq.append(data[i:i+seq_length])
        y_seq.append(data[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

seq_length = 100
X_seq, y_seq = create_sequences(X, seq_length)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
```
2. Define the RNN architecture
```python
# Build the RNN model
model_rnn = models.Sequential([
    layers.SimpleRNN(128, input_shape=(seq_length, 1)),
    layers.Dense(1)  # Output is a single value (next point in the sequence)
])
```

3. Compile and train the model
```python
# Compile the model
model_rnn.compile(optimizer='adam', loss='mse')

# Train the model
model_rnn.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
```

### Step 5: Compare performance

After training the models, compare their performance based on the following:

- FNN: classification accuracy on the Iris dataset
- CNN: classification accuracy on the CIFAR-10 dataset
- RNN: mean squared error for predicting the next value in the sine wave sequence

## Conclusion

In this lab, you implemented three distinct deep learning architectures—FNNs, CNNs, and RNNs—each tailored to specific types of data and tasks. By training and evaluating these models on the Iris dataset, CIFAR-10 images, and synthetic sine wave data, you observed the unique strengths and applications of each architecture. This hands-on experience will deepen your understanding of how different neural network designs can be applied effectively in various scenarios, preparing you for more advanced projects in the AI/ML domain. Remember to consider the nature of your data and the specific requirements of your task when choosing the appropriate architecture in your future work.
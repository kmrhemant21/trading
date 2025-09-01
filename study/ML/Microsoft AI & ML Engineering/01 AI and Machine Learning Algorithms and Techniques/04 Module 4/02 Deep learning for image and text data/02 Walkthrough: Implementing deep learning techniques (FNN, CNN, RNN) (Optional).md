# Walkthrough: Implementing deep learning techniques (FNN, CNN, RNN) (Optional)

## Introduction

In this walkthrough, you will be guided through three different neural network architectures—feedforward neural networks (FNNs), convolutional neural networks (CNNs), and recurrent neural networks (RNNs)—to gain experience with various deep learning techniques. This walkthrough will guide you through the proper implementation and training of each model.

By the end of this walkthrough, you will be able to:

- Have hands-on experience in implementing and training three distinct neural network architectures: FNNs for classification tasks, CNNs for image recognition, and RNNs for time-series prediction. 
- Understand the strengths and suitable applications of each architecture, enabling you to select the appropriate model for different machine learning problems.

## Implementing a Feedforward Neural Network

### Objective
We built a simple FNN to classify the Iris dataset.

### Solution walkthrough

**Step 1: Load and prepare the data**

You started by loading the Iris dataset, one-hot encoding the target labels, and splitting the data into training and testing sets.

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

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Step 2: Build the FNN**

A simple FNN architecture was created with two hidden layers. ReLU activation functions were applied to introduce non-linearity.

```python
# Build the FNN model
model_fnn = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 output classes
])
```

**Step 3: Compile and train the model**

The model was compiled with the Adam optimizer and categorical crossentropy loss, then trained for 20 epochs.

```python
# Compile and train the model
model_fnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_fnn.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
```

### Expected outcome

After training, the FNN should achieve accuracy levels above 90 percent on the Iris dataset, as the problem is relatively simple.

You can evaluate the model using:

```python
loss, accuracy = model_fnn.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')
```

## Implementing a convolutional neural network

### Objective

We used a CNN to classify images from the CIFAR-10 dataset.

### Solution walkthrough

**Step 1: Load and preprocess the data**

You normalized the images to have pixel values between zero and one to facilitate efficient training.

```python
# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalize the data
train_images, test_images = train_images / 255.0, test_images / 255.0
```

**Step 2: Build the CNN**

The CNN consisted of two convolutional layers followed by max-pooling layers, which reduce the spatial dimensions of the data.

```python
# Build the CNN model
model_cnn = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 output classes
])
```

**Step 3: Compile and train the model**

The model was compiled with sparse categorical crossentropy (suitable for integer labels) and trained for ten epochs.

```python
# Compile and train the model
model_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_cnn.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(test_images, test_labels))
```

### Expected outcome

After training, the CNN should achieve accuracy between 70–80 percent on the test data, as CIFAR-10 is a more challenging dataset.

Evaluate the model using:

```python
loss, accuracy = model_cnn.evaluate(test_images, test_labels)
print(f'Test Accuracy: {accuracy}')
```

## Implementing a recurrent neural network

### Objective

We built an RNN to predict the next value in a sine wave sequence, a classic example of time-series prediction.

### Solution walkthrough

**Step 1: Create the data**

A synthetic sine wave dataset was created and split into sequences for training and testing.

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

**Step 2: Build the RNN**

A simple RNN architecture was implemented with one recurrent layer and a single output neuron for predicting the next value in the sequence.

```python
# Build the RNN model
model_rnn = models.Sequential([
    layers.SimpleRNN(128, input_shape=(seq_length, 1)),
    layers.Dense(1)  # Single output for next value prediction
])
```

**Step 3: Compile and train the model**

The model was compiled using the mean squared error (MSE) loss function and trained for ten epochs.

```python
# Compile and train the model
model_rnn.compile(optimizer='adam', loss='mse')
model_rnn.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
```

### Expected outcome

The RNN should be able to predict the sine wave sequence with low MSE after training.

You can evaluate the model using:

```python
mse = model_rnn.evaluate(X_test, y_test)
print(f'Test MSE: {mse}')
```

## Summary of results

After completing the activity:

- **FNN**: you should have achieved more than 90 percent accuracy on the Iris dataset, showcasing that FNNs are well-suited for simple classification tasks.
- **CNN**: the CNN should have achieved around 70–80 percent accuracy on the CIFAR-10 dataset, highlighting the CNN's ability to recognize spatial features in image data.
- **RNN**: the RNN should have minimized MSE for predicting the sine wave, demonstrating the RNN's capacity for handling sequential data.

Each architecture has strengths for specific tasks, and understanding how to implement and optimize them is crucial for solving different types of problems.

## Conclusion

In this activity and walkthrough, you have gained practical experience in implementing and training three distinct neural network architectures. You learned how each model operates and its applicability to various types of data, preparing you to choose the appropriate architecture for your own machine learning projects. By understanding these foundational techniques in AI/ML engineering, you are now equipped to tackle more complex challenges in the field.
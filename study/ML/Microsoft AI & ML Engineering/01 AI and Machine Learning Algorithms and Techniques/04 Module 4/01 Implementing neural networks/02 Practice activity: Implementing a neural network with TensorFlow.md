# Practice activity: Implementing a neural network with TensorFlow

## Introduction

In this activity, you will implement a neural network using TensorFlow. This activity provides hands-on experience with TensorFlow, a powerful framework for machine learning and deep learning.

By the end of this activity, you will be able to:

- Build a simple feedforward neural network.
- Train the model on a dataset.
- Evaluate the model's performance using TensorFlow.

## Step-by-step guide for implementing a neural network using TensorFlow

### Step 1: Set up the environment

Before we start, ensure that TensorFlow is installed in your environment. You can install TensorFlow using the following command:

```bash
pip install tensorflow
```

Once TensorFlow is installed, import the necessary libraries to begin building the neural network.

```python
import tensorflow as tf
from tensorflow.keras import layers, models
```

### Step 2: Load and preprocess the dataset

For this activity, you will use the Fashion MNIST dataset, which consists of 28x28 grayscale images of fashion items, with 10 different classes. TensorFlow provides a built-in utility to load this dataset.

```python
# Load the Fashion MNIST dataset
(train_images, train_activityels), (test_images, test_activityels) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize the pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0
```

**Explanation**  
Normalizing the image data ensures that the neural network trains efficiently, as the pixel values range from 0 to 1 instead of from 0 to 255.

### Step 3: Define the neural network model

Now you will define the architecture of the neural network using the TensorFlow Keras API. The network will consist of:

- An input layer that flattens the 28 × 28 image into a one-dimensional vector.
- A hidden layer with 128 neurons and the ReLU activation function.
- An output layer with 10 neurons (one for each fashion class) using softmax activation.

```python
# Define the model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Input layer to flatten the 2D images
    layers.Dense(128, activation='relu'),  # Hidden layer with 128 neurons
    layers.Dense(10, activation='softmax') # Output layer with 10 classes
])
```

**Explanation**  
Flatten layer: Converts the 28 × 28 matrix into a one-dimensional array of 784 features.

Dense layers: A fully connected layer with 128 neurons and a ReLU activation function for the hidden layer, followed by an output layer with 10 neurons (one per class) that uses softmax to output probabilities.

### Step 4: Compile the model

After defining the architecture, you need to compile the model. During compilation, you specify:

- The optimizer: For this activity, we will use Adam, a widely used optimizer that adjusts learning rates during training.
- The loss function: Since this is a classification task, we will use sparse categorical crossentropy.
- Metrics: We will track accuracy to monitor model performance.

```python
# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

### Step 5: Train the model

With the model compiled, you can now train it on the Fashion MNIST dataset. We will train the model for 10 epochs with a batch size of 32.

```python
# Train the model
model.fit(train_images, train_activityels, epochs=10, batch_size=32)
```

**Explanation**  
Epochs: This refers to the number of times the model will go through the entire training dataset. Ten epochs is a good starting point for this task.

Batch size: This refers to the number of samples processed before the model's weights are updated. A batch size of 32 is a common choice.

### Step 6: Evaluate the Model

Once the model is trained, you can evaluate its performance on the test data. This will give you a sense of how well the model generalizes to new, unseen data.

```python
# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_activityels)

print(f'Test accuracy: {test_acc}')
```

**Explanation**  
The test accuracy metric provides insight into how well the model performs on the test dataset. You should aim for an accuracy of around 85–90 percent for this particular dataset.

### Step 7: Experimentation (optional)

After successfully implementing the basic neural network, you are encouraged to experiment with the model. Here are a few ideas:

- Add more hidden layers to make the network deeper.
- Change the number of neurons in the hidden layer.
- Try different activation functions, such as tanh or sigmoid, and observe their impact on the model's performance.
- Adjust the optimizer: Test how using SGD instead of Adam affects training and accuracy.

Example of adding another hidden layer:

```python
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),  # Additional hidden layer with 64 neurons
    layers.Dense(10, activation='softmax')
])
```

## Deliverables

By the end of this activity, you should have the following:

- Your complete code that builds, trains, and evaluates the neural network.
- A brief summary of the test accuracy you achieved, along with any modifications you made to the model architecture (if any).

## Conclusion

This activity guided you through the process of implementing a basic neural network using TensorFlow. You now have hands-on experience with defining a neural network, training it on a dataset, and evaluating its performance. Experiment with the architecture to better understand how changes in layers, neurons, and activation functions affect the model's learning process.
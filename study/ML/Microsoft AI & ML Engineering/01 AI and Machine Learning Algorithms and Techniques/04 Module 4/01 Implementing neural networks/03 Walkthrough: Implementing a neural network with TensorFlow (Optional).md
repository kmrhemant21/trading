# Walkthrough: Implementing a neural network with TensorFlow (Optional)

## Introduction

In this walkthrough, you will build a neural network using TensorFlow, train it on the Fashion MNIST dataset, and evaluate its performance. This guide provides a step-by-step walkthrough of the solution, ensuring that you understand the key implementation details.

By the end of this walkthrough, you will be able to:

- Load and preprocess a dataset using TensorFlow.
- Define a neural network architecture using the Keras Sequential API.
- Compile and train the neural network on a dataset.
- Evaluate the model's performance using test data.
- Experiment with different model parameters to enhance accuracy.

## Step-by-step walkthrough for building a neural network using TensorFlow

### Step 1: Load the dataset

The first step was to load the Fashion MNIST dataset. TensorFlow provides an easy utility for loading this data, which contains 70,000 grayscale images of fashion items across 10 categories.

```python
import tensorflow as tf

# Load the Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize the pixel values between 0 and 1
train_images = train_images / 255.0
test_images = train_images / 255.0
```

**Explanation**

Normalizing pixel values to a range of 0 to 1 ensures efficient model training and helps avoid issues with large input values that can negatively impact the optimization process.

### Step 2: Define the neural network model

Next, you defined the architecture of the neural network using the Sequential API in TensorFlow. The network consisted of:

- An input layer that flattens the 28 × 28 pixel images into a one-dimensional vector.
- A hidden layer with 128 neurons and a ReLU (rectified linear unit) activation function.
- An output layer with 10 neurons using the softmax activation function.

```python
from tensorflow.keras import layers, models

# Define the model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Input layer that flattens the image
    layers.Dense(128, activation='relu'),  # Hidden layer with 128 neurons
    layers.Dense(10, activation='softmax') # Output layer with 10 classes
])
```

**Explanation**

- **Flatten layer**: This transforms each 28 × 28 image into a 1D array of 784 values, preparing it for the fully connected layers.
- **Dense layers**: The hidden layer with 128 neurons and ReLU activation adds nonlinearity, allowing the model to capture complex patterns. The output layer uses softmax activation to produce probabilities for each of the 10 classes.

### Step 3: Compile the model

Before training, you compiled the model by specifying:

- The optimizer: Adam was chosen for this task because it adjusts learning rates automatically during training.
- The loss function: Sparse categorical crossentropy is used for multi-class classification.
- Metrics: Accuracy is used to track the model's performance.

```python
# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

**Explanation**

- **Adam optimizer**: Adam is a popular optimizer for deep learning models due to its adaptive learning rate, which often leads to faster convergence.
- **Sparse categorical cross-entropy**: This is a suitable loss function when dealing with integer labels (i.e., when each label is a class index).
- **Accuracy metric**: The accuracy metric helps you monitor how well the model is predicting the correct classes during training and evaluation.

### Step 4: Train the model

You trained the model on the Fashion MNIST dataset using the following code (the model was trained for 10 epochs, with a batch size of 32):

```python
# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=32)
```

**Explanation**

- **Epochs**: This refers to the number of times the model sees the entire training dataset. Training for 10 epochs is generally a good start for this problem.
- **Batch size**: A batch size of 32 means the model updates its weights after processing 32 samples, balancing computational efficiency and model performance.

### Step 5: Evaluate the model

Once the model was trained, the next step was to evaluate its performance on the test dataset. This checks how well the model generalizes to unseen data.

```python
# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)

print(f'Test accuracy: {test_acc}')
```

**Expected output**

You should expect to see a test accuracy between 85 and 90 percent. This accuracy is reasonable for a simple neural network on the Fashion MNIST dataset.

**Explanation**

- **Test loss and accuracy**: The test loss indicates how well the model's predictions match the true labels on the test data, while the test accuracy shows the percentage of correct predictions.

### Experimentation (optional)

For learners who wanted to explore more, the activity offered opportunities to experiment with different parameters, such as:

- Adding more hidden layers to make the model deeper and potentially more accurate.
- Increasing or decreasing the number of neurons in the hidden layer to adjust the model's capacity.
- Changing the activation function to something other than ReLU, such as sigmoid or tanh, and observing the impact.
- Using a different optimizer, such as stochastic gradient descent (SGD), and comparing its performance to Adam.

Example of adding an additional hidden layer:

```python
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),  # New hidden layer with 64 neurons
    layers.Dense(10, activation='softmax')
])
```

## Summary of results

After completing this activity, you should have:

- Successfully implemented a simple neural network using TensorFlow.
- Achieved a test accuracy of around 85–90 percent on the Fashion MNIST dataset.
- Gained experience in tweaking the model's architecture, such as modifying the number of layers, neurons, and activation functions.

## Conclusion

This walkthrough provided the key steps involved in building, training, and evaluating a neural network using TensorFlow. It reinforced your understanding of how to use the Keras API while also providing opportunities for experimentation. You now have a solid foundation for building more complex neural networks and tackling more challenging datasets.
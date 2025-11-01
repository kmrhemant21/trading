# Practice activity: Applying transfer learning

## Introduction
In this hands-on activity, you will implement transfer learning to solve a classification task using a pretrained model. Transfer learning allows you to use knowledge from pretrained models to solve new, related tasks effectively. This activity helps you to analyze a pretrained model's architecture, fine-tune it for a specific dataset, and evaluate its performance against a baseline model.

By the end of this activity, you will be able to:

- Analyze the architecture and purpose of a pretrained model.
- Fine-tune a pretrained model for a specific dataset.
- Evaluate the model's performance and compare it to a baseline.
- Examine differences in learned features before and after fine-tuning.

## Step-by-step guide to applying transfer learning
This activity will guide you through the following steps:

1. Step 1: Load and prepare a dataset.
2. Step 2: Examine a pretrained model.
3. Step 3: Fine-tune the model.
4. Step 4: Evaluate the model.
5. Step 5: Compare to a baseline model.

### Step 1: Load and prepare the dataset
To begin, we will use the CIFAR-10 dataset, a widely used dataset in computer vision tasks. CIFAR-10 consists of 60,000 32x32 RGB images evenly distributed across 10 distinct classes. These classes represent everyday objects, animals, and vehicles:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

Each image is labeled with one of these classes, providing a diverse dataset for evaluating the model's classification performance. The dataset will be split into three subsets for effective training, validation, and testing. This split ensures that the model generalizes well to unseen data, and validation allows fine-tuning of hyperparameters.

We will preprocess the dataset to normalize pixel values (scaling them between 0 and 1) and convert labels into a one-hot encoded format suitable for classification tasks. Finally, we'll create a validation set by splitting a portion of the training data, ensuring proper evaluation during training.

```python
# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert labels to one-hot encoded format
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Split training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Display dataset information
print(f"Training samples: {x_train.shape[0]}, Validation samples: {x_val.shape[0]}, Test samples: {x_test.shape[0]}")
```

### Step 2: Examine a pretrained model
Transfer learning leverages pretrained models such as MobileNetV2. By examining the model's architecture, we can understand its layers and visualize features learned during pretraining.

```python
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt

# Load the MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Display model architecture
base_model.summary()

# Inspect layers and find the first layer with weights
layer_with_weights = None
for layer in base_model.layers:
    if layer.get_weights():
        layer_with_weights = layer
        break

if layer_with_weights:
    print(f"First layer with weights: {layer_with_weights.name}")
    weights = layer_with_weights.get_weights()[0]
    if weights.ndim == 4:  # Check if weights are compatible for visualization
        plt.imshow(weights[:, :, :, 0], cmap='viridis')
        plt.title(f'Visualizing Features from {layer_with_weights.name}')
        plt.show()
    else:
        print(f"Cannot visualize weights from layer {layer_with_weights.name}: incompatible dimensions.")
else:
    print("No layers with weights found in the model.")
```

### Step 3: Fine-tune the model
Fine-tuning a pretrained model involves freezing its base layers and adding custom layers specific to your task. We will use the CIFAR-10 dataset for classification.

```python
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# Freeze the base layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
output = Dense(10, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32)

# Print final validation accuracy
val_accuracy = history.history['val_accuracy'][-1]
print(f"Final Validation Accuracy: {val_accuracy:.2f}")
```

### Step 4: Evaluate the model
In this step, you will evaluate the performance of the fine-tuned model on the test set and analyze the intermediate feature maps to understand what the model has learned. These feature maps provide insights into how the model processes images at different stages, helping us to interpret the features that contribute to its predictions.

```python
# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Analyze feature maps from an intermediate layer
from tensorflow.keras.models import Model
intermediate_layer_model = Model(inputs=model.input, outputs=base_model.get_layer('block_1_expand_relu').output)
intermediate_output = intermediate_layer_model.predict(x_test[:5])

# Display feature map dimensions
print(f"Feature maps for the first test sample have shape: {intermediate_output[0].shape}")
```

### Step 5: Compare to a baseline model
To understand the value of transfer learning, we'll train a simple CNN from scratch and compare its performance to the fine-tuned model.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten

# Define a baseline model
baseline_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile and train the baseline model
baseline_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
baseline_history = baseline_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32)

# Evaluate the baseline model
baseline_loss, baseline_accuracy = baseline_model.evaluate(x_test, y_test, verbose=2)
print(f"Baseline Model Test Accuracy: {baseline_accuracy:.2f}")

# Summarize comparison
print(f"Transfer Learning Test Accuracy: {test_accuracy:.2f}")
print(f"Baseline Model Test Accuracy: {baseline_accuracy:.2f}")
```

### Step 6: Visualize and reflect
Visualize the performance of both models to reflect on the impact of transfer learning.

```python
# Compare training and validation accuracy
plt.plot(history.history['accuracy'], label='Transfer Learning Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Transfer Learning Validation Accuracy')
plt.plot(baseline_history.history['accuracy'], label='Baseline Training Accuracy')
plt.plot(baseline_history.history['val_accuracy'], label='Baseline Validation Accuracy')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Performance Comparison: Transfer Learning vs Baseline')
plt.grid(True)  # Add grid for clarity
plt.show()
```

## Expected observations and interpretations

### Transfer learning model:

- **Observation**: faster convergence and higher validation accuracy compared to the baseline model.
- **Interpretation**: transfer learning accelerates training and improves generalization, particularly with limited data.

### Baseline model:

- **Observation**: slower convergence and lower overall accuracy.
- **Interpretation**: training a model from scratch requires significantly more data and time to achieve comparable results.

### Validation vs. training accuracy:

- **Observation**: the transfer learning model shows a smaller gap between training and validation accuracy.
- **Interpretation**: this indicates reduced overfitting, as pretrained models have prior knowledge that helps them to generalize better.

## Real-world scenario
Imagine you are working on an e-commerce platform that categorizes product images into different categories for better search and recommendation features. The dataset you are working with consists of thousands of product images, but building a model from scratch is resource-intensive and time-consuming. By applying transfer learning with a pretrained model, you can leverage existing knowledge from large-scale image datasets, fine-tune it to your specific categories, and achieve higher accuracy with less data. This ensures that the model can effectively learn category-specific features, leading to better product categorization and user experience.

## Conclusion
Reflect on whether you were able to successfully implement transfer learning for the classification task. Consider any challenges you faced during the activity and how transfer learning improved model performance compared to training a model from scratch. Transfer learning is a powerful technique that leverages pretrained knowledge, allowing you to save time and achieve better results, even with limited data.

Try applying transfer learning techniques to a new dataset of your choice. Experiment with different pretrained models, fine-tuning methods, and evaluation strategies to deepen your understanding of how transfer learning enhances model performance and adaptability.

# Practice activity: Implementing optimization techniques

## Introduction
In this activity, you will apply key optimization techniques to a machine learning (ML) model to improve its response time and accuracy. By the end of this activity, you'll have hands-on experience with techniques such as model pruning, quantization, and hyperparameter tuning. These methods are designed to help you optimize your model for real-world applications where speed and precision are essential.

By the end of this activity, you will be able to:

- Apply model pruning to reduce model complexity and improve performance
- Explain how quantization helps in speeding up inference.

## Step-by-step process to optimize ML models
This reading will guide you through the following steps:

1. Step 1: Set up the environment
2. Step 2: Apply model pruning
3. Step 3: Apply model quantization
4. Step 4: Perform hyperparameter tuning
5. Step 5: Evaluate the trade-offs

### Step 1: Set up the environment
**Instructions**  
Create a new Jupyter notebook.  Start by setting up your Python 3.8 Azure ML kernel. Ensure you have the necessary libraries installed, including Scikit-Learn, TensorFlow, or PyTorch, depending on the model you're working with.

**Setup commands**
```python
import subprocess

# Install scikit-learn and tensorflow using pip
subprocess.check_call(["pip", "install", "scikit-learn", "tensorflow"])
!pip install tensorflow-model-optimization
#warning! This runs ont eh 3.8 -- AzureML kernel by default... 
!conda install tf-keras
#need tf-keras for tf-model-optimization to work
```

**Explanation**  
This setup ensures that you have the right tools to apply optimization techniques such as pruning and quantization.

### Step 2: Apply model pruning
Begin by applying model pruning to reduce the size of your model. In this step, you'll remove unnecessary neurons or branches that have minimal impact on the overall accuracy.

**Instructions**
1. Train the model on a dataset.
2. Apply pruning by removing neurons or branches with the least contribution to model performance.

**Example (Tensorflow)**
```python
import tensorflow_model_optimization as tfmot
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a simple model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Apply pruning to the model
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.0, final_sparsity=0.5, begin_step=0, end_step=1000)
}
pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

# Compile the pruned model
pruned_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the pruned model to finalize pruning
callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
pruned_model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test), callbacks=callbacks)

# Strip pruning wrappers to remove pruning-specific layers and metadata
pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
```

**Explanation**  
Pruning reduces the size of the model and the computational resources required during inference without sacrificing much accuracy. This is especially useful for deploying models on resource-constrained devices.

### Step 3: Apply model quantization
Next, apply quantization to your model. This will reduce the precision of the model's parameters, speeding up inference, especially on devices with limited computational resources.

**Instructions**
1. Quantize the model's weights from 32-bit floating-point to 8-bit integers.
2. Evaluate the model's performance after quantization to ensure that accuracy remains acceptable.

**Example (Tensorflow lite quantization)**
```python
# Convert the pruned model to a TensorFlow Lite quantized model
converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()
```

**Explanation**  
Quantization reduces the size of the model and speeds up inference by lowering the precision of its weights, which is particularly useful for deployment on mobile devices or edge devices.

### Step 4: Perform hyperparameter tuning
Perform hyperparameter tuning to find the optimal combination of parameters that maximize accuracy while maintaining reasonable response times.

**Instructions**
1. Use a grid search or a randomized search to explore different hyperparameter combinations.
2. Choose the parameters that offer the best balance between model performance and computational cost.

**Example (grid search in Scikit-Learn)**
```python
# Measure accuracy of the quantized model using the test set
interpreter = tf.lite.Interpreter(model_content=quantized_model)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# Evaluate accuracy
correct_predictions = 0
for i in range(len(x_test)):
    input_data = x_test[i:i+1].astype('float32')
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_index)
    predicted_label = output.argmax()
    if predicted_label == y_test[i]:
        correct_predictions += 1

accuracy = correct_predictions / len(x_test)
print(f'Quantized model accuracy: {accuracy * 100:.2f}%')
```

**Explanation**  
Hyperparameter tuning helps you find the best settings for your model, improving accuracy while minimizing computational requirements.

### Step 5: Evaluate the trade-offs
Finally, evaluate the trade-offs between model complexity, accuracy, and response time. Document the changes in performance (e.g., accuracy and latency) before and after applying these optimization techniques.

**Instructions**
1. Compare the original model's accuracy and response time with the optimized model.
2. Identify the key trade-offs. (For example, did pruning reduce accuracy but speed up inference significantly?)
3. Document your findings, and decide which optimizations are most suitable for your use case.

**Explanation**  
Understanding the trade-offs between accuracy and speed helps you make informed decisions about which optimizations to apply in production environments.

## Conclusion
In this activity, you applied several optimization techniques—pruning, quantization, and hyperparameter tuning—to improve the performance of your ML model. By balancing accuracy with response time, you can ensure that your model is not only accurate but also efficient and ready for real-world applications.

# Practice activity: Testing and optimizing the agent

## Introduction
In this activity, you will apply various testing and optimization techniques to evaluate and improve the performance of a machine learning agent. By the end of this activity, you will have measured key performance metrics such as accuracy, response time, and resource utilization and implemented optimization methods like pruning, quantization, and hyperparameter tuning.

By the end of this activity, you will be able to:

- Measure the performance of a machine learning agent by evaluating accuracy, response time, resource utilization, and error rate.
- Apply optimization techniques such as model pruning, quantization, and feature selection to improve agent efficiency.
- Perform stress testing to assess the scalability of the agent.
- Analyze trade-offs between accuracy and response time.
- Continuously monitor performance and make adjustments as necessary.

## Step-by-step process to test and optimize a machine learning agent
Create a new Jupyter notebook. Make sure you have the Python 3.8 Azure ML kernel selected. Start by training a model. Once the model has finished training, prune and quantize it. Specific details about the training process, the pruning process, and the quantization process can be found in other parts of this course. 

The remainder of this reading will guide you through the following steps:

1. Step 1: Measuring performance metrics
2. Step 2: Applying model pruning
3. Step 3: Quantizing the model
4. Step 4: Feature selection
5. Step 5: Stress testing
6. Step 6: Evaluating trade-offs
7. Step 7: Continuous monitoring and retraining

### Step 1: Measuring performance metrics
**Instructions**  
Start by measuring the agent's key performance metrics: accuracy, precision, response time, and resource utilization.

**What to do**
- Use a test dataset to measure accuracy and precision.
- Measure response time for multiple predictions.
- Monitor the agent's CPU and memory usage during prediction.

**Example (accuracy and precision calculation)**
```python
import tensorflow as tf 
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score, precision_score

# y_true: actual labels, y_pred: agent's predictions
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='binary')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
```

**Example (response time calculation)**
```python
import time

start_time = time.time()
agent.predict(input_data)
end_time = time.time()

response_time = end_time - start_time
print(f'Response Time: {response_time} seconds')
```

**Explanation**  
Evaluating these metrics helps us understand how well the agent is performing and whether it is operating efficiently.

### Step 2: Applying model pruning
**Instructions**  
Apply model pruning to reduce the size of the agent without significantly sacrificing accuracy.

**What to do**
- Identify and remove unnecessary neurons or layers in the model to reduce complexity.
- Retrain the pruned model and measure its performance against the original model.

**Example (model pruning in TensorFlow)**
```python
import tensorflow_model_optimization as tfmot

# Define pruning parameters
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0, final_sparsity=0.5, begin_step=0, end_step=1000
    )
}

# Apply pruning to the Sequential model
pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

# Compile the pruned model
pruned_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Retrain the pruned model to finalize pruning
callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
pruned_model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test), callbacks=callbacks)

# Strip pruning wrappers to remove pruning-specific layers and metadata
pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
```

**Explanation**  
Pruning simplifies the model by removing components that don't significantly contribute to its performance, helping to improve inference time and resource efficiency.

### Step 3: Quantizing the model
**Instructions**  
Apply quantization to speed up inference by reducing the precision of the model's weights.

**What to do**
- Convert the model's weights from 32-bit floating point to 8-bit integers using quantization.
- Measure the impact of quantization on response time and accuracy.

**Example (quantization with TensorFlow Lite)**
```python
converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()
```

**Explanation**  
Quantization reduces the precision of the model's weights, leading to faster inference times, particularly when deploying the agent on devices with limited computational resources.

### Step 4: Feature selection
**Instructions**  
Use feature selection to remove irrelevant features and improve the model's efficiency.

**What to do**
- Apply a feature selection method such as recursive feature elimination (RFE) or principal component analysis.
- Retrain the model using only the most important features and evaluate its performance.

**Example (feature selection with RFE in scikit-learn)**
```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
rfe = RFE(model, n_features_to_select=5)
rfe = rfe.fit(X_train, y_train)

# Train the model with selected features
X_train_selected = rfe.transform(X_train)
model.fit(X_train_selected, y_train)
```

**Explanation**  
Feature selection reduces the complexity of the model by focusing only on the most important features, which can improve both speed and accuracy.

### Step 5: Stress testing
**Instructions**  
Perform stress testing to evaluate how well the agent scales under heavy workloads or large datasets.

**What to do**
- Simulate high traffic or data loads by processing large batches of inputs or concurrent requests.
- Measure how the agent's response time, accuracy, and resource usage change under stress.

**Example**
```python
for i in range(1000):
    agent.predict(input_data)
```

**Explanation**  
Stress testing helps identify performance bottlenecks and ensures the agent can handle large volumes of data or high user traffic without significant performance degradation.

### Step 6: Evaluating trade-offs
**Instructions**  
Analyze the trade-offs between performance metrics, such as accuracy and response time, after applying the optimization techniques.

**What to do**
- Compare the agent's accuracy, response time, and resource utilization before and after optimization.
- Determine whether the optimizations led to an acceptable balance between accuracy and speed.

**Explanation**  
Finding the right balance between speed and accuracy ensures that the agent is optimized for its intended application, whether that requires real-time processing or high predictive accuracy.

### Step 7: Continuous monitoring and retraining
**Instructions**  
Set up a continuous monitoring system to track the agent's performance over time. Regularly retrain the model to adapt to changes in data or user behavior.

**What to do**
- Monitor key performance metrics such as accuracy, response time, and resource utilization in real time.
- Schedule regular retraining sessions to ensure the agent remains effective.

**Explanation**  
Continuous monitoring helps maintain the agent's effectiveness in production environments, where data and usage patterns may evolve over time.

## Conclusion
In this activity, you evaluated and optimized a machine learning agent by measuring performance metrics, applying pruning and quantization, and conducting stress testing. By balancing the trade-offs between accuracy, response time, and resource utilization, you ensured that the agent is both effective and efficient in real-world applications.

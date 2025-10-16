# Practice activity: Evaluating agent effectiveness

## Introduction
 
In this activity, you will evaluate the effectiveness of a machine learning (ML) agent based on key performance metrics such as accuracy, response time, and resource utilization. By the end of this activity, you'll understand how to assess the performance of the agent using various evaluation methods and identify areas for improvement.

By the end of this activity, you will be able to:

- Measure the accuracy, precision, and error rate of your ML agent.
- Evaluate the agent's response time and resource utilization under different conditions.
- Perform stress testing to understand the scalability of the agent.

## Step-by-step process to evaluate the effectiveness of an ML agent
This reading will guide you through the following steps:

1. Step 1: Set up the environment
2. Step 2: Measure accuracy and precision
3. Step 3: Evaluate response time
4. Step 4: Resource utilization
5. Step 5: Perform stress testing
6. Step 6: Use benchmarking and cross-validation
7. Step 7: Collect user feedback (if applicable)

### Step 1: Set up the environment
#### Instructions
Create a new Jupyter notebook. Start by ensuring your ML agent is properly set up and operational. You will need tools for measuring performance metrics such as accuracy, response time, and error rate.  

#### Setup commands
```python
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
```

This is the same model that we have trained in previous versions of this activity. It is trained on the MNIST dataset using TensorFlow.

#### Explanation
This setup will allow you to evaluate various performance metrics for the agent, including accuracy, response time, and more.

### Step 2: Measure accuracy and precision
Begin by evaluating the agent's accuracy and precision. These metrics will help you understand how well the agent is performing in terms of making correct predictions.

#### Instructions
- Run the agent on a test dataset, and measure the number of correct predictions (accuracy).
- Calculate precision by evaluating how often the positive predictions are correct.

#### Example (scikit-learn)
```python
from sklearn.metrics import accuracy_score, precision_score
import numpy as np

# Make predictions on the test set
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Calculate precision (average='weighted' to handle multiple classes)
precision = precision_score(y_test, y_pred, average='weighted')
print(f'Precision: {precision:.4f}')
```

#### Explanation
Accuracy and precision are key metrics for assessing how reliably the agent performs in making predictions. High accuracy with low precision might indicate that the agent is making too many irrelevant predictions.

### Step 3: Evaluate response time
Measure how quickly the agent processes inputs and returns predictions. Response time is critical in applications requiring real-time decision-making.

#### Instructions
- Use the time module to measure how long it takes the agent to make predictions on a given input.
- Record the average response time across multiple predictions.

#### Example
```python
import time

# Measure response time for multiple iterations
start_time = time.time()
for _ in range(25):
    model.predict(x_test)
end_time = time.time()

average_response_time = (end_time - start_time) / 25
print(f"Average Response Time: {average_response_time:.4f} seconds")
```

#### Explanation
Measuring response time is important for understanding the agent's efficiency, especially in environments that require quick decision-making. A lower response time is essential for applications like recommendation engines, where delays can affect the user experience.

### Step 4: Resource utilization
Evaluate the agent's resource consumption, including CPU and memory usage, to ensure that it operates efficiently under various workloads. High resource consumption might indicate the need for optimization.

#### Instructions
- Monitor the agent's CPU and memory usage while it processes inputs.
- Use tools such as psutil to measure resource utilization in real time.

#### Example
```python
import psutil

# Monitor resource usage
cpu_usage = psutil.cpu_percent()
memory_usage = psutil.virtual_memory().percent

# for better results, measure CPU usage while inference is active,
# and measure memory usage against a baseline before the model is loaded
print(f"CPU Usage: {cpu_usage}%")
print(f"Memory Usage: {memory_usage}%")
```

#### Explanation
Efficient resource utilization is key for agents deployed in production environments, especially when handling large datasets or high user traffic. Monitoring this ensures the agent doesn't consume more resources than necessary.

### Step 5: Perform stress testing
Perform stress testing by running the agent under extreme conditions to evaluate its scalability. This will help identify bottlenecks and areas that need optimization.

#### Instructions
- Simulate high data volumes or concurrent requests to see how the agent performs under load.
- Measure the impact on response time, accuracy, and resource utilization during stress testing.

#### Example
```python
import numpy as np
import time

# Ensure correct shape before repeating
print("Original x_test shape:", x_test.shape)  # Expected: (10000, 28, 28)

# Properly duplicate test data along batch axis
large_input = np.repeat(x_test, 10, axis=0)  # Expands batch size only

# Verify new shape
print("Large input shape after fix:", large_input.shape)  # Should be (100000, 28, 28)

# Measure performance under stress
start_time = time.time()
model.predict(large_input)  # Now matches model input (batch_size, 28, 28)
end_time = time.time()

print(f"Response Time under Stress (Reduced Size): {end_time - start_time:.4f} seconds")
```

#### Explanation
Stress testing helps assess how well the agent can scale and whether its performance degrades under high data volumes or heavy loads.

### Step 6: Use benchmarking and cross-validation
Use benchmarking and cross-validation to compare your agent's performance against standard models or other agents performing similar tasks.

#### Instructions
- Benchmark the agent against a baseline model (e.g., a simpler ML algorithm).
- Use cross-validation to evaluate the agent on different subsets of data and ensure consistent performance.

#### Example (cross-validation in Scikit-Learn)
```python
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# Example data generation for demonstration (replace with actual data)
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
agent_model = RandomForestClassifier()  # Replace with your actual model

# Perform 5-fold cross-validation
cv_scores = cross_val_score(agent_model, X, y, cv=5)

# Print the cross-validation scores for each fold
print(f'Cross-Validation Scores: {cv_scores}')

# Print the mean and standard deviation of the scores
print(f'Mean CV Score: {cv_scores.mean():.4f}')
print(f'Standard Deviation of CV Scores: {cv_scores.std():.4f}')
```

#### Explanation
Benchmarking and cross-validation provide a robust way to compare the agent's effectiveness and ensure that it generalizes well to new data.

### Step 7: Collect user feedback (if applicable)
If your agent interacts with users, collect user feedback through surveys or other feedback mechanisms to gauge satisfaction and identify areas for improvement.

#### Instructions
- Distribute feedback forms or surveys to users who have interacted with the agent.
- Analyze the feedback to identify recurring issues or opportunities for improvement.

#### Explanation
User feedback is a valuable metric for agents designed for interaction, as it provides direct insights into user satisfaction and the agent's overall effectiveness.

## Conclusion
In this activity, you evaluated the effectiveness of your ML agent by measuring accuracy, precision, response time, resource utilization, and scalability. These evaluations are crucial for ensuring your agent remains effective in real-world applications. By tracking these factors, you can ensure your model is reliable, efficient, and ready for deployment.

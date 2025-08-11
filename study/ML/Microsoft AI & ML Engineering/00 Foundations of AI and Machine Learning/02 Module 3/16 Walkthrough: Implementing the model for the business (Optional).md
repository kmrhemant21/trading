# Walkthrough: Implementing the model for the business (Optional)

## Introduction

This walkthrough provides a detailed solution to the activity in which you were asked to develop, train, and prepare an ML model for business deployment. The project involved creating a customer churn prediction model for a telecommunications company, optimizing it for deployment, and ensuring it could be integrated into a production environment. Below, we’ll review the steps you should have followed and explain the rationale behind each one.

By the end of this walkthrough, you will be able to: 

- Set up your environment for developing an ML model.
- Import necessary libraries and load datasets.
- Preprocess data for model training.
- Develop, train, and evaluate an ML model using TensorFlow, PyTorch, or Scikit-learn.
- Optimize and prepare the model for deployment.

---

## 1. Setting up your environment

### Framework selection

- **TensorFlow**: Selected for its scalability, production readiness, and robust ecosystem, making it ideal for business deployment.
- **PyTorch**: Chosen for its flexibility and ease of experimentation; particularly valuable during the development phase.
- **Scikit-learn**: Selected for its simplicity and effectiveness in traditional ML tasks; suitable for straightforward implementation.

### Library installation

Ensure you have the necessary libraries installed:

```bash
pip install tensorflow      # For TensorFlow
pip install torch torchvision  # For PyTorch
pip install scikit-learn     # For Scikit-learn
```

### Environment setup

Open your coding environment (e.g., Jupyter Notebook or VSCode), and import the necessary libraries. This ensures that your environment is ready for data processing, model development, and evaluation.

---

## 2. Importing libraries and loading the dataset

### TensorFlow example

```python
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
```

### PyTorch example  

```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
```

### Scikit-learn example

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

### Loading the dataset

Using the provided link to the dataset, download the file and import this in your notebook. Make sure it is named correctly and is the correct file type.

The customer churn dataset was loaded into the environment using pandas:

```python
import pandas as pd
data = pd.read_csv('customer_churn.csv')
```

### Dataset exploration

A brief exploration of the dataset was performed to understand its structure:

```python
print(data.head())
print(data.info())
```

This step is crucial for identifying the features available, understanding data types, and checking for missing values.

---

## 3. Data preprocessing

### Handling missing values

Any missing values in the dataset were handled appropriately, either by dropping rows/columns or by filling them with suitable values:

```python
data = data.dropna()  # Example: Dropping rows with missing values
```

### Encoding categorical variables

Categorical variables were converted into numerical format using one-hot encoding:

```python
data = pd.get_dummies(data, drop_first=True)
```

### Data splitting

The dataset was split into training and testing sets to evaluate the model’s performance:

```python
from sklearn.model_selection import train_test_split

X = data.drop('Churn', axis=1)
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 4. Developing and training the model

### Model selection

Depending on the framework, the model architecture was designed to predict customer churn.

#### TensorFlow example

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

#### PyTorch example

```python
class ChurnModel(nn.Module):
    def __init__(self):
        super(ChurnModel, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = nn.functional.dropout(x, 0.5, training=self.training)
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

model = ChurnModel()
```

#### Scikit-learn example

```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
```

### Model training

The model was trained on the training data. Monitoring the training process helps ensure that the model is learning correctly.

#### TensorFlow example

```python
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
```

#### PyTorch example

```python
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (simplified example)
for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    outputs = model(torch.tensor(X_train.values).float())
    loss = criterion(outputs.squeeze(), torch.tensor(y_train.values).float())
    loss.backward()
    optimizer.step()
```

#### Scikit-learn example

```python
model.fit(X_train, y_train)
```

---

## 5. Evaluating and optimizing the model

### Model evaluation

The model’s performance was evaluated using the test set. Key metrics such as accuracy were calculated to assess the model's effectiveness.

#### TensorFlow example

```python
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')
```

#### PyTorch example

```python
model.eval()
outputs = model(torch.tensor(X_test.values).float())
predictions = (outputs.squeeze().detach().numpy() > 0.5).astype(int)
accuracy = np.mean(predictions == y_test.values)
print(f'Test accuracy: {accuracy}')
```

#### Scikit-learn example

```python
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Test accuracy: {accuracy}')
```

### Model optimization

Depending on the chosen framework, optimization techniques were applied to prepare the model for deployment. This could include reducing model complexity, pruning, or quantization.

#### TensorFlow example

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

#### PyTorch example

```python
# Apply dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

#### Scikit-learn example

```python
# Simplify model by limiting its maximum depth
pruned_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, max_features='sqrt') 

pruned_model.fit(X_train, y_train) 
pruned_predictions = pruned_model.predict(X_test) 
pruned_accuracy = accuracy_score(y_test, pruned_predictions) 
print(f'Pruned Test accuracy: {pruned_accuracy}')
```

---

## 6. Preparing the model for deployment

### Saving the model

The trained model was saved in a format suitable for deployment.

#### TensorFlow example

```python
model.save('churn_model.h5')
```

#### PyTorch example

```python
torch.save(model.state_dict(), 'churn_model.pth')
```

#### Scikit-learn example

```python
import joblib

joblib.dump(model, 'churn_model.pkl')
```

### Documentation

Documentation was provided detailing the model’s architecture, how to load it, and steps for deployment. This documentation ensures that the model can be easily integrated into the business environment and maintained over time.

---

## Conclusion

By following these steps, you should have successfully developed, trained, and prepared an ML model for business deployment. This exercise was designed to simulate real-world business requirements, emphasizing the importance of not only developing accurate models but also ensuring they are efficient, scalable, and ready for integration into production systems. If you encounter challenges, review this walkthrough to identify areas for improvement and refine your approach.

This walkthrough provides a comprehensive solution to the activity, offering detailed explanations and examples to guide you through the process of developing and deploying an ML model in a business context.

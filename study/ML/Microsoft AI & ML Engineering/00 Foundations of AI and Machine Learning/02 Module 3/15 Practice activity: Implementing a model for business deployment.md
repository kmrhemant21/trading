# Practice activity: Implementing a model for business deployment  

#### Introduction  
In this activity, you will put into practice the skills and knowledge you’ve acquired by implementing an ML model designed for deployment in a business environment. This hands-on exercise will guide you through the process of developing, training, and preparing a model for deployment. The goal is to simulate real-world business requirements, where models need to be not only accurate but also efficient, scalable, and ready for integration into production systems.  

By the end of this activity, you will:  
- Develop an ML model suitable for solving a business problem.  
- Train the model using a provided dataset.  
- Optimize the model for deployment, considering factors such as scalability, efficiency, and integration.  
- Prepare the model for deployment in a production environment.  

---

#### Business scenario  
**Project name:** Customer Churn Prediction  

**Business overview:**  
You have been hired by a telecommunications company to develop a machine-learning model that predicts customer churn. The company wants to identify customers who are likely to cancel their service so they can take proactive steps to retain them. The model you develop will be integrated into the company’s customer relationship management (CRM) system and used by the marketing team to target at-risk customers with retention offers.  

---

#### Project requirements  
- **Predictive accuracy:** The model must accurately predict whether a customer is likely to churn based on historical data.  
- **Scalability:** The model should be able to handle a large volume of data, as the company has millions of customers.  
- **Integration:** The model needs to be easily integrated into the company’s existing CRM system, which is built on a Python-based backend.  
- **Efficiency:** The model should be optimized for real-time or near-real-time predictions to allow timely interventions by the marketing team.  

---

#### Step-by-step instructions  

##### Step 1: Set up your environment  
**Choose your framework**  
Decide whether you’ll be using TensorFlow, PyTorch, or Scikit-learn for this activity. Each framework has strengths that could be advantageous depending on how you prioritize the project requirements.  
- TensorFlow is great for scalability and production environments.  
- PyTorch is excellent for experimentation and flexibility.  
- Scikit-learn is ideal for straightforward implementation of classical ML models.  

**Set up your development environment**  
Open your preferred coding environment (e.g., Jupyter Notebook, VSCode, or any Python IDE).  

---

##### Step 2: Import libraries and load the dataset  
**Import the required libraries**  
Import the necessary libraries for your chosen framework, as well as any additional libraries for data manipulation and visualization.  

**TensorFlow example**  
```python
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
```

**PyTorch example**  
```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
```

**Scikit-learn example**  
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

**Load the customer churn dataset**  
Using the provided link to the dataset, download the file and import it into your notebook. Make sure it is named correctly and is the correct file type. This dataset file is designed to be small and straightforward, making it simple to use for understanding these processes. We will explore this topic in greater depth later in the “Advanced AI and Machine Learning Techniques and Capstone” course.  

```python
data = pd.read_csv('customer_churn.csv')
```

**Explore the dataset**  
```python
print(data.head())
print(data.info())
```

---

##### Step 3: Preprocess the data  
**Handle missing values and simplify the dataset**  
Identify and handle any missing values in the dataset. This could involve filling missing values, dropping rows or columns, or using imputation techniques. Removing unnecessary columns, such as `CustomerID`, helps keep the dataset clean.  

```python
data = data.drop(columns=['CustomerID']) # Simplify the dataset
data = data.dropna()  # Simple example of dropping missing values
```

**Encode categorical variables**  
Convert categorical variables into numerical format using techniques like one-hot encoding.  

```python
data = pd.get_dummies(data, drop_first=True)
```

**Split the data**  
Split the data into training and testing sets to evaluate the model’s performance.  

```python
X = data.drop('Churn', axis=1)
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

##### Step 4: Develop and train the model  
**Choose and build the model**  
Depending on your chosen framework, build a model suited to the task of classification.  

**TensorFlow example**  
```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

**PyTorch example**  
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

**Scikit-learn example**  
```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
```

**Compile and train the model**  
Train the model using the training data. Monitor the training process to ensure the model is learning appropriately.  

**TensorFlow example**  
```python
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
```

**PyTorch example**  
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

**Scikit-learn example**  
```python
model.fit(X_train, y_train)
```

---

##### Step 5: Evaluate and optimize the model  
**Evaluate the model’s performance**  
After training, evaluate the model using the test data. Consider metrics such as accuracy, precision, recall, and F1 score to assess the model’s performance.  

**TensorFlow example**  
```python
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')
```

**PyTorch example**  
```python
model.eval()
outputs = model(torch.tensor(X_test.values).float())
predictions = (outputs.squeeze().detach().numpy() > 0.5).astype(int)
accuracy = np.mean(predictions == y_test.values)
print(f'Test accuracy: {accuracy}')
```

**Scikit-learn example**  
```python
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Test accuracy: {accuracy}')
```

**Optimize for deployment**  
Optimize the model for deployment by considering techniques such as model pruning, quantization, or simplifying the model architecture without sacrificing too much accuracy.  

**TensorFlow example**  
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

**PyTorch example**  
```python
# Apply dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

**Scikit-learn example**  
```python
# Simplify model by limiting its maximum depth
pruned_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, max_features='sqrt') 

pruned_model.fit(X_train, y_train) 
pruned_predictions = pruned_model.predict(X_test) 
pruned_accuracy = accuracy_score(y_test, pruned_predictions) 
print(f'Pruned Test accuracy: {pruned_accuracy}')
```

---

##### Step 6: Prepare the model for deployment  
**Save the model**  
Save the trained model so it can be deployed in the production environment.  

**TensorFlow example**  
```python
model.save('churn_model.h5')
```

**PyTorch example**  
```python
torch.save(model.state_dict(), 'churn_model.pth')
```

**Scikit-learn example**  
```python
import joblib
joblib.dump(model, 'churn_model.pkl')
```

**Document the model and deployment instructions**  
Provide documentation that includes details about the model, how to load it, and any necessary steps for deploying it into the business environment. This documentation is crucial for ensuring smooth integration and maintenance.  

---

#### Deliverable  
By the end of this activity, you should have a completed Jupyter Notebook or Python script, including all steps from data preprocessing to model deployment preparation.  

**Optional:** Write a brief reflection on the challenges you faced and how you overcame them.  

---

#### Conclusion  
This activity provides you with practical experience in developing an ML model with a focus on real-world business deployment. By following these instructions, you’ll gain valuable insights into how to create models that are not only accurate but also ready for production use in a business context.  

# Practice activity: Creating a decision-making algorithm in Python

## Introduction
In this activity, you will create a decision-making algorithm using Python. Specifically, you will implement a decision tree model to classify data, make predictions, and evaluate the model's performance. This will help you understand how decision-making algorithms work and how they can be applied to solve classification problems.

By the end of this activity, you will be able to:

- Implement a decision tree algorithm for making decisions based on input data.
- Train the algorithm on a dataset and test its performance on unseen data.
- Evaluate the model using accuracy as the performance metric.
- Visualize the decision tree to better understand how decisions are made (optional).

## Step-by-step process for creating a decision-making algorithm
This reading will guide you through the following steps:

1. Step 1: Set up the environment
2. Step 2: Load and explore the dataset
3. Step 3: Preprocess the data
4. Step 4: Implementing the decision tree algorithm
5. Step 5: Evaluate the model
6. Step 6: Visualizing the decision tree (optional) 

### Step 1: Set up the environment
**Instructions**
Start by setting up your Python environment. You will need Scikit-Learn for building the decision tree and pandas for data manipulation.

Install the necessary libraries using the following commands:

```python
!pip install scikit-learn
!pip install pandas
```

**Explanation**
These libraries will allow you to handle the dataset and implement the decision-making algorithm efficiently.

### Step 2: Load and explore the dataset
Load the dataset into a pandas DataFrame and explore its structure. Understanding the dataset is crucial before implementing the algorithm.

**Instructions**
1. Load the dataset.
2. Explore the first few rows and check for any missing values.

**Code example**
```python
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load the Breast Cancer dataset and convert it into a DataFrame
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Explore the dataset
print(df.head())
print(df.info())
```

**Explanation**
Exploring the dataset helps you to understand the features, the data types, and whether any preprocessing is required.

### Step 3: Preprocess the data
Clean the data by handling missing values (if any) and splitting it into training and testing sets. This will allow you to train the model on one part of the data and test its performance on the other.

**Instructions**
1. Handle missing values using imputation or by removing rows with missing data.
2. Split the dataset into features (X) and labels (y).
3. Use train_test_split to divide the data into training and testing sets.

**Code example**
```python
from sklearn.model_selection import train_test_split

# Handle missing values (example: filling missing values with the median)
df.fillna(df.median(), inplace=True)

# Split the data into features and labels
X = df.drop('label_column', axis=1)
y = df['label_column']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Explanation**
Splitting the data into training and testing sets ensures that the model is evaluated on data it hasn't seen during training, giving you a better understanding of how well it generalizes.

### Step 4: Implementing the decision tree algorithm
Implement a decision tree algorithm using DecisionTreeClassifier from Scikit-Learn. Train the model on the training data, and use it to make predictions on the test data.

**Instructions**
1. Import DecisionTreeClassifier.
2. Train the decision tree model on the training data.
3. Use the trained model to make predictions on the test set.

**Code example**
```python
from sklearn.tree import DecisionTreeClassifier

# Train the decision tree model
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

# Make predictions on the test data
y_pred = tree.predict(X_test)
```

**Explanation**
The decision tree algorithm splits the data into branches based on the most informative features, with each split representing a decision point and the leaves representing the final decisions.

### Step 5: Evaluate the model
Evaluate the model using accuracy as the performance metric. This will give you an idea of how often the model makes the correct decision.

**Instructions**
1. Import accuracy_score from Scikit-Learn.
2. Calculate the accuracy by comparing the predictions to the actual labels in the test set.

**Code example**
```python
from sklearn.metrics import accuracy_score

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy * 100:.2f}%")
```

**Explanation**
Accuracy measures how often the model predicts the correct label. It's a simple and effective metric for evaluating classification models.

### Step 6: Visualizing the decision tree (optional) 
Visualize the decision tree to understand how decisions are made at each step. This is useful for interpreting the model's decision-making process.

**Instructions**
1. Import plot_tree from Scikit-Learn.
2. Plot the decision tree to visualize its structure.

**Code example**
```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Plot the decision tree
plt.figure(figsize=(12,8))
plot_tree(tree, filled=True, feature_names=X.columns, class_names=['Class 1', 'Class 2'])
plt.show()
```

**Explanation**
Visualizing the decision tree helps you understand the logic behind each decision. You can see which features the model considers to be the most important and how it splits the data at each node.

## Conclusion
In this activity, you implemented a decision-making algorithm using a decision tree. You trained the model on a dataset, made predictions on unseen data, and evaluated the model's performance using accuracy. By following these steps, you gained hands-on experience in building and evaluating decision-making algorithms.

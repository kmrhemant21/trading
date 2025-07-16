# Practice activity: Implementing and comparing models

## Introduction

In this activity, you will implement two popular machine learning models—logistic regression and decision trees—using the Scikit-learn library in Python. You will compare the performance of these models on a classification problem, evaluate them using key metrics, and reflect on the advantages and disadvantages of each approach. The goal is to understand when to use each model based on the dataset and problem complexity.

By the end of this walkthrough, you will be able to:

- Set up a decision tree model using Scikit-learn, prepare data, and select features for training.
- Evaluate the performance of the decision tree using accuracy, confusion matrices, and classification reports.
- Visualize the decision tree's structure for interpretation and apply tuning methods to improve model performance and prevent overfitting.

---

## 1. Setting up your environment

Before starting, ensure you have the necessary libraries installed. If not, use the following command to install the required packages:

```bash
pip install numpy pandas scikit-learn matplotlib
```

---

## 2. Importing the required libraries

Start by importing the necessary libraries:

```python
# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn import tree
```

---

## 3. Loading and preparing the data

We will use a sample dataset where we aim to classify whether students pass or fail based on the number of study hours and their previous exam scores. You can use this dataset or apply the same procedure to another binary classification problem.

```python
# Sample dataset: Study hours, previous exam scores, and pass/fail labels
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'PrevExamScore': [30, 40, 45, 50, 60, 65, 70, 75, 80, 85],
    'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the data
print(df.head())
```

**Output**

| StudyHours | PrevExamScore | Pass |
|------------|---------------|------|
| 1          | 30            | 0    |
| 2          | 40            | 0    |
| 3          | 45            | 0    |
| 4          | 50            | 0    |
| 5          | 60            | 0    |

This dataset contains two features: `StudyHours` and `PrevExamScore`, and the target variable `Pass` (0 = fail, 1 = pass).

---

## 4. Splitting the data into training and testing sets

We will split the data into training and testing sets to train the models on one portion of the data and evaluate them on the other portion.

```python
# Features (X) and Target (y)
X = df[['StudyHours', 'PrevExamScore']]  # Features
y = df['Pass']  # Target variable (0 = Fail, 1 = Pass)

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data: {X_train.shape}, {y_train.shape}")
print(f"Testing data: {X_test.shape}, {y_test.shape}")
```

This will split the data into 80% for training and 20% for testing, allowing us to evaluate the models on unseen data.

---

## 5. Implementing logistic regression

We will first implement logistic regression, a simple classification model:

```python
# Initialize the Logistic Regression model
logreg_model = LogisticRegression()

# Train the Logistic Regression model
logreg_model.fit(X_train, y_train)

# Make predictions using the test set
y_pred_logreg = logreg_model.predict(X_test)

# Evaluate the Logistic Regression model
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print(f"Logistic Regression Accuracy: {accuracy_logreg}")
```

This step involves training the logistic regression model, making predictions, and evaluating its accuracy on the test set.

---

## 6. Implementing decision trees

Next, we will implement decision trees, a more flexible model that can capture complex relationships in the data:

```python
# Initialize the Decision Tree Classifier
tree_model = DecisionTreeClassifier(random_state=42)

# Train the Decision Tree model
tree_model.fit(X_train, y_train)

# Make predictions using the test set
y_pred_tree = tree_model.predict(X_test)

# Evaluate the Decision Tree model
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print(f"Decision Tree Accuracy: {accuracy_tree}")
```

Here, we follow the same steps for the decision tree model: training, making predictions, and evaluating its performance on the test set.

---

## 7. Comparing model performance

After implementing both models, it’s important to compare their performance using metrics such as accuracy, a confusion matrix, and a classification report.

```python
# Evaluate Logistic Regression
print("Logistic Regression:")
print(f"Accuracy: {accuracy_logreg}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_logreg))
print("Classification Report:")
print(classification_report(y_test, y_pred_logreg))

# Evaluate Decision Tree
print("Decision Tree:")
print(f"Accuracy: {accuracy_tree}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tree))
print("Classification Report:")
print(classification_report(y_test, y_pred_tree))
```

By comparing the accuracy and other evaluation metrics (precision, recall, and F1 score) for both models, you’ll see which model performs better and why.

---

## 8. Visualizing the decision tree

One advantage of decision trees is their interpretability. Let’s visualize the tree structure:

```python
# Visualize the decision tree
plt.figure(figsize=(12,8))
tree.plot_tree(tree_model, feature_names=['StudyHours', 'PrevExamScore'], class_names=['Fail', 'Pass'], filled=True)
plt.title('Decision Tree for Classifying Pass/Fail')
plt.show()
```

This visualization allows us to see how the decision tree splits the data based on the number of study hours and the previous exam scores, helping us understand how it arrives at its classification decisions.

---

## 9. Reflection and analysis

After running both models, compare the results.

- Logistic regression is simpler and works well when the relationship between the features and the target is linear.
- Decision trees offer more flexibility and can capture nonlinear relationships, but they can overfit if not properly tuned (e.g., by limiting the depth of the tree).

Based on your findings, reflect on the following questions:

- Which model performed better on the test set? Why?
- How does the complexity of the data impact the performance of each model?
- What are the advantages and disadvantages of using logistic regression versus decision trees?

---

## Conclusion

You implemented two machine learning models—logistic regression and decision trees—and compared their performance on a classification task. By completing this activity, you should have a better understanding of:

- How to implement logistic regression and decision tree models using Scikit-learn.
- How to evaluate model performance using accuracy, confusion matrices, and classification reports.
- When to use logistic regression versus decision trees based on the complexity of the data.

# Practice activity: Implementing classification models

## Introduction
In this activity, you will implement various classification models using Python. The goal is to build and evaluate such models as logistic regression, decision trees, and support vector machines (SVMs) to classify data. You will work with a dataset, preprocess the data, and train these models to see how they perform in a real-world classification task.

By the end of this activity, you will be able to:

- Describe how to preprocess data for classification tasks.
- Implement and train multiple classification models using Python.
- Evaluate and compare the performance of each model.

## Step-by-step process to implement classification models
Create a new Jupyter notebook. Make sure you have the appropriate Python 3.8 Azure ML kernel selected.

The remaining of this reading will guide you through the following steps:

1. Step 1: Set up the environment
2. Step 2: Load and explore the dataset
3. Step 3: Preprocess the data
4. Step 4: Implement a logistic regression model
5. Step 5: Implement a decision tree model

## Step 1: Set up the environment
### Instructions
First, ensure you have the necessary libraries installed. We'll be using Scikit-Learn for machine learning models, pandas for data manipulation, and matplotlib or seaborn for visualization.

Install the required libraries using the following commands:

```python
pip install scikit-learn
pip install pandas
pip install matplotlib seaborn
```

### Explanation
These libraries will provide the tools to load, manipulate, and visualize the dataset, as well as implement and evaluate classification models.

## Step 2: Load and explore the dataset
### Instructions
Download the dataset you'll be using for this lab. The dataset should contain both features (inputs) and labels (outputs) for the classification task.

Load the dataset into a pandas DataFrame and explore its structure (e.g., checking for missing values, understanding the feature types).

### Steps
1. Import load_breast_cancer from Scikit-Learn.
2. Using pandas dataframe, explore the head of the data.

Understanding the dataset helps us determine which features need to be pre-processed. We'll clean the data, handle missing values, and encode any categorical variables before training the models.

### Code example
```python
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load Breast Cancer dataset and convert to DataFrame
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Explore the dataset
print(df.head())
print(df.info())
```

### Explanation
Understanding the structure of your dataset is crucial for selecting the right preprocessing steps and models.

## Step 3: Preprocess the data
A user might choose not to split the data into training, evaluation, and test sets when they are working with pretrained models specifically designed for such tasks as classification or regression, in which a programmer has already trained and fine-tuned the model on large, robust datasets. In such cases, the model is already generalized well enough that additional model training is unnecessary, and the user's main task is simply to evaluate the model's performance on their own data. Here, splitting the data into just training and testing sets (rather than separate training, evaluation, and test sets) is often sufficient because of the following:

- Focus on the model application, not training
  
  Since the model is already pretrained, the user is not performing additional training. Instead, they're mainly applying the model to the data to assess and potentially fine-tune its suitability.

- Limited data availability
  
  If the dataset is relatively small, further splitting it into three parts (train, eval, test) may result in insufficient data for each split. Using only a training and testing split maximizes the data available for assessing real-world performance.

- Simplified preprocessing for inference-only scenarios
  
  In cases in which the user needs to verify or apply only the model's predictions to new data (e.g., categorizing new customer reviews or sales data), a simpler split suffices to check that the model functions as expected in practice without full retraining.

### Instructions
Understand the data requirements: assess the dataset to determine whether splitting into training, evaluation, and test sets is necessary. For pretrained models that do not require additional training, a simpler split into training and testing sets might suffice.

Preprocessing the data:

- Handle any missing data by either imputing values (e.g., using mean or median imputation) or removing rows/columns with excessive missing values.
- Encode categorical variables if applicable, using methods such as pd.get_dummies for one-hot encoding or LabelEncoder for ordinal encoding.

Split the data:

- Divide the dataset into training and testing sets using an 80–20 split. This step ensures there is enough data for training the model and sufficient separate data for evaluating its performance.

Code implementation:

- Use the train_test_split function from the scikit-learn library to perform the data split. Set a random seed for reproducibility.

Run and verify:

- Execute the preprocessing and splitting code.
- Verify the split by checking the shapes of the training and test sets to ensure the code has divided the data as intended.

### Steps
1. Handle missing data (e.g., using mean/median imputation or removing rows with missing values)
2. Preprocess the data by handling missing values, encoding categorical variables (if any), and splitting the data into training (80 percent) and test sets (20 percent) values.
3. Encode categorical variables using pd.get_dummies or LabelEncoder if necessary.
4. Split the data into training and testing sets using train_test_split from Scikit-Learn.

### Code example
```python
from sklearn.model_selection import train_test_split

# Handle missing data (example: filling missing values with the median)
df.fillna(df.median(), inplace=True)

# Split the data into features and labels
X = df.drop('label_column', axis=1)
y = df['label_column']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Explanation
Preprocessing ensures that your data is clean and ready for ML models to use. Splitting the dataset into training and test sets allows us to evaluate the model's performance on unseen data.

## Step 4: Implement a logistic regression model
### Instructions
Train a logistic regression model on the training data, and evaluate its performance on the test data.

### Steps
1. Import LogisticRegression from Scikit-Learn.
2. Train the model using fit().
3. Predict the labels for the test data, and calculate accuracy.

### Code example
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Train logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Make predictions
y_pred = log_reg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy * 100:.2f}%")
```

### Explanation
Logistic regression is a simple yet effective model for binary classification tasks. Accuracy is one of the metrics used to evaluate how well the model is performing.

## Step 5: Implement a decision tree model
Decision trees split the data based on feature values and make decisions at each node.

### Instructions
Train a decision tree model, and evaluate its performance on the test set.

### Steps
1. Import DecisionTreeClassifier from Scikit-Learn.
2. Train the model on the training data.
3. Make predictions and evaluate the accuracy.

### Code example
```python
from sklearn.tree import DecisionTreeClassifier

# Train decision tree model
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

# Make predictions
y_pred_tree = tree.predict(X_test)

# Evaluate the model
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print(f"Decision Tree Accuracy: {accuracy_tree * 100:.2f}%")
```

### Explanation
Decision trees are highly interpretable models that make decisions by splitting the data based on the most informative features. However, they can be prone to overfitting if not tuned properly.

## Step 6: Implement a support vector machine model
An SVM model is great for high-dimensional spaces. SVMs find a hyperplane that separates the data points into different classes with maximum margin.

### Instructions
Train a support vector machine (SVM) model, and evaluate its performance on the test set.

### Steps
1. Import support vector classifier (SVC) from Scikit-Learn.
2. Train the model on the training data.
3. Make predictions and evaluate the accuracy.

### Code example
```python
from sklearn.svm import SVC

# Train SVM model
svm = SVC()
svm.fit(X_train, y_train)

# Make predictions
y_pred_svm = svm.predict(X_test)

# Evaluate the model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm * 100:.2f}%")
```

### Explanation
SVMs are powerful models, particularly in high-dimensional spaces. They work by finding a hyperplane that separates data points into different classes with the maximum margin.

## Step 7: Evaluate and compare model performance
### Instructions
Compare the performance of the different models using accuracy, precision, recall, and the F1 score.

### Steps
1. Import additional evaluation metrics, including precision_score, recall_score, and f1_score.
2. Calculate these metrics for each model, and print the results for comparison.

### Code example
```python
from sklearn.metrics import precision_score, recall_score, f1_score

# Evaluate performance
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Logistic Regression - Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
```

### Explanation
Accuracy is not always the best metric for evaluating classification models, especially with imbalanced datasets. Precision, recall, and the F1 score provide a more complete picture of model performance.

## Conclusion
In this activity, you successfully implemented several classification models using Python, including logistic regression, decision trees, and SVMs. By training and evaluating these models on a dataset, you gained experience in using common metrics to compare their performance. Understanding how different models work and how to evaluate them is crucial for building reliable machine learning systems.

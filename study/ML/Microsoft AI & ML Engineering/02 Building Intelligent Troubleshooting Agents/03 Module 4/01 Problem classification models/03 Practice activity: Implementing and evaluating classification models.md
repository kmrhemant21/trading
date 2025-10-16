# Practice activity: Implementing and evaluating classification models

## Introduction
In this activity, you will implement and evaluate multiple classification models to classify data based on various features. The goal is to understand how different models work, train them, and compare their performance using evaluation metrics such as accuracy, precision, recall, and F1 score.

By the end of this activity, you will be able to:

- Implement three different classification models: logistic regression, decision tree, and support vector machines (SVM).
- Train and test these models on a dataset.
- Evaluate and compare the models using accuracy and other performance metrics, including precision, recall, and F1 score.

## Step-by-step process to implement and evaluate classification models
This reading will guide you through the following steps:

1. Step 1: Set up the environment
2. Step 2: Load and explore the dataset
3. Step 3: Preprocess the data
4. Step 4: Implement classification models
5. Step 5: Evaluate model performance
6. Step 6: Compare model performance

## Step 1: Set up the environment
### Instructions
Start by setting up your Python environment. You will need Scikit-Learn for ML models, pandas for data manipulation, and matplotlib or seaborn for visualization.

Install the necessary libraries using the following commands:

```python
pip install scikit-learn
pip install pandas
pip install matplotlib seaborn
```

### Explanation
These libraries will allow you to manipulate your dataset, build ML models, and visualize your results.

## Step 2: Load and explore the dataset
### Instructions
Load the dataset into a pandas dataframe, and explore its structure.

Check for missing values, understand the distribution of the features, and get an overall sense of the dataset.

### Code example
```python
import pandas as pd

# Load the dataset
df = pd.read_csv('your-dataset.csv')

# Explore the dataset
print(df.head())
print(df.info())
```

### Explanation
It's important to explore the dataset before proceeding with the model implementation to understand the types of features and how clean the data is.  Preprocessing data is a critical step in any data analysis or machine learning pipeline, as it ensures that data is clean, consistent, and suitable for modeling. Raw data often contains missing values, outliers, or inconsistencies that can negatively impact the accuracy and reliability of predictive models. We create a more robust dataset that models can better interpret by using preprocessing techniques like handling missing values, normalizing or scaling features, encoding categorical variables, and removing noise. Effective preprocessing improves model performance and helps prevent overfitting and underfitting, leading to more accurate and meaningful insights from data.

## Step 3: Preprocess the data
Clean the dataset by handling missing values, encoding categorical variables if necessary, and splitting the data into training and testing sets.

### Instructions
Handle missing data using techniques such as mean/median imputation or by removing rows with missing values.

Encode any categorical variables using pd.get_dummies or LabelEncoder.

Split the data into training and testing sets using train_test_split from Scikit-Learn.

### Code example
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

### Explanation
Preprocessing ensures the dataset is clean and properly formatted before training the models. Splitting the data into training and test sets helps in evaluating the model's performance on unseen data.

## Step 4: Implement classification models
### Logistic regression
#### Instructions
Import LogisticRegression from Scikit-Learn.

Train the model using fit() on the training data.

Predict the labels for the test data and calculate accuracy.

#### Code example
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

#### Explanation
Logistic regression is an effective model for binary classification tasks and is often used as a baseline model in ML.

### Decision tree
#### Instructions
Import DecisionTreeClassifier from Scikit-Learn.

Train the decision tree on the training data.

Make predictions and evaluate the accuracy.

#### Code example
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

#### Explanation
Decision trees are easy to interpret and provide high accuracy, but they can overfit the training data if not pruned properly.

### Support vector machines
#### Instructions
Import SVC from Scikit-Learn.

Train the SVM model on the training data.

Make predictions and evaluate the accuracy.

#### Code example
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

#### Explanation
SVM is a powerful classification model, especially when the data is not linearly separable. It works well in high-dimensional spaces.

## Step 5: Evaluate model performance
In addition to accuracy, evaluate the performance of each model using precision, recall, and F1 score to get a more comprehensive view of how well the models perform.

### Instructions
Import precision_score, recall_score, and f1_score from Scikit-Learn.

Calculate these metrics for each model.

### Code example
```python
from sklearn.metrics import precision_score, recall_score, f1_score

# Evaluate logistic regression model
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Logistic Regression - Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
```

### Explanation
Accuracy alone doesn't always tell the full story, especially with imbalanced datasets. Precision, recall, and F1 score offer additional insights into how the model handles false positives and false negatives.

## Step 6: Compare model performance
### Instructions
Compare the performance of the three models using the accuracy, precision, recall, and F1 score you calculated.

### Discussion points
- Which model performed best on the test set?
- Were there any significant differences in precision, recall, or F1 score among the models?
- How does the choice of model affect the overall results?

### Explanation
By comparing these models, you'll get a sense of which algorithms work best for this specific dataset. In real-world applications, it's common to experiment with multiple models and tune them to optimize performance.

## Conclusion
In this activity, you implemented three classification models: logistic regression, decision tree, and SVM. You also evaluated their performance using accuracy, precision, recall, and F1 score and compared their strengths and weaknesses. This activity gave you practical experience in building and evaluating ML models for classification tasks.
# Practice activity: Solution recommendation

## Introduction
In this activity, you will build a solution recommendation system using machine learning techniques. The goal is to implement an algorithm that takes a problem as input and recommends the most appropriate solution based on past data. You will learn how to preprocess data, train a recommendation model, and evaluate its effectiveness.

By the end of this activity, you will:

- Implement a machine learning model for recommending solutions based on input problems.
- Preprocess a dataset and split it into training and test sets.
- Train a recommendation system using past data and evaluate its performance.
- Improve recommendation accuracy by tuning the model's parameters (optional).

## Step-by-step process to create a solution recommendation system
This reading will guide you through the following steps:

1. Step 1: Set up the environment
2. Step 2: Load and explore the dataset
3. Step 3: Preprocess the data
4. Step 4: Implement the recommendation model
5. Step 5: Evaluate the model
6. Step 6: Improve the model (optional)

### Step 1: Set up the environment
**Instructions**  
Start by setting up your Python environment. You will need Scikit-Learn for implementing the recommendation algorithm and pandas for data manipulation.

Install the necessary libraries using the following commands:

```python
pip install Scikit-Learn
pip install pandas
```

**Explanation**  
These libraries will give you the tools to manipulate the dataset and build the recommendation model efficiently.

### Step 2: Load and explore the dataset
Load the dataset into a pandas DataFrame and explore its structure. Understanding the data will help to guide the recommendation system you'll build.

**Instructions**  
- Load the dataset.
- Explore the dataset by checking the first few rows and identifying missing values, if any.

**Code example**
```python
import pandas as pd

# Load the dataset
df = pd.read_csv('your-dataset.csv')

# Explore the dataset
print(df.head())
print(df.info())
```

**Explanation**  
Exploring the dataset is crucial to identify the problem and solution features, as well as any missing values that need to be handled.

### Step 3: Preprocess the data
Clean the data by handling missing values and splitting it into training and test sets. This ensures that the model is trained on one portion of the data and tested on unseen data to evaluate its performance.

**Instructions**  
- Handle missing values using techniques such as mean/median imputation or by removing rows with missing data.
- Split the dataset into features (X) and labels (y).
- Use train_test_split to split the data into training and test sets.

**Code example**
```python
from sklearn.model_selection import train_test_split

# Handle missing values (example: filling missing values with the median)
df.fillna(df.median(), inplace=True)

# Split the data into features (problem descriptions) and labels (solutions)
X = df.drop('solution_column', axis=1)
y = df['solution_column']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Explanation**  
Splitting the data into training and testing sets ensures that the model is evaluated on data it hasn't seen before, which helps to gauge its real-world effectiveness.

### Step 4: Implement the recommendation model
Implement a solution recommendation model using a machine learning algorithm such as KNeighborsClassifier. Train the model on the training data and use it to recommend solutions based on new input problems.

**Instructions**  
- Import the KNeighborsClassifier from Scikit-Learn.
- Train the model on the training data.
- Use the trained model to make predictions on the test set.

**Code example**
```python
from sklearn.neighbors import KNeighborsClassifier

# Train the KNN recommendation model
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn.predict(X_test)
```

**Explanation**  
The K-Nearest Neighbors (KNN) algorithm works by finding the closest examples in the dataset and using them to recommend a solution for new input problems.

### Step 5: Evaluate the model
Evaluate the recommendation model using accuracy as the main performance metric. You can also explore other metrics, such as precision, recall, or F1 score.

**Instructions**  
- Import accuracy_score from Scikit-Learn.
- Calculate the accuracy of the recommendation model by comparing its predictions to the actual solutions in the test set.

**Code example**
```python
from sklearn.metrics import accuracy_score

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Recommendation Model Accuracy: {accuracy * 100:.2f}%")
```

**Explanation**  
Accuracy measures how often the recommendation model predicts the correct solution. Other metrics such as precision and recall can provide additional insights, especially in imbalanced datasets.

### Step 6: Improve the model (optional) 
Try to improve the recommendation model by tuning its hyperparameters. For example, in KNN, you can adjust the number of neighbors (n_neighbors) to see if it improves accuracy. To change from the default parameters, locate the specific hyperparameters in the model's function call (e.g., KNeighborsClassifier(n_neighbors=5)) and replace the default values with your chosen ones. Experiment with different values by assigning a range, such as n_neighbors=3, n_neighbors=7, etc., and compare their impact on model performance. Use tools like grid search or random search for systematic tuning, allowing you to identify the optimal hyperparameters for the best results.

**Instructions**  
- Experiment with different values for the model's hyperparameters, such as the number of neighbors in KNN.
- Retrain the model and evaluate its performance.

**Code example**
```python
# Tuning the number of neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Evaluate the tuned model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Improved Model Accuracy: {accuracy * 100:.2f}%")
```

**Explanation**  
Tuning the model's hyperparameters can improve its performance by finding the best settings for the specific dataset. Experimenting with these parameters is a common practice to boost the model's effectiveness.

## Conclusion
In this activity, you successfully created a solution recommendation system using a machine learning model. You learned how to preprocess the data, train a model, and evaluate its performance. Additionally, you explored ways to improve the model by tuning its hyperparameters.

# Walkthrough: Applying metrics and cross-validation (Optional)

## Introduction

This walkthrough will review the solution to the activity where you applied evaluation metrics and cross-validation to assess the performance of a machine learning model. 

Cross-validation ensures that your model's performance is not dependent on a single train-test split, providing a more reliable measure of its generalization. This guide will explain each step and the reasoning behind it, ensuring you have correctly implemented and understood the activity.

By the end of this walkthrough, you'll be able to:

- Implement cross-validation techniques.
- Calculate and interpret evaluation metrics.
- Analyze model performance.

---

## 1. Loading and preparing the data

You used a dataset with the features `StudyHours` and `PrevExamScore` to predict whether students would pass or fail a presumptive future exam (not shown). The first step was to load the data into a pandas DataFrame:

```python
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'PrevExamScore': [30, 40, 45, 50, 60, 65, 70, 75, 80, 85],
    'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)
```

`StudyHours` and `PrevExamScore` were the features. `Pass` (`0 = Fail`, `1 = Pass`) was the target variable. This data was well suited for a binary classification problem using models like logistic regression.

---

## 2. Splitting the data and training a logistic regression model

You began by splitting the dataset into training and testing sets and applying a logistic regression model to the training set:

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Features and target variable
X = df[['StudyHours', 'PrevExamScore']]
y = df['Pass']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)
```

You used `train_test_split` to randomly assign 80 percent of the data for training and 20 percent for testing. After training the logistic regression model, you made predictions on the testing set.

---

## 3. Applying evaluation metrics

Next, you calculated the model’s accuracy, precision, recall, and F1 score to evaluate its performance:

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')
```

Here’s what these metrics show:

- **Accuracy**: the proportion of correct predictions out of all predictions made. In this case, the model's accuracy is the percentage of students it correctly classified as pass or fail.
- **Precision**: the proportion of positive predictions (`Pass`) that were correct.
- **Recall**: the proportion of actual positive cases (`Pass`) that were correctly identified.
- **F1 score**: the harmonic mean of precision and recall, providing a balanced measure of the two.

---

## 4. Introducing cross-validation

Cross-validation was then introduced to provide a more robust evaluation of the model's performance. You used five-fold cross-validation, which splits the dataset into five parts (folds), trains the model on four folds, and tests it on the remaining fold. This process is repeated for each fold, and the average performance is calculated:

```python
from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation and calculate accuracy for each fold
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

# Display the accuracy for each fold and the mean accuracy
print(f'Cross-validation accuracies: {cv_scores}')
print(f'Mean cross-validation accuracy: {cv_scores.mean()}')
```

The `cross_val_score` function calculates the accuracy for each fold. The mean of these scores gives a more reliable estimate of how well the model generalizes to unseen data.

---

## 5. Cross-validation with multiple metrics

In addition to accuracy, you calculated precision, recall, and F1 score using cross-validation to assess the model’s performance across various metrics:

```python
from sklearn.model_selection import cross_validate

# Define multiple scoring metrics
scoring = ['accuracy', 'precision', 'recall', 'f1']

# Perform cross-validation
cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)

# Print results for each metric
print(f"Cross-validation Accuracy: {np.mean(cv_results['test_accuracy'])}")
print(f"Cross-validation Precision: {np.mean(cv_results['test_precision'])}")
print(f"Cross-validation Recall: {np.mean(cv_results['test_recall'])}")
print(f"Cross-validation F1-Score: {np.mean(cv_results['test_f1'])}")
```

This approach provided a more comprehensive evaluation of the model. Accuracy, precision, recall, and F1 score were calculated for each fold, and the average was reported for each metric.

---

## 6. Cross-validation with a regression model

For regression tasks, you used metrics such as R-squared and mean absolute error (MAE) to evaluate the model's performance. Here’s how you applied cross-validation to a regression model:

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Sample dataset for regression
X_reg = df[['StudyHours']]
y_reg = df['PrevExamScore']

# Initialize a linear regression model
reg_model = LinearRegression()

# Perform 5-fold cross-validation using R-squared as the metric
cv_scores_r2 = cross_val_score(reg_model, X_reg, y_reg, cv=5, scoring='r2')

print(f'Cross-validation R-squared scores: {cv_scores_r2}')
print(f'Mean R-squared score: {np.mean(cv_scores_r2)}')
```

In this case, the R-squared metric measured how much of the variance in the target variable was explained by the features. A higher R-squared value indicates a better fit. You could also calculate mean squared error (MSE) or MAE using the same process.

---

## Conclusion

You successfully applied evaluation metrics and cross-validation to a classification problem using logistic regression. You also explored how cross-validation could be used with regression models to obtain more robust performance estimates.

### Key takeaways:

- Cross-validation provides a more reliable evaluation by testing the model on multiple data splits.
- Accuracy, precision, recall, and F1 score are essential metrics for evaluating classification models.
- For regression models, R-squared, MSE, and MAE are commonly used metrics.
- Using cross-validation with multiple metrics helps ensure that the model’s performance is well-rounded and not dependent on a single evaluation metric.
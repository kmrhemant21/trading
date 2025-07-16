# Walkthrough: Implementing and comparing models (Optional)

## Introduction

In this walkthrough, we will review the steps from the activity where you implemented and compared two machine learning models—logistic regression and decision trees—using the Scikit-learn library. The goal of the activity was to understand the differences between these models and evaluate their performance on a binary classification task. This guide will walk you through each step of the activity, ensuring you have correctly implemented both models and can interpret their results.

By the end of this walkthrough, you will be able to:

- Build and train both logistic regression and decision tree models using Scikit-learn for binary classification tasks.
- Assess the effectiveness of each model using metrics such as accuracy, a confusion matrix, and a classification report to understand their strengths and weaknesses.
- Visualize the structure of a decision tree to interpret its decision-making process and compare its performance against the logistic regression model.

---

## 1. Loading and preparing the data

We used a dataset that included the features `StudyHours` and `PrevExamScore` to predict whether students passed or failed. We first loaded this data into a pandas DataFrame:

```python
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'PrevExamScore': [30, 40, 45, 50, 60, 65, 70, 75, 80, 85],
    'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)
print(df.head())
```

The dataset had the following columns:

- **StudyHours**: the number of hours a student studied
- **PrevExamScore**: the score from a previous exam
- **Pass**: the target variable, where 0 indicates fail and 1 indicates pass

This data is appropriate for a binary classification task because the target variable has two possible outcomes.

---

## 2. Splitting the data into training and testing sets

We split the data into training and testing sets using the `train_test_split` function. This ensures that the models are trained on one portion of the data and evaluated on another:

```python
X = df[['StudyHours', 'PrevExamScore']]  # Features
y = df['Pass']  # Target variable (0 = Fail, 1 = Pass)

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data: {X_train.shape}, {y_train.shape}")
print(f"Testing data: {X_test.shape}, {y_test.shape}")
```

- **Training set (80%)**: Used to train the models
- **Test set (20%)**: Used to evaluate the models’ performance on unseen data

This split helps ensure that the models can generalize to new data and are not overfitted to the training set.

---

## 3. Implementing logistic regression

We first implemented logistic regression, a simple yet powerful model for binary classification:

```python
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)

# Make predictions using the test set
y_pred_logreg = logreg_model.predict(X_test)

# Evaluate the Logistic Regression model
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print(f"Logistic Regression Accuracy: {accuracy_logreg}")
```

In this step, we:

- Initialized the logistic regression model using `LogisticRegression()`.
- Trained the model using the `fit()` method on the training data.
- Used the trained model to make predictions on the test set.
- Evaluated the model's performance by calculating its accuracy.

---

## 4. Implementing decision trees

Next, we implemented a decision tree model, which offers more flexibility and captures complex relationships in the data:

```python
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

# Make predictions using the test set
y_pred_tree = tree_model.predict(X_test)

# Evaluate the Decision Tree model
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print(f"Decision Tree Accuracy: {accuracy_tree}")
```

In this step, we followed a similar process:

- Initialized the decision tree model using `DecisionTreeClassifier()`.
- Trained the model on the training data.
- Made predictions on the test set using the `predict()` method.
- Evaluated the model by calculating its accuracy.

---

## 5. Comparing model performance

After implementing both models, we compared their performance using evaluation metrics such as accuracy, a confusion matrix, and a classification report:

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

The accuracy is the proportion of correct predictions. Additionally, we used the following metrics:

- **Confusion matrix**, which displays the counts of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN), helping us understand the model’s performance beyond simple accuracy.
- **Classification report**, which provides detailed metrics for each class, including:
  - **Precision**: The proportion of predicted positives that were actually correct
  - **Recall**: The proportion of actual positives that were correctly identified
  - **F1 score**: A harmonic mean of precision and recall, giving a balanced measure of both

---

## 6. Visualizing the decision tree

One of the main advantages of decision trees is their interpretability. We visualized the tree structure to see how the model makes decisions:

```python
plt.figure(figsize=(12,8))
tree.plot_tree(tree_model, feature_names=['StudyHours', 'PrevExamScore'], class_names=['Fail', 'Pass'], filled=True)
plt.title('Decision Tree for Classifying Pass/Fail')
plt.show()
```

The decision tree visualization shows how the model splits the data at each node based on the features `StudyHours` and `PrevExamScore` to predict whether a student passes or fails.

---

## 7. Reflecting on model performance

Based on the accuracy and the confusion matrix, we compared the models:

- **Logistic regression**: this model assumes a linear relationship between the features and the target variable. It is simpler and less prone to overfitting but may not capture complex patterns in the data.
- **Decision tree**: this model can handle nonlinear relationships and complex data patterns. However, decision trees are prone to overfitting, especially if they are allowed to grow too deep. In this case, we can use techniques such as pruning, or limiting the depth of the tree to prevent overfitting.

If the decision tree accuracy was significantly higher than the logistic regression accuracy, it suggests that the data has more complex, nonlinear relationships. If the accuracy was similar or lower, logistic regression might be the better choice due to its simplicity and robustness.

---

## Conclusion

We successfully implemented and compared two machine learning models—logistic regression and decision trees. We evaluated both models using accuracy, a confusion matrix, and a classification report. Additionally, we visualized the decision tree to gain insights into its decision-making process.

**Key takeaways:**

- Logistic regression is simple and interpretable, and it works well for linear problems.
- Decision trees offer more flexibility for nonlinear data but are prone to overfitting if not properly controlled.
- Comparing models using a variety of evaluation metrics helps ensure we understand their strengths and weaknesses.

This activity demonstrated the importance of choosing the right model based on the complexity of the data and the specific problem at hand.

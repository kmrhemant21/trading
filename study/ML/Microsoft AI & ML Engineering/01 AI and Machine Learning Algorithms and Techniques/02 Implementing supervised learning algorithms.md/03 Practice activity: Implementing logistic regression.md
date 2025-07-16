# Practice activity: Implementing logistic regression

## Introduction

In this activity, you will learn how to implement a logistic regression model using Python and the popular machine learning library Scikit-learn. Logistic regression is commonly used for classification tasks, where the goal is to categorize data into distinct classes (e.g., spam vs. not spam, pass vs. fail). We’ll use logistic regression to predict binary outcomes and evaluate the model’s performance.

By the end of this activity, you will be able to:

- Set up and train a logistic regression model using Scikit-learn.
- Interpret model outputs and performance metrics such as accuracy and a confusion matrix, also known as an error matrix.
- Visualize the logistic regression curve and predicted probabilities with Matplotlib.

---

## 1. Setting up your environment

Ensure that you have the necessary libraries installed. If you haven’t installed them yet, use the following command to install the required packages:

```bash
pip install numpy pandas scikit-learn matplotlib
```

---

## 2. Importing required libraries

Start by importing the libraries we’ll need for this activity:

```python
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
```

- NumPy and pandas will help us handle numerical and tabular data.
- Scikit-learn's LogisticRegression will be used to build the model.
- Matplotlib will allow us to visualize the results.

---

## 3. Loading and preparing the data

We’ll use a sample dataset to classify whether students pass or fail based on the number of their study hours. You can use this dataset or substitute it with your own.

```python
# Sample dataset: Study hours and whether students passed or failed
data = {
   'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
   'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the data
print(df.head())
```

Here, `StudyHours` is our feature, and `Pass` is the target label, where 0 indicates failure and 1 indicates passing.

---

## 4. Splitting the data into training and testing sets

We will split the dataset into training and testing sets, allowing us to train the model on one portion of the data and evaluate it on another:

```python
# Features (X) and Target (y)
X = df[['StudyHours']]  # Feature(s)
y = df['Pass']          # Target variable (0 = Fail, 1 = Pass)

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shape of the training and testing sets
print(f"Training data: {X_train.shape}, {y_train.shape}")
print(f"Testing data: {X_test.shape}, {y_test.shape}")
```

This will split the dataset into 80% for training and 20% for testing, ensuring the model is evaluated on unseen data.

---

## 5. Training the logistic regression model

Now we’ll initialize and train the logistic regression model using the training data:

```python
# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Display the model's learned coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")
```

- **Intercept:** This is the bias term in the logistic regression equation.
- **Coefficient:** This value indicates how much the log odds of passing change with each additional hour of study.

---

## 6. Making predictions

Once the model is trained, we can use it to predict whether students pass or fail based on the number of their study hours:

```python
# Make predictions on the testing set
y_pred = model.predict(X_test)

# Display the predictions
print("Predicted Outcomes (Pass/Fail):", y_pred)
print("Actual Outcomes:", y_test.values)
```

---

## 7. Evaluating the model

To evaluate how well the logistic regression model performed, we’ll use several metrics, including accuracy, a confusion matrix, and a classification report (which includes precision, recall, and F1 score):

```python
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Generate classification report
class_report = classification_report(y_test, y_pred)

# Display evaluation metrics
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
```

- **Accuracy:** The percentage of correctly predicted outcomes out of all predictions
- **Confusion matrix:** A table that shows the number of correct and incorrect predictions categorized by true positives, true negatives, false positives, and false negatives
- **Classification report:** A report that provides detailed metrics such as precision, recall, and F1 score for each class

### Understanding Model Evaluation Metrics

Here’s a deep dive into these core classification‐evaluation concepts, with definitions, formulas, interpretation tips, and a worked example.

---

### Accuracy

**Definition:** The fraction of all predictions that the model got right.

```math
\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}
              = \frac{TP + TN}{TP + TN + FP + FN}
```

* **TP** = True Positives (correctly predicted positives)
* **TN** = True Negatives (correctly predicted negatives)
* **FP** = False Positives (incorrectly predicted positives)
* **FN** = False Negatives (incorrectly predicted negatives)

**Interpretation:**

* Ranges from 0 to 1 (or 0%–100%).
* Easy to understand, but can be **misleading** on imbalanced datasets (e.g., 95% accuracy if you always predict the majority class).

---

### Confusion Matrix

A 2×2 (binary) or *K*×*K* (multiclass) table that shows how predictions break down against actual labels.

### Binary Confusion Matrix Layout

|                     | **Predicted Positive** | **Predicted Negative** |
| ------------------- | ---------------------- | ---------------------- |
| **Actual Positive** | TP                     | FN                     |
| **Actual Negative** | FP                     | TN                     |

* **TP (True Positive):** Model predicted positive, and it was positive.
* **FN (False Negative):** Model predicted negative, but it was positive.
* **FP (False Positive):** Model predicted positive, but it was negative.
* **TN (True Negative):** Model predicted negative, and it was negative.

**Why it matters:**

* Shows *where* the model is making mistakes (Type I vs. Type II errors).
* Basis for more nuanced metrics (precision, recall, etc.).

---

### Classification Report

Summarizes **precision**, **recall**, and **F1 score** (and often support) for each class.

### Key Metrics

| Metric        | Formula                                                                                   | What it tells you                                           |
| ------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| **Precision** | $\frac{TP}{TP + FP}$                                                                      | Of all predicted positives, how many were correct? (low FP) |
| **Recall**    | $\frac{TP}{TP + FN}$                                                                      | Of all actual positives, how many did we catch? (low FN)    |
| **F1 Score**  | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean of precision & recall; balances both          |

* **Support**: Number of true instances of each class (i.e., TP + FN).

### Example Report (for a 2-class problem)

|      Class      | Precision |  Recall  | F1-Score | Support |
| :-------------: | :-------: | :------: | :------: | :-----: |
|        0        |    0.92   |   0.95   |   0.93   |   200   |
|        1        |    0.89   |   0.84   |   0.86   |    50   |
| **Avg / Total** |  **0.91** | **0.93** | **0.91** |   250   |

* **Macro average:** Unweighted mean of per-class metrics.
* **Weighted average:** Mean of per-class metrics weighted by support.

---

### Putting It All Together: A Simple Walk-through

Suppose you have 300 test samples:

* Model predicts 180 true positives, 10 false negatives, 20 false positives, and 90 true negatives.

1. **Accuracy**

   $$
     \frac{TP + TN}{\text{Total}} = \frac{180 + 90}{300} = 0.90 \quad(90\%)
   $$

2. **Confusion Matrix**

   |              | Pred + | Pred – |
   | ------------ | :----: | :----: |
   | **Actual +** |   180  |   10   |
   | **Actual –** |   20   |   90   |

3. **Precision (for “+” class)**

   $$
     \frac{TP}{TP + FP} = \frac{180}{180 + 20} = 0.90
   $$

4. **Recall (for “+” class)**

   $$
     \frac{TP}{TP + FN} = \frac{180}{180 + 10} \approx 0.95
   $$

5. **F1 Score (for “+” class)**

   $$
     2 \times \frac{0.90 \times 0.95}{0.90 + 0.95} \approx 0.925
   $$

---

## 5. When to Use Which Metric

| Scenario                                   | Preferred Metric(s)            |
| ------------------------------------------ | ------------------------------ |
| **Balanced dataset, cost of errors equal** | Accuracy                       |
| **Imbalanced dataset**                     | Precision, Recall, F1, ROC-AUC |
| **Concerned about false positives**        | High Precision                 |
| **Concerned about false negatives**        | High Recall                    |
| **Need single “balanced” metric**          | F1 Score                       |

---

**Key Takeaways**

* **Accuracy** is simple but can hide class‐imbalance issues.
* **Confusion Matrix** gives full error breakdown.
* **Precision, Recall & F1** let you focus on different error types and strike a balance.

Use all of these together to get a complete picture of your classifier’s performance!


Class imbalance occurs when the classes in your dataset are not represented equally—one (or more) class(es) has far fewer examples than the others. This skew can lead to a model that appears to perform well (high accuracy) but fails to correctly predict the minority class. Here’s a deeper look:

---

## 1. What Is Class Imbalance?

* **Definition:** A dataset is “imbalanced” when the number of instances of one class (the minority class) is much smaller than that of another (the majority class).
* **Example:**

  * Fraud detection: 0.5% of transactions are fraudulent (minority), 99.5% are legitimate (majority).
  * Medical diagnosis: 1% of patients have a rare disease, 99% are healthy.

---

## 2. Why It’s Problematic

1. **Misleading Accuracy**

   * A naive model that always predicts the majority class achieves very high accuracy (e.g., 99.5%) but never detects the minority class.
2. **Poor Minority‐Class Recall**

   * The model “learns” to ignore the minority class because minimizing overall error is easiest by focusing on the majority.
3. **Skewed Decision Boundary**

   * Classifiers (especially those assuming balanced priors) place the decision boundary closer to the minority region, making it hard to correctly classify minority samples.

---

## 3. Impact on Evaluation Metrics

* **Accuracy:** Inflated by the majority class—fails to reflect minority performance.
* **Precision/Recall:**

  * **Precision** for the minority = TP / (TP + FP) may look OK if very few positives are predicted.
  * **Recall** for the minority = TP / (TP + FN) will be very low (many FN).
* **ROC-AUC** can also be misleading under extreme skew; **Precision-Recall AUC** is often more informative.

---

## 4. Strategies to Mitigate Class Imbalance

### A. Data‐Level Methods

1. **Random Oversampling**

   * Duplicate minority-class samples until balanced.
2. **Random Undersampling**

   * Remove majority-class samples to match minority count.
3. **SMOTE (Synthetic Minority Over-sampling Technique)**

   * Generate synthetic minority examples by interpolating between existing ones.
4. **Hybrid Sampling**

   * Combine oversampling the minority with undersampling the majority.

### B. Algorithm-Level Methods

1. **Class Weights / Cost-Sensitive Learning**

   * Assign a higher penalty for misclassifying minority samples.
   * e.g., in scikit-learn:

     ```python
     LogisticRegression(class_weight='balanced')
     ```
2. **Threshold Adjustment**

   * Move the decision threshold (default 0.5) to favor the minority class (e.g., predict positive if p ≥ 0.3).
3. **Ensemble Methods**

   * **BalancedBaggingClassifier**, **EasyEnsemble**, etc., that internally resample or weight.

### C. Evaluation‐Level Methods

* Focus on metrics that highlight minority performance:

  * **Recall** (sensitivity), **Precision**, **F1-score** for the minority.
  * **Precision-Recall curves** rather than ROC curves.

---

## 5. Best Practices

1. **Always Examine Class Distribution**

   * Before modeling, check relative frequencies.
2. **Choose the Right Metric**

   * Don’t rely on accuracy—use recall, precision, F1, or PR-AUC.
3. **Cross-Validate with Stratification**

   * Ensure each fold preserves the same imbalance ratio.
4. **Combine Techniques**

   * Often a mix of sampling + class weighting + threshold tuning yields the best results.

---

### Quick Illustrative Example

Suppose you have 1,000 samples: 950 negatives, 50 positives.

* **Baseline Accuracy if you always predict “negative”:**

  $$
    \frac{950}{1000} = 95\%
  $$

  Yet recall on positives = 0% (you never catch any).

* **If you oversample positives to 950 (so 950 pos, 950 neg):**

  * The model must learn to distinguish both classes, and metrics like recall on positives become meaningful.

---

By recognizing class imbalance early and applying these strategies, you ensure your model truly learns the minority class—and not just the “easy” majority.

---

## 8. Visualizing the results

Logistic regression produces probabilities for each outcome. We can visualize the sigmoid function, which is the key characteristic of logistic regression, and plot the model’s predictions against the actual data points:

```python
# Create a range of study hours for plotting
study_hours_range = np.linspace(X.min(), X.max(), 100)

# Calculate predicted probabilities using the sigmoid function
y_prob = model.predict_proba(study_hours_range.reshape(-1, 1))[:, 1]

# Plot the actual data points
plt.scatter(X_test, y_test, color='blue', label='Actual Data')

# Plot the logistic regression curve
plt.plot(study_hours_range, y_prob, color='red', label='Logistic Regression Curve')

# Add labels and title
plt.xlabel('Study Hours')
plt.ylabel('Probability of Passing')
plt.title('Logistic Regression: Study Hours vs. Pass/Fail')
plt.legend()

# Show the plot
plt.show()
```

This visualization helps you understand the relationship between the number of study hours and the likelihood of passing. The sigmoid curve shows the probability of passing as the number of study hours increases.

---

## Conclusion

In this activity, you successfully implemented a logistic regression model to predict whether students would pass or fail based on the number of their study hours. Key takeaways include:

- Logistic regression is a powerful tool for binary classification problems, where the goal is to predict a categorical outcome (e.g., pass/fail, yes/no).
- Model evaluation metrics such as accuracy, confusion matrices, and classification reports provide insights into the performance of the model.
- Visualization of the sigmoid function gives a clearer picture of how logistic regression estimates probabilities.

By following these steps, you will have the knowledge to apply logistic regression to other classification problems in your own machine learning projects.
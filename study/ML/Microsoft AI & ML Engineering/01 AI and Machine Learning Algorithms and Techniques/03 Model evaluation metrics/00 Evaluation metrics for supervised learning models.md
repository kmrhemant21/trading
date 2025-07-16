# Evaluation metrics for supervised learning models

## Introduction

Evaluation metrics play a critical role in assessing the performance of supervised learning models. These metrics help us understand how well a model predicts outcomes and whether it can generalize to unseen data. Different tasks, such as classification and regression, require different evaluation metrics.

By the end of this reading, you’ll be able to:

- **Identify key evaluation metrics:** understand and describe the most commonly used evaluation metrics for classification and regression models.
- **Apply metrics to model performance:** evaluate the performance of ML models using appropriate metrics such as accuracy, precision, recall, mean squared error (MSE), and R-squared.
- **Choose the right metric for the task:** select the most suitable evaluation metric based on the specific problem and dataset characteristics, ensuring accurate model assessment.

---

## Evaluation metrics for classification models
| Term              | Description                                             |
|-------------------|---------------------------------------------------------|
| True Positive (TP)| The model predicted positive, and it was actually positive. |
| False Positive (FP)| The model predicted positive, but it was actually negative. |
| True Negative (TN)| The model predicted negative, and it was actually negative. |
| False Negative (FN)| The model predicted negative, but it was actually positive. |

Classification models predict discrete outcomes, such as whether an email is spam or not spam or whether a customer will churn or remain. Below are some key evaluation metrics used to assess the performance of classification models:

### Accuracy

Accuracy measures the percentage of correct predictions out of all predictions made.

It is calculated as:

$$
\text{Accuracy} = \frac{\text{True Positives} + \text{True Negatives}}{\text{True Positives} + \text{True Negatives} + \text{False Positives} + \text{False Negatives}}
$$

**Example:**  
If a model correctly predicts 90 out of 100 instances, its accuracy is 90 percent. However, accuracy may not always be the best metric for imbalanced datasets, in which one class is much more frequent than the other.

---

### Precision

Precision measures the percentage of true positive predictions out of all positive predictions that the model makes. It is important in cases in which false positives are costly, such as in medical diagnoses or spam detection.

It is calculated as:

$$
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
$$

**Example:**  
In spam detection, precision is the proportion of emails predicted as spam that are actually spam. A high precision value indicates fewer false positives.

---

### Recall (sensitivity or true positive rate)

Recall measures the percentage of true positive predictions out of all actual positives. It is important when the cost of missing positive instances is high, such as in disease detection.

It is calculated as:

$$
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
$$

**Example:**  
In a cancer detection model, recall is the proportion of actual cancer cases that the model correctly identifies.

---

### F1 score

The F1 score is the harmonic mean of precision and recall. It provides a balanced metric when both precision and recall are important, especially for imbalanced datasets.

It is calculated as:

$$
\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**Example:**  
A model with high precision but low recall or vice versa will have a lower F1 score, indicating that it is not performing well on both metrics.

---

### Confusion matrix

A confusion matrix is a table used to summarize the performance of a classification model. It shows the number of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN).

|                | Predicted positive | Predicted negative |
|----------------|-------------------|-------------------|
| Actual positive| True positive (TP) | False negative (FN)|
| Actual negative| False positive (FP)| True negative (TN) |

From this matrix, you can calculate accuracy, precision, recall, and other metrics. It provides a more comprehensive view of model performance than accuracy alone.

---

### ROC curve and AUC

The receiver operating characteristic (ROC) curve plots the true positive rate (recall) against the false positive rate (FPR) at different threshold levels. The area under the curve (AUC) measures the overall performance of the classifier.

AUC ranges from 0 to 1, where a value closer to 1 indicates a better-performing model.

ROC AUC is particularly useful when you want to evaluate how well a model can distinguish between classes across different thresholds.

---

## Evaluation metrics for regression models

Regression models predict continuous values, such as house prices or temperatures. The following metrics are commonly used to evaluate regression models:

### Mean squared error

MSE measures the average squared difference between the predicted and actual values. It is sensitive to large errors because the errors are squared, making it useful for situations in which larger errors are more significant.

\[
$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} \left(y_i - \hat{y}_i\right)^2
$$
\]

**Example:**  
In a house price prediction model, if the predicted price is \$200,000 and the actual price is \$250,000, the mean squared error for that prediction is \$2.5 * 10^9.

---

### Root mean squared error

Root mean squared error (RMSE) is the square root of the mean squared error, which brings the error metric back to the same units as the target variable. RMSE is more interpretable because it is in the same unit as the data being predicted.

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

**Example:**  
If the MSE of a model predicting house prices is \$625,000,000, then the RMSE will be \$25,000, making it easier to interpret.

---

### Mean absolute error

Mean absolute error (MAE) measures the average absolute difference between the predicted and actual values. Unlike MSE, it does not square the errors, so it is less sensitive to outliers.

### Mean absolute error

Mean absolute error (MAE) measures the average absolute difference between the predicted and actual values. Unlike MSE, it does not square the errors, so it is less sensitive to outliers.

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

**Example:**  
In a weather forecasting model, MAE tells you the average difference between the predicted and actual temperatures.

**Example:**  
In a weather forecasting model, MAE tells you the average difference between the predicted and actual temperatures.

---

### R-squared (coefficient of determination)

R-squared explains the proportion of variance in the dependent variable that is predictable from the independent variable(s). It ranges from 0 to 1, where 1 indicates a perfect fit, and 0 means the model does not explain any of the variance.

$$
R^2 = 1 - \frac{\sum_{i=1}^{n} \left(y_i - \hat{y}_i\right)^2}{\sum_{i=1}^{n} \left(y_i - \bar{y}\right)^2}
$$

**Example:**  
An R-squared value of 0.9 means that 90 percent of the variance in house prices is explained by the model.

---

### Adjusted R-squared

Adjusted R-squared adjusts the R-squared value based on the number of features in the model. It penalizes the addition of irrelevant features, providing a more accurate measure of model performance, especially in cases of overfitting.

$$
\text{Adjusted } R^2 = 1 - \left( \frac{(1 - R^2)(n - 1)}{n - p - 1} \right)
$$

Where:

- \( n \) is the number of data points.
- \( p \) is the number of predictors in the model.

**Example:**  
If adding more features to a model decreases the adjusted R-squared, it suggests that the additional features are not improving the model.

---

## Choosing the right evaluation metric

Choosing the right evaluation metric depends on the problem you're solving and the nature of the data. For instance:

- For imbalanced classification problems, use precision, recall, F1 score, or ROC AUC instead of accuracy.
- For regression models, if large errors are particularly undesirable, consider using RMSE or MSE. If you want a metric that is less sensitive to outliers, use MAE.
- For complex models, look at R-squared and adjusted R-squared to assess how well the model explains the variance in the target variable.

---

## Conclusion

Evaluation metrics are essential for understanding and improving ML models. By using the right metrics, you can accurately assess model performance, make necessary adjustments, and ensure that your model is well suited for the task at hand. 

Whether you're working with classification or regression problems, these metrics will provide you with the insight needed to create reliable and effective models.

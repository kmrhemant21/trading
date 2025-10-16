# Practice Activity: Applying evaluation metrics in fine-tuning models

## Introduction

In this reading, we will explore how to apply the evaluation metrics—accuracy, precision, recall, F1 score, receiver operating characteristic - area under the curve (ROC-AUC), and others—to fine-tuned models. Properly applying these metrics helps you measure model performance in a way that aligns with the specific goals of your task. We will also walk through scenarios in which each metric is most applicable and provide examples of how to use them in practice.

By the end of this reading, you will be able to:

- Describe when and why to use different evaluation metrics, such as accuracy, precision, recall, and F1 score.
- Calculate and interpret key evaluation metrics for fine-tuned models using practical Python code.
- Apply ROC-AUC, loss functions, and confusion matrices to analyze model performance.
- Choose the most appropriate evaluation metrics based on your specific task and the costs of false positives and false negatives.
- Use evaluation metrics to guide model improvements and ensure better generalization to unseen data.

## Specific examples of how to apply evaluation metrics

### Example 1: Accuracy

#### When to use
Accuracy is the most intuitive metric and works well for balanced datasets where the number of positive and negative instances is approximately equal. For instance, if you are classifying images of cats and dogs and both classes are equally represented in your dataset, accuracy provides a clear indication of how well the model is performing overall.

#### Example

```python
from sklearn.metrics import accuracy_score

# Actual labels and predicted labels from your model
y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 0, 1, 0, 0, 1]

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")
```

#### Interpretation
In this example, the model correctly predicted the outcome five out of six times, so the accuracy would be 0.83 (or 83 percent).

### Example 2: Precision

#### When to use
Precision is especially useful when false positives carry a significant cost. For example, in fraud detection, high precision ensures that when the model flags a transaction as fraudulent, it is likely to be correct, reducing the cost of investigating legitimate transactions.

#### Example

```python
from sklearn.metrics import precision_score

# Calculate precision
precision = precision_score(y_true, y_pred)
print(f"Precision: {precision}")
```

#### Interpretation
If the precision score is 0.80, this means that 80 percent of the transactions flagged as fraudulent by the model were indeed fraudulent. This metric helps control the rate of false positives.

### Example 3: Recall (Sensitivity)

#### When to use
Recall is important when false negatives are costly. In medical diagnoses, for example, recall ensures that a model identifies most of the actual positive cases (e.g., patients with a disease), even if that means some false positives occur.

#### Example

```python
from sklearn.metrics import recall_score

# Calculate recall
recall = recall_score(y_true, y_pred)
print(f"Recall: {recall}")
```

#### Interpretation
If the recall is 0.90, this means the model is identifying 90 percent of the actual positive cases. In scenarios such as disease detection, it is more important to catch as many positive cases as possible, even if some negatives are falsely flagged as positive.

### Example 4: F1 score

#### When to use
The F1 score is a balance between precision and recall. It is particularly useful in cases of imbalanced datasets, in which one class is far more prevalent than the other. For example, in fraud detection, you want a balance between precision (reducing false positives) and recall (catching as many fraud cases as possible).

#### Example

```python
from sklearn.metrics import f1_score

# Calculate F1 score
f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1}")
```

#### Interpretation
An F1 score of 0.85 suggests that the model is performing well, balancing both precision and recall. It is particularly useful when neither false positives nor false negatives should be ignored.

### Example 5: Confusion matrix

#### When to use
A confusion matrix is useful when you want to visualize how well the model is performing on both classes. It shows you the number of true positives (TPs), true negatives (TNs), false positives (FPs), and false negatives (FNs), helping you understand where the model is making errors.

#### Example

```python
from sklearn.metrics import confusion_matrix

# Generate confusion matrix
matrix = confusion_matrix(y_true, y_pred)
print(matrix)
```

#### Interpretation
A confusion matrix provides a more detailed breakdown of your model's predictions, showing TPs, TNs, FPs, and FNs. This helps in understanding where the model is making errors. For example, consider the following confusion matrix:

| | Predicted positive | Predictive negative |
|---|---|---|
| **Actual positive** | 50 (TP) | 10 (FN) |
| **Actual negative** | 5 (FP) | 35 (TN) |

In this case:

- True positives (TP): the model correctly predicted 50 positive instances.
- False negatives (FN): the model missed 10 positive instances.
- False positives (FP): the model incorrectly flagged 5 negative instances as positive.
- True negatives (TN): the model correctly predicted 35 negative instances.

This matrix allows us to see that the model performs well with true positives and true negatives but struggles slightly with false negatives. Depending on the task, such as medical diagnosis, you might prioritize improving recall to catch more positive cases, even if it means slightly increasing the number of false positives.

### Example 6: ROC-AUC 

#### When to use
ROC-AUC is ideal for binary classification tasks and helps evaluate how well your model distinguishes between the positive and negative classes. This is particularly useful in imbalanced datasets, in which accuracy may not tell the full story.

#### Example

```python
from sklearn.metrics import roc_auc_score

# Calculate ROC-AUC score
roc_auc = roc_auc_score(y_true, y_pred)
print(f"ROC-AUC: {roc_auc}")
```

#### Interpretation
An ROC-AUC score close to 1.0 indicates that the model is very good at distinguishing between classes. A score of 0.5 suggests that the model is no better than random guessing. For example, a high ROC-AUC score in medical diagnostics ensures that the model effectively differentiates between healthy and diseased patients.

### Example 7: Loss function

#### When to use
Loss is used primarily during model training to show how well the model is learning. For example, in classification problems, cross-entropy loss measures how far off the model's predictions are from the actual values. A lower loss typically indicates a better model during training.

#### Example

```python
import torch
import torch.nn as nn

# Define the cross-entropy loss function
# CrossEntropyLoss is used for classification tasks where the model outputs class probabilities.
# It combines LogSoftmax and Negative Log Likelihood Loss into one function, making it efficient for such tasks.
loss_fn = nn.CrossEntropyLoss()

# Example prediction and actual class (as tensors)
# Here, we create a tensor called 'output' representing the predicted scores (unnormalized) for two data points.
# Each row corresponds to a data point, and the values represent the scores for each class.
# Note that CrossEntropyLoss internally applies the softmax function to these scores to obtain probabilities.
output = torch.tensor([[0.5, 1.5], [2.0, 0.5]])

# 'target' is a tensor representing the actual classes for the two data points.
# In this example, the first data point belongs to class 1, and the second data point belongs to class 0.
# These class indices are zero-based, meaning 0 represents the first class, 1 represents the second, and so on.
target = torch.tensor([1, 0])

# Calculate loss
# The CrossEntropyLoss function will take the predicted scores ('output') and the actual labels ('target')
# to compute the loss value, which quantifies how well the model's predictions match the actual labels.
# Lower loss values indicate better predictions, while higher values indicate more errors.
loss = loss_fn(output, target)

# Print the computed loss value
# '.item()' is used to get the Python scalar value from the tensor containing the loss.
print(f"Loss: {loss.item()}")
```

#### Interpretation
A lower loss value indicates that the model's predictions are closer to the actual values. Loss functions are particularly important during the training phase but are less interpretable during evaluation.

## Choosing the right metric for your task
The evaluation metric you choose should align with the specific goals of your model and the cost of different types of errors:

- Accuracy is useful when classes are balanced and there are no severe consequences for false positives or false negatives.
- Precision should be used when false positives are costly (e.g., fraud detection).
- Recall is critical when missing positives is dangerous (e.g., medical diagnostics).
- F1 score is ideal when both precision and recall matter, especially in imbalanced datasets.
- ROC-AUC is important for binary classification, particularly when you need to evaluate a model's ability to distinguish between positive and negative classes.
- Confusion matrix provides a comprehensive view of the model's performance and can guide improvements.

## Conclusion
Applying the appropriate evaluation metrics to your fine-tuned models ensures that you are accurately measuring their performance in relation to your specific goals. Depending on your task—whether it is fraud detection, medical diagnostics, or general classification—selecting the right metric will help you balance trade-offs between precision, recall, and overall model accuracy.
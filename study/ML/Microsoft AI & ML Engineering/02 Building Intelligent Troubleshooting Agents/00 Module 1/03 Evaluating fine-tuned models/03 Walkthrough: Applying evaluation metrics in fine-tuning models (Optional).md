# Walkthrough: Applying evaluation metrics in fine-tuning models (Optional)

## Introduction

In this reading, we will walk through the proper solution to the activity where you applied evaluation metrics to assess the performance of your fine-tuned model. By following these steps, you'll gain a better understanding of how to interpret the key metrics and how to use them to improve your model's performance.

By the end of this reading, you will be able to:

- Calculate and interpret accuracy, precision, recall, and F1 score for your fine-tuned model.
- Use confusion matrices to identify patterns in your model's errors.
- Evaluate your model's ability to distinguish between classes using ROC-AUC.
- Identify signs of overfitting and underfitting through loss calculations and improve your model accordingly.
- Make informed decisions about which metrics to prioritize based on your task's goals and requirements.

## Step-by-step process to apply evaluation metrics

This reading will guide you through the following steps:

1. Step 1: Dataset preparation and model training
2. Step 2: Accuracy calculation
3. Step 3: Precision and recall
4. Step 4: F1 Score calculation
5. Step 5: Confusion matrix
6. Step 6: ROC-AUC calculation
7. Step 7: Loss calculation (optional)
8. Step 8: Interpreting the results

### Step 1: Dataset preparation and model training

Before we dive into the evaluation metrics, you should have already fine-tuned your model on a task-specific dataset. Ensure that you have split your data into training, validation, and test sets.

Reminder: if your model performed poorly during training, check for potential overfitting (high training accuracy but low test accuracy) or underfitting (low accuracy on both training and test sets). Once you have fine-tuned your model, it's time to evaluate its performance using the proper metrics.

### Step 2: Accuracy calculation

Accuracy is the most basic metric and gives an overall picture of how well your model is performing.

#### Code example

```python
from sklearn.metrics import accuracy_score

# Actual labels (y_true) and predicted labels (y_pred) from the model
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")
```

#### Explanation 

Accuracy is calculated by dividing the number of correct predictions by the total number of predictions. This metric is useful when your dataset is balanced, but it may not be informative for imbalanced datasets.

### Step 3: Precision and recall

Next, you should calculate precision and recall. Precision helps you understand the accuracy of positive predictions, while recall focuses on capturing all actual positives.

#### Code example

```python
from sklearn.metrics import precision_score, recall_score

# Calculate precision
precision = precision_score(y_true, y_pred)
print(f"Precision: {precision}")

# Calculate recall
recall = recall_score(y_true, y_pred)
print(f"Recall: {recall}")
```

#### Explanation

Precision is important when false positives are costly (e.g., fraud detection).

Recall is important when missing positives (false negatives) are costly (e.g., medical diagnoses).

Balancing precision and recall depends on your task's specific goals.

### Step 4: F1 Score calculation

The F1 score combines precision and recall into a single metric, making it useful when both false positives and false negatives matter.

#### Code example

```python
from sklearn.metrics import f1_score

# Calculate F1 score
f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1}")
```

#### Explanation

The F1 score is the harmonic mean of precision and recall, balancing the two metrics. It is particularly useful for imbalanced datasets, where accuracy alone might be misleading.

### Step 5: Confusion matrix

The confusion matrix gives a complete view of your model's performance, showing how many true positives, true negatives, false positives, and false negatives were predicted.

#### Code example

```python
from sklearn.metrics import confusion_matrix

# Generate confusion matrix
matrix = confusion_matrix(y_true, y_pred)
print(f"Confusion Matrix:\n{matrix}")
```

#### Explanation

A confusion matrix allows you to see where the model is making mistakes. For example, if you see a high number of false positives, you may need to adjust your model to improve precision.

### Step 6: ROC-AUC calculation

For binary classification tasks, calculating the receiver operating characteristic - area under the curve (ROC-AUC) score helps you assess how well the model distinguishes between classes.

#### Code example

```python
from sklearn.metrics import roc_auc_score

# Calculate ROC-AUC
roc_auc = roc_auc_score(y_true, y_pred)
print(f"ROC-AUC: {roc_auc}")
```

#### Explanation

The ROC-AUC score evaluates the trade-off between true positive rate (recall) and false positive rate across various thresholds. A high ROC-AUC score (close to 1) indicates that the model is performing well at distinguishing between classes.

### Step 7: Loss calculation (optional)

During training, monitoring the loss function helps to ensure the model is learning. Cross-entropy loss is commonly used for classification tasks.

#### Code example

```python
import torch.nn as nn

# Define the cross-entropy loss function
loss_fn = nn.CrossEntropyLoss()

# Example prediction and actual class (as tensors)
output = torch.tensor([[0.5, 1.5], [2.0, 0.5]])
target = torch.tensor([1, 0])

# Calculate loss
loss = loss_fn(output, target)
print(f"Loss: {loss.item()}")
```

#### Explanation

A lower loss value indicates that the model is making better predictions during training. Monitoring loss helps to detect overfitting or underfitting early in the training process.

### Step 8: Interpreting the results

Once you've calculated these metrics, it's important to interpret them in the context of your task:

- **Accuracy**: how well is the model performing overall?
- **Precision**: are false positives costly? If so, focus on improving precision.
- **Recall**: are false negatives costly? If so, recall should be the priority.
- **F1 score**: use the F1 score for tasks where both false positives and false negatives matter.
- **Confusion matrix**: analyze where the model is making errors (false positives or false negatives) and adjust the model accordingly.
- **ROC-AUC**: evaluate how well the model distinguishes between classes, especially in imbalanced datasets.

## Conclusion

By applying and interpreting these evaluation metrics, you can better understand how your fine-tuned model is performing and where improvements can be made. These metrics provide insights into not only the overall accuracy but also the balance between false positives and false negatives, helping you to optimize the model for your specific task.
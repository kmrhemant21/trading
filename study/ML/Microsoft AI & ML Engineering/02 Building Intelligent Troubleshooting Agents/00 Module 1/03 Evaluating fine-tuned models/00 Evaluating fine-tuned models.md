# Evaluating fine-tuned models

## Introduction

After fine-tuning a pretrained model, it is critical to evaluate its performance on a task-specific dataset. Evaluation helps determine how well the model has adapted to the new task and whether it can generalize to unseen data. In this reading, we will cover key metrics and methods for evaluating fine-tuned models, including accuracy, precision, recall, and F1 score.

By the end of this reading, you will be able to:

- Explain the importance of evaluating fine-tuned models on unseen data.
- Use key evaluation metrics such as accuracy, precision, recall, and F1 score to assess model performance.
- Recognize signs of overfitting and underfitting during fine-tuning.
- Compare the effectiveness of different fine-tuning techniques, including traditional fine-tuning, LoRA, and QLoRA.
- Optimize the trade-off between performance and resource efficiency when evaluating fine-tuning methods.

## Why evaluation matters

The goal of fine-tuning is to adapt a general-purpose, pretrained model to perform well on a specific task. However, even after fine-tuning, there is no guarantee that the model will perform optimally. Evaluating the model is necessary to:

- Ensure the model can generalize to unseen data.
- Identify potential overfitting or underfitting issues.
- Compare the performance of different fine-tuning techniques, such as traditional fine-tuning, LoRA, or QLoRA.

## Key metrics for evaluation

### Accuracy

Accuracy measures the proportion of correctly predicted instances out of the total instances. This is a common metric for classification tasks.

**Example**

$$\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}$$

Imagine a binary classification task with 100 instances:

- Correct predictions: 80
- Total predictions: 100

$$\text{Accuracy} = \frac{80}{100} = 0.8 \text{ (or } 80\%)$$

**When to use**: accuracy is useful when class distribution is balanced and the cost of false positives and false negatives is roughly the same.

### Precision

Precision measures how many of the model's positive predictions are actually correct. Precision is especially useful when false positives are costly.

**Example**

$$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$$

Imagine a binary classification task with 100 instances:

- Correct predictions: 80
- Total predictions: 100

$$\text{Precision} = \frac{80}{80 + 20} = \frac{80}{100} = 0.8$$

**When to use**: precision is important in tasks in which minimizing false positives is more critical than false negatives, such as spam detection or fraud detection.

### Recall

Recall (also known as sensitivity) measures how many actual positives the model successfully identifies. It is particularly useful when minimizing false negatives is essential.

**Example**

$$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$

In a spam detection system:

- True positives: 70 (emails correctly classified as spam)
- False positives: 10 (non-spam emails incorrectly classified as spam)

$$\text{Recall} = \frac{70}{70 + 20} = \frac{70}{90} \approx 0.778$$

**When to use**
Precision is essential in tasks in which:

- False positives are more critical than false negatives.

Examples:

- Spam detection: it avoids incorrectly flagging important emails as spam.
- Fraud detection: it minimizes incorrectly tagging legitimate transactions as fraudulent.

### F1 score

The F1 score is the harmonic mean of precision and recall, providing a balanced metric when both false positives and false negatives matter.

**Example**

$$F_1 = 2 \times \left(\frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}\right)$$

**When to use**: the F1 score is useful when you need a balance between precision and recall, particularly in imbalanced datasets.

### Confusion matrix

A confusion matrix shows the true positives, true negatives, false positives, and false negatives in a table format. It helps visualize the model's performance.

| | Predicted positive | Predicted negative |
|---|---|---|
| **Actual positive** | True positive (TP) | False negative (FN) |
| **Actual negative** | False positive (FP) | True negative (TN) |

**When to use**: a confusion matrix is valuable for understanding where the model is making errors and identifying class imbalances.

## Evaluating performance on unseen data

When evaluating fine-tuned models, it is crucial to test their performance on a test set that the model has not seen during training or validation. This provides an unbiased measure of the model's ability to generalize to real-world data.

- Validation set: used during fine-tuning to tune hyperparameters and monitor performance
- Test set: used after fine-tuning to evaluate the model's final performance on unseen data

## Overfitting and underfitting

One of the key risks during fine-tuning is overfitting or underfitting the model.

### Overfitting

Overfitting occurs when the model performs well on the training data but poorly on unseen data. This happens when the model has memorized the training set instead of learning features that generalize to new data.

**Signs of overfitting**

- High accuracy on the training set but low accuracy on the validation or test sets.

**Solutions**

- Use regularization techniques (e.g., dropout), reduce model complexity, or use data augmentation.

### Underfitting

Underfitting happens when the model performs poorly on both the training and test data. This indicates that the model is too simple to capture the underlying patterns in the data.

**Signs of underfitting**

- Low accuracy on both training and test sets.

**Solutions**

- Increase the complexity of the model, provide more training data, or train for more epochs.

## Comparing techniques: Traditional fine-tuning, LoRA, and QLoRA

When comparing the performance of different fine-tuning techniques, you should consider both the model's performance metrics and the resource efficiency of each technique.

- Performance: compare accuracy, F1 score, precision, and recall across techniques.
- Efficiency: consider training time, memory usage, and computational cost.

For example, if comparing traditional fine-tuning, LoRA, and QLoRA techniques:

- Traditional fine-tuning: typically achieves high performance but requires significant memory and time.
- LoRA: reduces memory usage by fine-tuning only low-rank matrices, often without a major loss in performance.
- QLoRA: combines quantization with low-rank adaptation, further reducing memory usage while maintaining competitive performance.

## Conclusion

Evaluating fine-tuned models is a critical step in understanding their performance and generalization ability. By using a combination of such metrics as accuracy, precision, recall, and F1 score, you can get a comprehensive picture of how well your model performs on the task at hand. It's also important to assess the model's resource efficiency, particularly when comparing different fine-tuning techniques such as LoRA and QLoRA.

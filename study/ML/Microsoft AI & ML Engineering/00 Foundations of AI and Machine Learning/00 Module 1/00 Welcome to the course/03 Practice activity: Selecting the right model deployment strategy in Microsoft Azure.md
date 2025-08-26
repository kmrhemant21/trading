# Practice activity: Selecting the right model deployment strategy in Microsoft Azure

## Introduction

In this activity, you will play the AI/ML engineer at an e-commerce company with the goal of driving customer retention. Review the business use case below and follow the instructions to solve the business problem.

By the end of this activity, you will be able to:

- Analyze and evaluate different ML models in the context of predicting customer churn.
- Make informed decisions about the most suitable model for this use case.
- Justify your choice of model based on model interpretability, predictive accuracy, and resource constraints.

## Scenario

**Business use case: Predicting customer churn for an e-commerce platform**

You are working as an AI/ML engineer for an e-commerce company that is concerned about customer retention. The company has noticed that a significant number of customers are not returning to make additional purchases after their initial transactions. Your task is to develop an ML model that can predict customer churn—the likelihood that a customer will stop doing business with the company.

The company has provided a dataset that includes various customer attributes, such as purchase history, browsing behavior, customer support interactions, and demographic information. Your goal is to build a model that identifies customers at high risk of churning so that the marketing team can target these customers with retention strategies.

## Models to consider

### Model A: Logistic regression

A simple, interpretable model that uses a linear approach to classify customers into “will churn” or “will not churn” categories based on their attributes.

**Pros:**

- High interpretability: easy to understand and explain to non-technical stakeholders.
- Quick to train and deploy: suitable for situations where rapid deployment is needed.
- Low computational cost: ideal for smaller datasets and less complex environments.

**Cons:**

- Limited complexity: may not capture complex patterns in customer behavior.
- May struggle with large, highly variable datasets.

### Model B: Random forest

An ensemble learning method that creates multiple decision trees and merges them to improve accuracy and prevent overfitting.

**Pros:**

- Handles large datasets well: can manage thousands of variables and data points.
- High accuracy: often performs well in prediction tasks, especially with complex data.
- Robust to overfitting: ensemble methods reduce the risk of overfitting to the training data.

**Cons:**

- Less interpretable: the model's complexity makes it harder to explain the results.
- Slower to train: requires more computational resources and time compared to simpler models.

### Model C: Gradient boosting machine (GBM)

Another ensemble technique that builds models sequentially, each new model correcting the errors of the previous one. It’s known for its high predictive power.

**Pros:**

- High predictive accuracy: particularly effective for small to medium-sized structured datasets.
- Can handle non-linear relationships: excels in capturing complex patterns in the data.

**Cons:**

- Computationally intensive: requires significant processing power and time, especially with large datasets.
- Risk of overfitting: if not tuned properly, GBM models can overfit the training data.

## Activity instructions

1. **Analyze the use case.** Consider the specific business problem—predicting customer churn in an e-commerce setting. Think about what aspects of the models are most important for this use case, such as interpretability, accuracy, and computational efficiency.

2. **Evaluate each model.** Reflect on the pros and cons of logistic regression, random forest, and gradient boosting machine (GBM) in the context of the customer churn prediction task. Consider the dataset you are working with, the importance of model interpretability to your business stakeholders, and the resources (time and computational power) you have available for training and deploying the model.

3. **Make your selection.** Choose the model that you believe is best suited for this business use case, and justify your choice by explaining why you selected this model over the others. Consider factors such as the complexity of the problem, the need for accuracy versus interpretability, and the resources available for deployment. 

4. **Justify your selection.** Write down one to two paragraphs about which model you chose and your justification. Be sure to include why that model works better than others.

## Conclusion

In this activity, you assessed various ML models to predict customer churn for an e-commerce platform. By balancing model interpretability with predictive accuracy and computational efficiency, you made an informed choice that addresses the specific needs of the business. Your ability to justify this selection will empower the marketing team to implement targeted retention strategies, ultimately enhancing customer loyalty and driving sustainable growth.
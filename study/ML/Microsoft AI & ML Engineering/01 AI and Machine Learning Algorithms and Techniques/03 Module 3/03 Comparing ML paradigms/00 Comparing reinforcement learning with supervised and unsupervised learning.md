# Comparing reinforcement learning with supervised and unsupervised learning

## Introduction
In the field of ML, three primary learning paradigms are used to teach machines how to perform tasks: supervised learning, unsupervised learning, and reinforcement learning (RL). While all three paradigms aim to train models to make predictions or decisions, the nature of the learning process, the type of feedback provided, and the tasks they are used for differ significantly.

By the end of this reading, you will be able to:

* Differentiate between reinforcement learning and the other two paradigms, supervised and unsupervised learning.

## Supervised learning
### Definition
Supervised learning is a type of ML where the model learns from labeled data. Each input in the dataset is associated with the correct output, and the model's objective is to learn the mapping from inputs to outputs.

### How it works
* **Input**: a labeled dataset, where each input example has a corresponding output label.
* **Learning**: the model minimizes a loss function, comparing its predictions to the true labels.
* **Output**: a prediction model that can accurately assign labels to new, unseen inputs.

### Key characteristics
* **Direct feedback**: the model receives explicit feedback (the correct output label) for each prediction during training.
* **Common tasks**: supervised learning is widely used for classification (e.g., image classification, spam detection) and regression (e.g., predicting house prices).
* **Evaluation**: the model is evaluated using metrics such as accuracy, precision, recall, and mean squared error (MSE).

### Example
Consider training a supervised learning model to classify images of cats and dogs. Each image in the training dataset is labeled as either "cat" or "dog," and the model learns to distinguish between the two based on the labeled examples.

## Unsupervised learning
### Definition
Unsupervised learning involves training models on data without labeled outputs. The goal is to discover hidden patterns or structures within the data without any explicit feedback on what those patterns should be.

### How it works
* **Input**: an unlabeled dataset, where the relationships between data points are not predefined.
* **Learning**: the model tries to identify patterns or group similar data points together.
* **Output**: clusters of similar data points, lower-dimensional representations, or new features.

### Key characteristics
* **No feedback**: there are no labels or predefined outputs, so the model does not receive feedback during training.
* **Common tasks**: unsupervised learning is used for clustering (e.g., customer segmentation, image grouping) and dimensionality reduction (e.g., principal component analysis [PCA], t-distributed stochastic neighbor embedding [t-SNE]).
* **Evaluation**: evaluation is more challenging, as there are no ground truth labels. Methods such as silhouette score or visual inspection of clusters are often used.

### Example
In an unsupervised learning scenario, you might use a clustering algorithm like k-means to group customers based on their purchasing behavior. The algorithm identifies clusters of similar customers without being explicitly told which customers belong together.

## Reinforcement learning (RL)
### Definition
RL is a learning paradigm where an agent interacts with an environment and learns to take actions that maximize cumulative rewards. The agent receives feedback in the form of rewards or penalties after taking actions but does not receive direct supervision on which actions are optimal.

### How it works
* **Input**: an environment in which the agent operates, with states, actions, and rewards.
* **Learning**: the agent learns by exploring the environment, taking actions, and receiving feedback in the form of rewards. Over time, the agent adjusts its policy to maximize the long-term reward.
* **Output**: a policy that dictates which action to take in each state to achieve the highest cumulative reward.

### Key characteristics
* **Delayed feedback**: Unlike supervised learning, the agent does not receive immediate feedback on each individual action. Rewards may be delayed, and the agent must consider the long-term consequences.
* **Trial-and-error**: the agent learns through exploration (trying new actions) and exploitation (using known actions that yield high rewards).
* **Common tasks**: RL is used in tasks that require sequential decision-making, such as game playing (e.g., AlphaGo), robotics, and autonomous vehicle navigation.
* **Evaluation**: performance is evaluated by metrics such as cumulative reward, time to convergence, and success rate.

### Example
In a classic RL example, an agent might learn to play a game such as chess or Go. It receives rewards for winning the game or making beneficial moves and learns over time to refine its strategy based on these rewards.

## Key differences between reinforcement, supervised, and unsupervised learning

| Paradigm | Supervised learning | Unsupervised learning | RL |
|----------|---------------------|------------------------|-----|
| Type of feedback | Labeled data with correct output for each input | No labeled data; model identifies patterns on its own | Feedback is in the form of rewards or penalties after actions |
| Goal | Learn a mapping from inputs to correct outputs | Discover hidden patterns or groupings in data | Learn to take actions that maximize long-term cumulative rewards |
| Learning process | Minimizes a loss function to improve accuracy | Optimizes for pattern recognition, clustering, or dimensionality reduction | Uses trial and error, exploring different actions to learn a policy |
| Tasks | Classification, regression | Clustering, dimensionality reduction | Sequential decision-making, game playing, robotics |
| Evaluation | Metrics such as accuracy, precision, recall, MSE | Clustering metrics, silhouette score, visual inspection | Cumulative reward, success rate, convergence time |
| Examples | Image classification, spam detection, price prediction | Customer segmentation, anomaly detection | Autonomous driving, robot navigation, game playing (e.g., AlphaGo) |

## When to use each learning paradigm
* **Supervised learning**: best used when you have a labeled dataset and the goal is to make accurate predictions (e.g., identifying whether an email is spam or not).
* **Unsupervised learning**: useful when you want to discover patterns or groupings in the data without predefined labels (e.g., segmenting customers based on purchasing behavior).
* **RL**: ideal for tasks involving sequential decision-making and environments where feedback comes in the form of rewards, often after a series of actions (e.g., training an agent to navigate a maze or play a video game).

## Conclusion
While all three paradigms—supervised learning, unsupervised learning, and RL—aim to train machines, they differ in their learning processes, the type of feedback they rely on, and the tasks they are best suited for. Supervised learning is appropriate for tasks with labeled data, unsupervised learning for discovering patterns, and RL for decision-making tasks with delayed rewards. By understanding these differences, you can select the most suitable paradigm for your ML tasks.

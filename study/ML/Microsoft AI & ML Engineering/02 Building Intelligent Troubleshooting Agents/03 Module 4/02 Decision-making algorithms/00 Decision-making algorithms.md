# Decision-making algorithms

## Introduction
AI and machine learning (ML) are transforming how we solve problems across industries. At the core of these innovations are decision-making algorithms—powerful tools that enable machines to make informed choices. Imagine teaching an AI agent to play chess or helping a medical system predict diseases from patient data—these breakthroughs are possible because of decision-making algorithms. In this reading, we will explore key algorithms that form the backbone of intelligent systems.

By the end of this reading, you will be able to:

- Identify various decision-making algorithms commonly used in AI/ML systems, such as decision trees, random forests, and reinforcement learning.
- Differentiate between these algorithms based on their structure, function, and appropriate use cases.
- Apply decision-making algorithms to practical AI/ML problems, such as classification, regression, and optimization tasks.

## Decision trees
A decision tree is a flowchart-like structure in which each internal node represents a decision based on a feature, each branch represents the decision's outcome, and each leaf node represents a class label (for classification) or a value (for regression).

### How it works
The algorithm recursively splits the data into subsets based on feature values that provide the best separation between classes (for classification) or the least error (for regression). This selection of the best split is typically based on metrics like Gini impurity, entropy (information gain), or mean squared error, depending on the type of task.

The process continues until the subsets are homogeneous or a stopping criterion, such as maximum depth, is met, which prevents the tree from becoming overly complex and helps avoid overfitting. This stopping point is essential for balancing accuracy and generalizability, allowing the model to perform well on unseen data.

### Example use case
Classification task: predicting whether a patient has a particular disease based on symptoms (e.g., fever, headache, or fatigue)

### Advantages
- Highly interpretable and easy to visualize
- Can handle both numerical and categorical data

### Disadvantages
- Prone to overfitting, especially with deep trees
- Can be unstable, as small changes in the data might result in different trees

## Random forests
A random forest is an ensemble learning method that operates by constructing multiple decision trees during training and outputting the class that is the mode of the classes (for classification) or mean prediction (for regression) of the individual trees.

### How it works
Random forests create multiple decision trees using bootstrapped subsets of the data and randomly selecting a subset of features at each split.

The final prediction is made by averaging the results of all trees for regression or by majority voting for classification.

### Example use case
Predictive modeling: predicting the likelihood of customer churn based on user activity and demographics

### Advantages
- Reduces overfitting compared to a single decision tree by averaging multiple trees
- Works well with large datasets and high-dimensional data

### Disadvantages
- Less interpretable than a single decision tree
- Can be computationally expensive and slower to train

## Support vector machines
Support vector machines (SVMs) are powerful classification algorithms that find a hyperplane in a high-dimensional space that maximally separates the data points of different classes. For nonlinearly separable data, SVMs use kernel functions to transform the input space into a higher dimension where a linear separation is possible.

### How it works
SVMs create a decision boundary, known as a hyperplane, that separates the data into different classes with the largest possible margin.

Support vectors are the data points closest to the hyperplane, which directly influence the position and orientation of the hyperplane.

### Example use case
Image classification: classifying images as either containing an object (e.g., a cat) or not based on pixel intensity values

### Advantages
- Effective in high-dimensional spaces
- Works well with clear margin of separation between classes

### Disadvantages
- Can be less effective in large datasets with noisy data

## Reinforcement learning algorithms
Reinforcement learning (RL) is an area of machine learning where an agent learns to make decisions by performing actions in an environment to maximize cumulative rewards. Instead of being given labeled data, the agent must explore the environment and learn from the feedback it receives (rewards or penalties) based on its actions. Without a known target comparison, the algorithm evaluates feedback as positive or negative by associating rewards with actions that lead to desirable outcomes and penalties with those that do not, allowing it to infer the best actions over time. This process relies on trial and error, where the agent gradually improves its strategy by reinforcing actions that yield higher rewards, even in the absence of predefined answers.

### How it works
The RL agent interacts with the environment by taking actions that lead to different states. For each action, the agent receives a reward (or penalty) that informs whether the action was beneficial.

The goal is to learn a policy that maximizes the total cumulative reward over time.

### Common algorithms
- **Q-learning**: a value-based method that learns the expected utility of taking a particular action in a given state and updates its values iteratively based on the rewards received
- **Deep Q-networks**: combines deep learning with Q-learning by using neural networks to approximate the Q-value function, allowing it to handle high-dimensional state spaces

### Example use case
Game AI: training an AI agent to play games such as chess or Go by learning which moves maximize the chance of winning

### Advantages
- Can be applied to problems in which labeled data is unavailable or difficult to obtain
- Excels at sequential decision-making tasks

### Disadvantages
- Requires a significant amount of data and time to train
- The exploration-exploitation trade-off (balancing the need to explore new actions vs. exploiting known good actions) can be challenging to manage

## Bayesian networks
A Bayesian network is a probabilistic graphical model that represents a set of variables and their conditional dependencies via a directed acyclic graph. Each node represents a variable, and edges between the nodes represent conditional dependencies between variables.

*A Bayesian network diagram illustrating causal links between weather variables like Season, Temperature, Rain, and Humidity.*


![slide_1.png](slide_1.png)

### How it works
Bayesian networks use Bayes' theorem to update the probability of an event based on new evidence.

The relationships between variables are represented as probabilities, allowing for the calculation of posterior probabilities given observed data.

### Example use case
Medical diagnosis: predicting the likelihood of a patient having a disease based on symptoms and medical history

### Advantages
- Provides a structured way to model uncertainty in complex systems
- Can incorporate both prior knowledge and observed data

### Disadvantages
- Requires detailed knowledge of conditional dependencies between variables
- Computationally expensive for large networks with many variables

## Markov decision processes
Markov decision processes (MDPs) are a mathematical framework for modeling decision-making in situations where outcomes are partly random and partly under the control of a decision-maker. MDPs are used in reinforcement learning and consist of states, actions, transition probabilities, and rewards.

### How it works
An agent in an MDP takes actions that move it from one state to another, with each transition governed by a probability distribution. The agent receives rewards based on the states it visits, and the goal is to maximize the total reward.

### Example use case
Robotics: teaching a robot to navigate through a room by maximizing the number of tasks completed within a given time frame

### Advantages
- Provides a formal mathematical approach for sequential decision-making
- Used in complex domains such as robotics and finance

### Disadvantages
- Assumes full knowledge of the environment's dynamics, which may not always be practical
- Can become intractable in very large or continuous state-action spaces

## Genetic algorithms
Genetic algorithms are optimization techniques inspired by the process of natural selection. They are used to find approximate solutions to optimization and search problems by evolving a population of candidate solutions over several generations.

### How it works
The algorithm starts with a population of random solutions. Each solution is evaluated using a fitness function, and the best solutions are selected to form a new generation through processes such as mutation, crossover, and selection.

Over time, the population "evolves" toward better solutions.

### Example use case
Optimization problems: finding the optimal set of parameters for an ML model by evolving different combinations over time

### Advantages
- Effective for solving complex optimization problems with large search spaces
- Can find global optima where other methods may get stuck in local optima

### Disadvantages
- Computationally expensive due to the need to evaluate a large number of candidate solutions
- Results can vary depending on the choice of mutation and crossover parameters

## Conclusion
Understanding decision-making algorithms is essential for developing AI/ML systems capable of solving complex problems. Each algorithm has its unique strengths and limitations, but knowing when and how to use them will empower you to build more efficient, effective AI solutions. Whether you are optimizing a business process or building an AI for interactive games, these decision-making tools are key to your success.

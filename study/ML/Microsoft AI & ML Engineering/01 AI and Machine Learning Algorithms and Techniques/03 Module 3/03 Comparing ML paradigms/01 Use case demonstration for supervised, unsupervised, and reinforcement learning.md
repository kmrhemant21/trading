# Use case demonstration for supervised, unsupervised, and reinforcement learning

## Introduction

AI and ML are transforming various industries by enabling machines to learn from data and make predictions or decisions. This reading will delve into three primary learning paradigms: supervised learning, unsupervised learning, and RL. By exploring practical use cases, you will gain insights into how these approaches function in real-world scenarios, equipping you with the knowledge to apply these concepts effectively.

By the end of this reading, you will be able to:

- Apply the concepts of supervised learning, unsupervised learning, and RL to real-world use cases. 
- Solidify your understanding of each learning paradigm by seeing how they function in practical scenarios. 
- Select the appropriate learning approach based on the problem at hand and the type of data available.

## Supervised learning: Predicting house prices

### Use case overview

In a supervised learning scenario, we aim to predict house prices based on features such as square footage, the number of bedrooms, and the neighborhood. We have a dataset where each house is labeled with its corresponding price, making it a classic supervised learning regression problem.

### Data

Input features (X): square footage, number of bedrooms, neighborhood, lot size

Target (Y): house price

### Solution approach

We can use a regression algorithm, such as linear regression, to map the input features to the target price. The model will be trained using labeled data, where the correct house prices are provided, allowing the model to learn how to predict prices based on the given features.

### Steps involved

1. Data collection: gather historical house data, including features (square footage, etc.) and prices.
2. Data preprocessing: clean the data, normalize the features, and split it into training and test sets.
3. Model training: train a linear regression model on the training data.
4. Model evaluation: use metrics such as MSE to evaluate the accuracy of the model on the test data.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Sample data (square footage, bedrooms, neighborhood as encoded values)
X = [[2000, 3, 1], [1500, 2, 2], [1800, 3, 3], [1200, 2, 1]]
y = [500000, 350000, 450000, 300000]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
```

### Outcome

After training, the model can predict house prices for new homes based on the given features. In this case, supervised learning is ideal because we have labeled data and a clear target variable.

## Unsupervised learning: Customer segmentation

### Use case overview

For a business trying to understand customer behavior, unsupervised learning can be applied to segment customers based on their purchasing habits. The company wants to group similar customers together to better target marketing strategies, but they do not have predefined labels for these customer segments.

### Data

Input features (X): number of purchases, total spending, product categories purchased

### Solution approach

In this case, we can use a clustering algorithm like k-means to group customers based on the similarity of their behaviors. The algorithm will learn to segment customers without needing predefined labels.

### Steps involved

1. Data collection: gather customer purchasing data, such as the number of purchases, spending amount, and categories purchased.
2. Data preprocessing: standardize the data to ensure that features such as spending and purchases are on similar scales.
3. Model training: use the k-means algorithm to create clusters of customers with similar behavior.
4. Cluster analysis: analyze the resulting clusters to understand the common characteristics of customers within each group.

```python
from sklearn.cluster import KMeans
import numpy as np

# Sample customer data (number of purchases, total spending, product categories)
X = np.array([[5, 1000, 2], [10, 5000, 5], [2, 500, 1], [8, 3000, 3]])

# Create and fit the KMeans model
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

# Print the cluster centers and labels
print(f"Cluster Centers: {kmeans.cluster_centers_}")
print(f"Labels: {kmeans.labels_}")
```

### Outcome

After clustering, the customers are grouped into distinct segments, allowing the business to target marketing efforts more effectively. For example, one cluster may represent high-spending, frequent buyers, while another represents lower-spending, infrequent buyers. Unsupervised learning is appropriate here because there are no predefined labels for customer segments.

## RL: Training an agent to play tic-tac-toe

### Use case overview

In this RL scenario, we train an agent to play the game of tic-tac-toe. The agent interacts with the game environment by placing its marks (X or O) on the board and receives rewards for winning (+1), losing (-1), or drawing (0). There are no labeled data or predefined strategies; the agent must learn through trial and error to improve its gameplay.

### Data

State: the current configuration of the tic-tac-toe board

Actions: the available positions where the agent can place its mark

Reward: +1 for a win, -1 for a loss, and 0 for a draw

### Solution approach

We can use a Q-learning algorithm to train the agent. The agent will play multiple games, and based on the outcomes, it will update its policy to maximize its chances of winning in future games.

### Steps involved

1. Define the environment: the tic-tac-toe board, possible moves, and rules.
2. Initialize Q-table: store the Q-values for each state-action pair.
3. Train the agent: the agent plays multiple games, updating its Q-values based on rewards from winning, losing, or drawing.
4. Evaluate the agent: after training, evaluate the agent's performance by playing it against a human player or another trained agent.

```python
import numpy as np

# Initialize Q-table with zeros for all state-action pairs
Q_table = np.zeros((9, 9))  # 9 possible states (board positions) and 9 possible actions

# Learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

# Sample function to select action using epsilon-greedy policy
def epsilon_greedy_action(state, Q_table, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(0, 9)  # Random action (explore)
    else:
        return np.argmax(Q_table[state])  # Best action (exploit)

# Update Q-values after each game (simplified example)
def update_q_table(state, action, reward, next_state, Q_table):
    Q_table[state, action] = Q_table[state, action] + alpha * (
        reward + gamma * np.max(Q_table[next_state]) - Q_table[state, action]
    )

# Example simulation of a game where the agent learns
for episode in range(1000):
    state = np.random.randint(0, 9)  # Random initial state
    done = False
    while not done:
        action = epsilon_greedy_action(state, Q_table, epsilon)
        next_state = np.random.randint(0, 9)  # Simulate next state
        reward = 1 if next_state == 'win' else -1 if next_state == 'loss' else 0  # Simulate rewards
        update_q_table(state, action, reward, next_state, Q_table)
        state = next_state
        if reward != 0:
            done = True  # End the game if win/loss
```

### Outcome

After playing many games, the agent learns to improve its strategy by adjusting its actions based on the rewards it receives. The agent can eventually play tic-tac-toe competitively, maximizing its chances of winning. RL is used here because the agent learns by interacting with the environment and receiving feedback from game outcomes.

## Conclusion

In this demonstration, we applied each learning paradigm—supervised learning, unsupervised learning, and RL—to practical use cases. Each paradigm is suited for different types of tasks and data:

- Supervised learning is best for tasks with labeled data, such as predicting house prices.
- Unsupervised learning is ideal for tasks where labels are not available, such as customer segmentation.
- RL shines in scenarios where an agent learns through interactions and feedback, such as training an agent to play a game.

Understanding these real-world applications will help you choose the appropriate learning method for your specific tasks.
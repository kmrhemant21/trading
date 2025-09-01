# Practice activity: Implementing reinforcement learning and other approaches

## Introduction
In this activity, you will implement three different ML paradigms: supervised learning, unsupervised learning, and reinforcement learning. This hands-on experience will help you understand the differences between these approaches and how they apply to specific use cases. 

By the end of the activity, you will be able to:

- Implement models for each learning paradigm. 
- Evaluate each model's performance on different tasks.

## Set up your environment
Before starting, make sure you have the necessary libraries installed. You will be using Python along with the following libraries:

- NumPy for numerical operations.
- Scikit-Learn for supervised and unsupervised learning.
- Gym for reinforcement learning environments.
- Matplotlib for visualization.

To install these libraries, use the following command:

```bash
pip install numpy scikit-learn gym matplotlib
```

## Supervised learning: Predicting house prices
In this task, you will use supervised learning to predict house prices based on features such as square footage, number of bedrooms, and location. You will train a linear regression model using a labeled dataset.

### Step-by-step guide:
#### Step 1: Prepare the dataset
You will use a small dummy dataset of house features and prices.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


X = np.array([[2000, 3, 1], [1500, 2, 2], [1800, 3, 3], [1200, 2, 1], [2200, 4, 2]])
y = np.array([500000, 350000, 450000, 300000, 550000])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

#### Step 2: Train the model
Train a linear regression model on the training data.

```python
# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
```

#### Step 3: Evaluate the model
Evaluate the model on the test data using mean squared error (MSE) as the evaluation metric.

```python
from sklearn.metrics import mean_squared_error

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
```

## Unsupervised learning: Customer segmentation
In this task, you will use unsupervised learning to group customers based on their purchasing behavior. You will implement the k-means clustering algorithm to discover natural groupings in the data.

### Step-by-step guide:
#### Step 1: Prepare the dataset
Create a dataset in which each row represents a customer, and the columns represent the number of purchases, total spending, and product categories purchased.

```python
from sklearn.cluster import KMeans
import numpy as np

# Sample customer data: number of purchases, total spending, product categories purchased
X = np.array([[5, 1000, 2], [10, 5000, 5], [2, 500, 1], [8, 3000, 3], [12, 6000, 6]])
```

#### Step 2: Train the model
Fit the k-means clustering algorithm to the dataset to create customer clusters.

```python
# Create and fit the KMeans model
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

# Print the cluster centers and labels
print(f"Cluster Centers: {kmeans.cluster_centers_}")
print(f"Labels: {kmeans.labels_}")
```

#### Step 3: Visualize the results
Use a simple plot to visualize the customer clusters (optional).

```python
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.xlabel('Number of Purchases')
plt.ylabel('Total Spending')
plt.title('Customer Segmentation using K-Means Clustering')
```

## Reinforcement learning: Training an AI agent to play tic-tac-toe
In this task, you will implement reinforcement learning using the Q-learning algorithm to train an AI agent to play tic-tac-toe. The agent will receive rewards for winning, losing, or drawing a game and will adjust its strategy to maximize its chances of winning.

### Step-by-step guide:
#### Step 1: Set up the environment
Use a basic grid environment in which the agent can place its marks (X or O) on the tic-tac-toe board.

```python
import numpy as np
import random
import matplotlib.pyplot as plt

# Initialize the Q-table
Q = {}

# Define the Tic-Tac-Toe board
def initialize_board():
    return np.zeros((3, 3), dtype=int)
```

#### Step 2: Implement and train with the Q-learning algorithm
Train the agent over multiple games, updating the Q-table based on rewards.

```python
# Check for a win
def check_win(board, player):
    for i in range(3):
        if np.all(board[i, :] == player) or np.all(board[:, i] == player):
            return True
    if board[0, 0] == board[1, 1] == board[2, 2] == player or board[0, 2] == board[1, 1] == board[2, 0] == player:
        return True
    return False

# Check for a draw
def check_draw(board):
    return not np.any(board == 0)

# Get available actions
def get_available_actions(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i, j] == 0]

# Choose an action using epsilon-greedy policy
def choose_action(state, board, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(get_available_actions(board))
    else:
        if state in Q and Q[state]:
            # Choose the action with the maximum Q-value
            return max(Q[state], key=Q[state].get)
        else:
            # No action in Q-table, choose random
            return random.choice(get_available_actions(board))

# Update Q-value
def update_q_value(state, action, reward, next_state, alpha, gamma):
    max_future_q = max(Q.get(next_state, {}).values(), default=0)
    current_q = Q.get(state, {}).get(action, 0)
    new_q = current_q + alpha * (reward + gamma * max_future_q - current_q)
    if state not in Q:
        Q[state] = {}
    Q[state][action] = new_q

# Convert board to a tuple (hashable type)
def board_to_tuple(board):
    return tuple(map(tuple, board))

# Train the agent
def train(episodes, alpha=0.1, gamma=0.9, epsilon=0.1):
    win_history = []
    for episode in range(episodes):
        board = initialize_board()
        state = board_to_tuple(board)
        done = False
        result = None  # Initialize result
        while not done:
            action = choose_action(state, board, epsilon)
            board[action[0], action[1]] = 1
            next_state = board_to_tuple(board)
            if check_win(board, 1):
                update_q_value(state, action, 1, next_state, alpha, gamma)
                result = 1  # Agent won
                done = True
            elif check_draw(board):
                update_q_value(state, action, 0.5, next_state, alpha, gamma)
                result = 0  # Draw
                done = True
            else:
                opponent_action = random.choice(get_available_actions(board))
                board[opponent_action[0], opponent_action[1]] = -1
                next_state = board_to_tuple(board)
                if check_win(board, -1):
                    update_q_value(state, action, -1, next_state, alpha, gamma)
                    result = -1  # Agent lost
                    done = True
                elif check_draw(board):
                    update_q_value(state, action, 0.5, next_state, alpha, gamma)
                    result = 0  # Draw
                    done = True
                else:
                    update_q_value(state, action, 0, next_state, alpha, gamma)
            state = next_state
        # Record the result
        if result == 1:
            win_history.append(1)
        else:
            win_history.append(0)
    return win_history

# Train the agent for 10000 episodes
win_history = train(10000)

# Calculate the moving average of win rate
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

# Set the window size for the moving average
window_size = 100

# Compute the moving average
win_rate = moving_average(win_history, window_size)

# Generate episodes for plotting
episodes = np.arange(window_size, len(win_history) + 1)

# Plot the win rate over time
plt.figure(figsize=(12,6))
plt.plot(episodes, win_rate, label='Win Rate')
plt.xlabel('Episodes')
plt.ylabel('Win Rate')
plt.title('Agent Win Rate Over Time (Moving Average over {} episodes)'.format(window_size))
plt.legend()
plt.show()
```

#### Step 3: Evaluate the agent
After training, you can evaluate the agent's performance by observing how it performs against another AI agent or a human player.

## Conclusion
After completing all three tasks, you should have a working understanding of how to implement and evaluate supervised learning, unsupervised learning, and reinforcement learning approaches. For each task:

- Document your observations on how the models performed.
- Include plots or metrics (e.g., MSE for supervised learning, cluster visualizations for unsupervised learning, and win rates for reinforcement learning).
- Reflect on which learning paradigm was most challenging to implement and why.
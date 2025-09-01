# Walkthrough: Applying model evaluation metrics (Optional)

## Introduction

In this walkthrough, we will explore essential model evaluation metrics for reinforcement learning (RL), focusing on applying these metrics to your Q-learning agent. Evaluating the performance of RL agents is a critical step in understanding how well they are learning from their environment and optimizing their actions. By using a variety of metrics, you can gain insights into the agent's learning progress and identify areas where further tuning or adjustments may be needed.

Throughout this walkthrough, you'll be guided through the practical implementation of several key evaluation metrics. These metrics will not only help you track your agent's performance but also provide a framework for improving future RL models in more complex environments. Understanding these metrics is crucial for interpreting the learning process and determining whether the agent is balancing exploration and exploitation effectively.

By the end of this walkthrough, you will be able to:
- Interpret the results of each metric and how to use them to improve the performance of your RL models.

## Step-by-step guide for model evaluation

### Step 1: Review environment and agent setup

The agent was trained in a 5 x 5 grid environment where it navigated from a random starting state to a goal state while avoiding pitfalls. The agent was trained using Q-learning, and the reward structure was as follows:

- +10 for reaching the goal (state 24).
- –10 for falling into the pitfall (state 12).
- –1 for other actions to encourage efficient navigation.

#### Key environment setup

```python
import numpy as np

# Define the environment
grid_size = 5
n_states = grid_size * grid_size
n_actions = 4  # Up, down, left, right

# Initialize reward matrix (goal: +10, pitfalls: -10, others: -1)
rewards = np.full((n_states,), -1)
rewards[24] = 10  # Goal at state 24 (bottom-right)
```

#### Key Q-learning algorithm

```python
def epsilon_greedy_action(Q_table, state, epsilon):
    # Epsilon-greedy strategy: with probability epsilon, take a random action (exploration)
    # otherwise take the action with the highest Q-value for the given state (exploitation)
    if np.random.rand() < epsilon:  # Exploration
        return np.random.randint(0, Q_table.shape[1])  # Random action
    else:  # Exploitation
        return np.argmax(Q_table[state])  # Action with the highest Q-value


alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate for epsilon-greedy policy

# Initialize the Q-table
Q_table = np.zeros((n_states, n_actions))

# Training loop
for episode in range(1000):
    state = np.random.randint(0, n_states)  # Start at random state
    done = False
    while not done:
        action = epsilon_greedy_action(Q_table, state, epsilon)
        next_state = np.random.randint(0, n_states)  # Random next state
        reward = rewards[next_state]

        # Q-learning update rule
        Q_table[state, action] = Q_table[state, action] + alpha * (reward + gamma * np.max(Q_table[next_state]) - Q_table[state, action])

        state = next_state
        if next_state == 24 or next_state == 12:  # End episode if goal or pitfall is reached
            done = True
```

### Step 2: Measure cumulative reward

The cumulative reward is a key metric that measures the total reward an agent collects over the course of an episode. It helps track how well the agent is learning to maximize positive rewards and avoid penalties.

#### Solution

To calculate cumulative rewards over the 1,000 episodes:

```python
import matplotlib.pyplot as plt

# Calculate and store cumulative rewards
cumulative_rewards = []
for episode in range(1000):
    total_reward = 0
    state = np.random.randint(0, n_states)
    done = False
    while not done:
        action = epsilon_greedy_action(Q_table, state, epsilon)
        next_state = np.random.randint(0, n_states)
        reward = rewards[next_state]
        total_reward += reward
        state = next_state
        if next_state == 24 or next_state == 12:
            done = True
    cumulative_rewards.append(total_reward)
```

#### Interpretation

After running the above code, you can plot the cumulative rewards:

```python
# Plot the cumulative rewards over episodes
plt.plot(cumulative_rewards)
plt.xlabel('Episodes')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward Over Episodes')
plt.show()
```

If the agent is learning well, you should see an upward trend in cumulative rewards as the episodes progress, indicating that the agent is improving at reaching the goal and avoiding pitfalls.

### Step 3: Measure episode length

Episode length measures the number of steps the agent takes to complete an episode. Decreasing episode lengths over time typically indicate that the agent is learning to reach the goal more efficiently.

#### Solution

```python
# Calculate and store episode lengths
episode_lengths = []
actions = []
for episode in range(1000):
    steps = 0
    state = np.random.randint(0, n_states)
    done = False
    while not done:
        action = epsilon_greedy_action(Q_table, state, epsilon)
        
        next_state = np.random.randint(0, n_states)
        steps += 1
        state = next_state
        if next_state == 24 or next_state == 12:
            done = True
    episode_lengths.append(steps)
```

#### Visualization

Plot the distribution of episode lengths to see how quickly the agent learns to reach the goal:

```python
# Plot a histogram of episode lengths
plt.hist(episode_lengths, bins=20)
plt.xlabel('Episode Length (Steps)')
plt.ylabel('Frequency')
plt.title('Distribution of Episode Lengths')
plt.show()
```

#### Interpretation

If the agent is learning to navigate efficiently, you will see more episodes with shorter lengths as the training progresses. This indicates that the agent is figuring out the optimal path to the goal.

### Step 4: Measure success rate and exploration vs. exploitation ratio

The success rate is a straightforward metric that measures how often the agent successfully reaches the goal. This metric helps assess how reliable the agent is at completing the task.

#### Solution

```python
# Redefine epsilon_greedy_action to log explorations & exploitations
actions = []
def epsilon_greedy_action(Q_table, state, epsilon):
    # Epsilon-greedy strategy: with probability epsilon, take a random action (exploration)
    # otherwise take the action with the highest Q-value for the given state (exploitation)
    if np.random.rand() < epsilon:  # Exploration
        actions.append('explore')
        return np.random.randint(0, Q_table.shape[1])  # Random action
    else:  # Exploitation
        actions.append('exploit')
        return np.argmax(Q_table[state])  # Action with the highest Q-value

# Calculate and store cumulative rewards and actions
cumulative_rewards = []
for episode in range(1000):
    total_reward = 0
    state = np.random.randint(0, n_states)
    done = False
    while not done:
        action = epsilon_greedy_action(Q_table, state, epsilon)
        next_state = np.random.randint(0, n_states)
        reward = rewards[next_state]
        total_reward += reward
        state = next_state
        if next_state == 24 or next_state == 12:
            done = True
    cumulative_rewards.append(total_reward)

# Calculate success rate
success_count = sum(1 for reward in cumulative_rewards if reward >= 10)
success_rate = success_count / len(cumulative_rewards)

# Exploration vs. exploitation ratio
#print(actions)
exploration_count = sum(1 for action in actions if action == 'explore')
exploitation_count = sum(1 for action in actions if action == 'exploit')
exploration_exploitation_ratio = exploration_count / (exploration_count + exploitation_count)
```

#### Interpretation

```python
print(f"Success Rate: {success_rate * 100}%")
print(f"Exploration vs. Exploitation Ratio: {exploration_exploitation_ratio}")
```

A high success rate means that the agent is consistently reaching the goal. If the success rate is low, you may need to tune the agent's parameters or provide more training episodes.

A good balance in the exploration vs. exploitation ratio is important for learning. If the agent is exploring too much, you may need to reduce ϵϵ to encourage more exploitation. Conversely, if the agent isn't exploring enough, it could be stuck in a suboptimal path and missing better strategies.

## Conclusion

In this walkthrough, you applied several key evaluation metrics—cumulative reward, episode length, success rate, and the exploration vs. exploitation ratio—to assess the performance of your Q-learning agent. By visualizing these metrics, you can get a clear sense of how well your agent is learning and what aspects may need further tuning or improvement. Experimenting with these metrics will allow you to better understand and optimize your RL models in various environments.

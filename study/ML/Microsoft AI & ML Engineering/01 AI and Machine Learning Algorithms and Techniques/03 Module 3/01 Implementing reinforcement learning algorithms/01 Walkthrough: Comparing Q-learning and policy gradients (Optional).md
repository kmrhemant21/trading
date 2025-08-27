# Walkthrough: Comparing Q-learning and policy gradients (Optional)

## Introduction

In this walkthrough, we will go through the correct implementation of two reinforcement learning algorithms—Q-learning and policy gradients—as well as provide a detailed explanation of their behavior and the results.

By the end of this walkthrough, you will be able to: 

- Describe how Q-learning and policy gradients compare in terms of performance, convergence, and overall effectiveness.

## Implement Q-learning

Q-learning is a value-based reinforcement learning algorithm that learns the optimal Q-values for state-action pairs through exploration and exploitation. With a solid understanding of Q-learning and its theoretical basis, let's move on to the practical implementation of the algorithm, starting with the initialization of the Q-table.

### Step-by-step guide:

#### Step 1: Initialize the Q-table

In this step, we initialized the Q-table to store the values for each possible state-action pair. The table had 25 rows (one for each state in a 5x5 grid) and 4 columns (one for each action: up, down, left, and right).

```python
import numpy as np

grid_size = 5
n_actions = 4

# Initialize Q-table with zeros
Q_table = np.zeros((grid_size * grid_size, n_actions))
```

#### Step 2: Define the reward matrix

The agent received a reward of +10 for reaching the goal (state 24), a penalty of –10 for falling into the pit (state 12), and –1 for every other action to encourage faster exploration of the grid.

```python
rewards = np.full((grid_size * grid_size,), -1)
rewards[24] = 10  # Goal state
rewards[12] = -10  # Pitfall state
```

#### Step 3: Implement the epsilon-greedy policy

To balance exploration and exploitation, we used an epsilon-greedy strategy. With probability ϵ, the agent chose a random action, and with probability 1−ϵ, it chose the best-known action based on the Q-values.

```python
def epsilon_greedy_action(Q_table, state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(0, n_actions)  # Explore
    else:
        return np.argmax(Q_table[state])  # Exploit
```

#### Step 4: Q-learning update rule

The Q-values were updated based on the Bellman equation:

Q(s,a)←Q(s,a)+α[r+γ max⁡a′Q(s′,a′)−Q(s,a)]

Where:

- Q(s, a) is the current Q-value for the state-action pair,
- s is the current state,
- a is the action taken in the current state,
- r is the immediate reward received for the current action,
- γ is the discount factor for future rewards,
- α is the learning rate,
- s′ is the new state, and
- a′ is the action that maximizes the future reward in the new state.

```python
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

for episode in range(1000):
    state = np.random.randint(0, grid_size * grid_size)  # Random start
    done = False
    while not done:
        action = epsilon_greedy_action(Q_table, state, epsilon)
        next_state = np.random.randint(0, grid_size * grid_size)  # Random next state
        reward = rewards[next_state]

        # Update Q-value using Bellman equation
        Q_table[state, action] = Q_table[state, action] + alpha * (reward + gamma * np.max(Q_table[next_state]) - Q_table[state, action])

        state = next_state
        if next_state == 24 or next_state == 12:
            done = True  # End episode if goal or pitfall is reached
```

## Policy gradient implementation

Policy gradients are a policy-based reinforcement learning method where the agent directly learns a policy by maximizing the probability of actions that lead to higher rewards.

### Step-by-step guide:

#### Step 1: Build the policy network

In this step, we built a neural network using TensorFlow to model the policy. The network took the current state as input and output action probabilities using a softmax activation function.

```python
import tensorflow as tf

n_states = grid_size * grid_size  # 25 states in the grid
n_actions = 4  # Four possible actions

model = tf.keras.Sequential([
    tf.keras.layers.Dense(24, activation='relu', input_shape=(n_states,)),
    tf.keras.layers.Dense(n_actions, activation='softmax')  # Output action probabilities
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
```

#### Step 2: Action selection

For each state, the agent selected an action based on the probabilities outputted by the policy network.

```python
def get_action(state):
    state_input = tf.one_hot(state, n_states)  # One-hot encode the state
    action_probs = model(state_input[np.newaxis, :])
    return np.random.choice(n_actions, p=action_probs.numpy()[0])
```

#### Step 3: Cumulative rewards calculation

To give more weight to actions that led to long-term success, we computed cumulative rewards for each time step during an episode.

```python
def compute_cumulative_rewards(rewards, gamma=0.99):
    cumulative_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        cumulative_rewards[t] = running_add
    return cumulative_rewards
```

#### Step 4: Update policy using REINFORCE

We used the REINFORCE algorithm to update the policy. The loss function was the negative log-probability of the actions taken, scaled by the cumulative rewards.

```python
def update_policy(states, actions, rewards):
    cumulative_rewards = compute_cumulative_rewards(rewards)

    with tf.GradientTape() as tape:
        state_inputs = tf.one_hot(states, n_states)
        action_probs = model(state_inputs)
        action_masks = tf.one_hot(actions, n_actions)

        # Log-probabilities of the actions taken
        log_probs = tf.reduce_sum(action_masks * tf.math.log(action_probs), axis=1)

        # Policy loss function
        loss = -tf.reduce_mean(log_probs * cumulative_rewards)

    # Apply gradients to update the policy network
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

## Comparing Q-learning and policy gradients

In reinforcement learning, Q-learning and policy gradients are two popular approaches for training agents to make optimal decisions. While both methods aim to maximize cumulative rewards, they differ in their underlying mechanisms and are suited for different types of problems. Compare the methods below.

### Speed of convergence

Q-learning tended to converge faster in this small grid environment. This is because Q-learning works well in environments with a discrete action space and fewer states, allowing the agent to build a reliable Q-table quickly.

Policy gradients required more episodes to stabilize because the agent learned the policy directly through gradient updates. However, policy gradients are more flexible in environments with continuous action spaces.

### Reward maximization

Both algorithms eventually reached the goal consistently after enough episodes. However, Q-learning was more consistent in terms of reward maximization early on due to its more structured exploration.

Policy gradients started slowly but eventually caught up and produced comparable results.

### Exploration vs. exploitation

Q-learning relies heavily on exploration through the epsilon-greedy policy. The agent systematically explored different paths, but it risked getting stuck in suboptimal actions when epsilon was too high.

Policy gradients did not explicitly balance exploration and exploitation; instead, they optimized the policy based on cumulative rewards, which naturally led to better action selection as the policy improved.

### Suitability for different problems

Q-learning is more suited to environments with a small number of discrete actions and states, such as grid-based games or simple navigation tasks.

Policy gradients are better suited for environments with a continuous action space or more complex scenarios where approximating a value function (such as Q-values) becomes difficult.

## Conclusion

In this walkthrough, you implemented and compared Q-learning and policy gradient algorithms in a simple grid environment. Both approaches demonstrated their strengths: Q-learning's faster convergence and policy gradients' flexibility. While Q-learning is easier to implement in smaller, discrete environments, policy gradients offer a more scalable solution for larger, continuous action spaces.

By comparing both algorithms, you now have a deeper understanding of when to use each approach based on the complexity of the problem and the type of action space.

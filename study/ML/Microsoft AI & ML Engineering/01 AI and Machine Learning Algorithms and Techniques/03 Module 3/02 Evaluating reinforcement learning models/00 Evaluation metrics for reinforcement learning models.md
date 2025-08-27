# Evaluation metrics for reinforcement learning models

## Introduction
Evaluating reinforcement learning (RL) models is essential to understanding how well an agent is learning and performing in a given environment. Unlike traditional supervised learning, where performance is measured by loss functions or accuracy, RL evaluation involves understanding both short-term and long-term decision-making.

> By the end of this lesson, you will be able to: 
>
> Describe the most commonly used evaluation metrics in RL, helping you assess the effectiveness and efficiency of an RL agent.

## Cumulative reward 

### Definition
Cumulative reward is the total sum of rewards an agent collects during an episode (or across multiple episodes). This metric measures how well an agent performs based on the reward system defined in the environment.

### Why it's important
Cumulative reward gives a direct measure of an agent's overall performance, indicating whether it is improving over time or exploiting suboptimal actions.

### Key consideration
For environments where future rewards are more important, you can compute cumulative rewards using discounting with a factor γ (discount factor).

### Example
If an agent receives rewards of +1, –1, +2, and 10 over an episode, its cumulative reward for that episode is 12. Discounting may reduce future rewards by γ, affecting their weight in the final total.

## Average reward per episode 

### Definition
Average reward per episode is the cumulative reward an agent receives divided by the number of episodes. This metric helps smooth out performance and identify overall trends over time.

### Why it's important
It smooths out performance fluctuations and gives a better picture of the agent's learning progress. Average rewards provide insight into whether the agent is becoming more consistent over time.

### Key consideration
Early fluctuations can skew average reward measurements, so it is important to assess average rewards over a long series of episodes to see trends.

### Example
If the agent's cumulative rewards over 5 episodes are 10, 12, 8, 9, and 11, the average reward per episode is (10 + 12 + 8 + 9 + 11) / 5 = 10. 

## Episode length 

### Definition
Episode length refers to the number of steps the agent takes to complete an episode (e.g., reaching a goal or failing). In some environments, shorter episode lengths may indicate that the agent is learning to reach the goal faster.

### Why it's important
A decrease in episode length over time may suggest that the agent is learning to achieve the desired outcome more efficiently. It's a useful metric in tasks where the goal is to minimize the time or steps necessary to achieve success.

### Key consideration 
Shorter episode lengths aren't always better if they result from the agent terminating early due to failures.

### Example
If an agent solves a maze in 50 steps in episode 1, but later reduces the number of steps to 30 in episode 100, this indicates learning progress in minimizing the time to reach the goal.

## Time to convergence 

### Definition
This metric measures the number of episodes or steps it takes for the agent to reach a stable policy (i.e., stop significantly improving its performance). An agent has reached convergence when its cumulative reward, actions, and performance stabilize.

### Why it's important
Time to convergence indicates how quickly an RL agent learns an optimal (or near-optimal) policy. A lower time to convergence is desirable, particularly in environments where training time is costly or computationally expensive.

### Key consideration
Ensure that the environment is not too easy or too deterministic, as this could result in artificially fast convergence without true learning.

### Example
In a 10 x 10 grid environment, if the agent's cumulative reward plateaus after 500 episodes and remains stable, we say the agent has converged after 500 episodes.

## Policy stability 

### Definition
Policy stability measures how often the agent changes its learned policy (i.e., the set of actions it takes in various states) after reaching convergence. It indicates how confident the agent is in its learned actions.

### Why it's important
A highly stable policy means that the agent has learned a consistent set of actions that maximize rewards, whereas instability may indicate that the agent is still exploring or that the environment is dynamic.

### Key consideration
In non-stationary environments (where the environment changes over time), a highly stable policy may not be ideal, as the agent needs to adapt.

### Example
After convergence, if the agent frequently changes actions in the same state, this may indicate uncertainty or noise in the policy, suggesting the need for further training or policy refinement.

## Exploration vs. exploitation ratio 

### Definition
This metric measures the balance between exploration (trying new actions to discover their outcomes) and exploitation (choosing known actions that yield high rewards). The calculation often depends on how frequently the agent explores compared to when it exploits.

### Why it's important 
A well-balanced exploration vs. exploitation ratio ensures that the agent discovers optimal strategies without getting stuck in suboptimal actions.

### Key consideration
Too much exploration can slow down learning, while too much exploitation can prevent the agent from finding better solutions.

### Example
In a Q-learning algorithm, if the agent follows an ϵϵ-greedy strategy with ϵ = 0.1, it explores 10 percent of the time and exploits 90 percent of the time. A too-low ϵ might cause the agent to miss potentially better strategies.

## Success rate 

### Definition
The success rate measures how often an agent completes a task or reaches a goal within a set number of episodes or steps. It's a ratio of successful episodes to total episodes.

### Why it's important
In many tasks, such as games or robotic control, the success rate is the most direct way to measure the agent's ability to achieve the desired outcome.

### Key consideration
The use of success rate often occurs in combination with other metrics to provide a more comprehensive view of performance, especially in environments where completing the task quickly is as important as completing it at all.

### Example
If an agent completes the task in 80 out of 100 episodes, the success rate is 80 percent. 

## Sample efficiency 

### Definition
Sample efficiency refers to how effectively an agent uses its experiences (state-action-reward tuples) to learn. High sample efficiency means that the agent learns well from relatively few episodes.

### Why it's important 
Sample efficiency is crucial in environments where collecting data or simulating episodes is expensive or time-consuming (e.g., real-world robotics or complex simulations).

### Key consideration 
Algorithms that prioritize sample efficiency tend to converge faster but may require more sophisticated methods such as experience replay or off-policy learning.

### Example
An agent that reaches an optimal policy after 500 episodes is more sample-efficient than one that requires 10,000 episodes to achieve the same performance.

## Computational complexity 

### Definition
Computational complexity measures the required time and resources to run an RL algorithm. This metric helps determine whether an algorithm is feasible for large-scale or real-time applications.

### Why it's important
In real-world systems, limited computational power or time constraints can affect the choice of RL algorithms. Efficient algorithms are necessary for applications such as autonomous vehicles or robotic systems.

### Key consideration
Consider both the time complexity (how long the algorithm takes to converge) and space complexity (memory usage for storing value functions or policies).

### Example
A model-free algorithm such as Q-learning may require more time to converge than a model-based approach but uses fewer resources since it does not need to model the environment explicitly.

## Conclusion
Evaluating RL models requires a set of tailored metrics that go beyond traditional ML evaluations. Cumulative rewards, episode length, policy stability, and other metrics provide insight into how well an agent is learning, exploring, and exploiting its environment. By understanding and applying these metrics, you can better assess the performance and effectiveness of your RL models and make informed decisions about algorithm selection and training strategies.
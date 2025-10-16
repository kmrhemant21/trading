# Practice activity: Multi-agent systems vs. single agent systems

## Introduction
In this practice activity, we will explore the differences between multiagent systems and single agent systems by comparing how each handles tasks, problem-solving, and interactions with the environment. The goal is to help you understand the advantages and limitations of each type and when it's more appropriate to use one over the other.

By the end of this activity, you will be able to:

- Identify key differences between single agent and multiagent systems.
- Explain the advantages and challenges of both types of systems.
- Analyze specific use cases to determine when to apply a multiagent approach versus a single agent approach.

## Key concepts
Explore the following key concepts:

- Single agent systems
- Multiagent systems

## 1. Single agent systems
A single agent system consists of one autonomous agent interacting with an environment to achieve a specific goal. The agent operates independently, gathering information, making decisions, and taking actions based solely on its perception of the environment.

### Characteristics of single agent systems
- **Simplicity**: only one agent is responsible for the task at hand, which can make system design and control simpler.
- **Limited perception and control**: the agent has access only to the information it can perceive, and it controls only its own actions.
- **Efficiency**: single agent systems can be efficient for tasks that do not require complex coordination or multiple perspectives.

### Example use case
A Roomba vacuum cleaner is a single agent system. It navigates a room, avoiding obstacles and cleaning the space based on its own sensors. It doesn't interact with other agents but instead operates autonomously.

## 2. Multiagent systems
A multiagent system consists of multiple agents that may interact with one another and the environment. These agents can either cooperate or compete to achieve their goals, depending on the system's design. Multiagent systems are used for complex tasks that require collaboration or the simultaneous efforts of multiple autonomous entities.

### Characteristics of multiagent systems
- **Decentralization**: control is distributed among multiple agents, each of which operates independently. You'll recall that centralization refers to a control or decision-making structure where a single agent or central authority coordinates and dictates the actions of all agents in the system.
- **Collaboration or competition**: agents can work together to solve problems or compete to achieve their own goals.
- **Scalability**: multiagent systems can handle larger, more complex problems that a single agent system would struggle with.
- **Emergent behavior**: the system can exhibit behavior that arises from the interactions between agents, often leading to more efficient solutions.

### Example use case
An example of a multiagent system is a fleet of autonomous delivery drones, each tasked with delivering packages in different regions of a city. The drones coordinate their routes and share traffic and weather data to optimize delivery times without overlapping.

## Practice activity: Multiagent vs. single agent systems
For this practice activity, consider the following five scenarios. Based on what you have learned, determine whether a multiagent system or a single agent system would be more suitable for each case. After making your decision, justify your choice by explaining how the system's characteristics align with the task at hand. You should take notes and compile a reflection at the end of the activity.

### Scenario 1: Warehouse management system
You are tasked with designing a system for managing a warehouse where inventory needs to be picked and moved to different storage locations. Multiple areas of the warehouse need to be covered, and tasks must be completed quickly and efficiently, especially during peak hours when demand is high.

**Question**: Would you design a multiagent system or a single agent system for this task? Why?

#### Considerations
- Does the task require coordination between agents to optimize performance?
- Could a single agent manage the entire warehouse efficiently, or would it be overloaded?

### Scenario 2: Autonomous lawn mower
A residential property owner wants to automate the mowing of their lawn. The lawn is medium-sized, and there are a few obstacles, such as trees and flower beds. The lawn needs to be maintained weekly, and the owner is looking for a solution that can operate autonomously with minimal oversight.

**Question**: Would a multiagent or single agent system be more appropriate for this task? Why?

#### Considerations
- Is the task simple enough for a single agent to handle?
- Would adding more agents improve efficiency, or would it lead to unnecessary complexity?

### Scenario 3: Search-and-rescue operation
A disaster has occurred in a large urban area, and a team of drones is deployed to search for survivors across different parts of the city. The search area is vast and filled with obstacles such as collapsed buildings, making communication and coordination between the drones crucial for covering the entire area effectively.

**Question**: Which system would you recommend: multiagent or single agent? Why?

#### Considerations
- Is the task too large and complex for a single agent to handle alone?
- How would multiple agents work together to ensure full coverage without missing key areas?

### Scenario 4: Customer service chatbot
A company wants to implement an AI-powered chatbot to assist customers with answering common questions about their products. The chatbot will be integrated into the company's website and will need to handle multiple customer queries simultaneously.

**Question**: Would a multiagent or single agent system be best for this task? Why?

#### Considerations
- Does the chatbot need to manage multiple interactions simultaneously?
- Would introducing multiple agents improve response time and service quality?

### Scenario 5: Autonomous traffic management
You are tasked with designing an AI system for managing traffic flow in a large city. The system needs to monitor traffic at different intersections, optimize the timing of traffic lights, and adapt to changes in real time, such as accidents or congestion. The goal is to reduce traffic jams and improve the overall flow of vehicles.

**Question**: Should this system be based on a multiagent system or a single agent system? Why?

#### Considerations
- Is the traffic management task too complex for a single agent to handle?
- How would agents collaborate to optimize traffic across the entire city?

## Reflection
After completing the scenarios, reflect on your answers. Consider the following questions:

- What are the key advantages of using a multiagent versus a single agent system?
- In which situations might the simplicity of a single agent system outweigh the potential benefits of a multiagent system?
- Can you think of other real-world applications in which one type of system would be more suitable than the other?

## Conclusion
Through this practice activity, you have explored various scenarios that demonstrate the strengths and weaknesses of multiagent systems compared to single agent systems. Understanding the differences between these systems will help you choose the best approach when designing AI solutions for different tasks and environments.

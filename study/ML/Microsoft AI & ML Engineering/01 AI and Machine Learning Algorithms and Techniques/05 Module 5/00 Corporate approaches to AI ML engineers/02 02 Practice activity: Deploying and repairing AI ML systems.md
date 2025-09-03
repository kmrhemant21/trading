# Practice activity: Deploying and repairing AI/ML systems

## Introduction

In this activity, you will be given an unresolved case study related to a real-world AI/ML implementation. Your task is to think through the deployment strategy and explore potential repair solutions using AI/ML systems engineering approaches. This activity aims to enhance your critical thinking and decision-making skills when it comes to deploying and maintaining AI/ML systems.

By the end of this activity, you will be able to: 

- Identify key challenges in AI/ML implementations.
- Propose effective AI/ML solutions.
- Develop a deployment strategy.
- Create a maintenance plan for AI/ML systems.

## Case study: Predictive maintenance for autonomous delivery drones

### Scenario

A logistics company uses a fleet of autonomous delivery drones to transport packages across urban areas. Recently, the company has experienced several unexpected drone failures, causing delays and increasing costs. While the drones are equipped with various sensors (e.g., battery, GPS, temperature, mechanical parts), the company is still using a scheduled maintenance system, which has proven ineffective at preventing these failures. The company wants to implement an AI/ML-powered predictive maintenance system to forecast potential drone failures and improve reliability.

### Step-by-step instructions:

#### Step 1: Problem identification

Analyze the case study to identify the key issues and challenges. Answer the following questions:

**What are the main problems?**

- The drones are experiencing unexpected failures, resulting in costly downtime.
- The current maintenance system is based on fixed schedules, which does not account for the actual condition of the drones.

**What data is available?**

- Sensor data from the drones (battery levels, GPS data, temperature, flight hours, motor speeds, mechanical status, etc.).
- Historical failure data and maintenance logs.

**What are the system's requirements?**

- Real-time monitoring of drone health.
- Early failure detection to allow for proactive maintenance.
- Integration with the drone fleet management system to trigger alerts or maintenance requests.

#### Step 2: Propose AI/ML solutions

Based on the problems identified, think through different AI/ML techniques that could be applied to solve the problem.

**Model selection:**

- Consider using time-series analysis or anomaly detection for sensor data to identify early signs of failure.
- Regression models could be used to predict the remaining useful life (RUL) of key drone components (e.g., motors, batteries).
- Classification models could classify whether a drone is at risk of failure based on sensor data.

**Data preprocessing:**

- Clean and preprocess the sensor data (normalize values, handle missing data).
- Perform feature engineering to create meaningful features from raw sensor data, such as the rate of battery discharge or temperature changes over time.

**Model training:**

- Train the model on historical data to learn patterns associated with drone failures.
- Use evaluation metrics such as accuracy, precision, and recall for classification tasks, or mean squared error and R-squared for regression tasks.

#### Step 3: Deployment strategy

Now that you have a potential AI/ML solution, think through the deployment process.

**Deployment environment:**

- The AI/ML model could be deployed on the cloud to handle the large volume of data from multiple drones. Alternatively, edge computing could be used to allow for real-time analysis on the drones themselves, reducing latency.
- Ensure the model is capable of processing real-time sensor data streams.

**Integration:**

- Integrate the predictive maintenance system with the existing drone management software. The system should trigger alerts when the model detects a potential failure or anomaly.
- Provide maintenance personnel with real-time dashboards that display drone health and predict failures.

**Monitoring and updates:**

- Implement a monitoring system to track the model's performance after deployment.
- Set up automated processes for model retraining as new data is collected to ensure that the model remains accurate over time.

#### Step 4: Repair and maintenance planning

Consider how you would maintain and adjust the AI/ML system over time.

**Error handling:**

- What steps would you take if the model produces too many false positives or false negatives? You may need to adjust the model's threshold for triggering alerts or retrain the model with additional data.

**Model drift:**

- As the drones age or new drone models are introduced, the model's predictions may degrade over time due to model drift. Regular retraining with new data will be necessary to ensure accuracy.

**Scaling the system:**

- If the company expands its drone fleet, consider how to scale the system to handle more data and ensure efficient predictions for larger numbers of drones.

## Deliverables

Please create the following after completing the activity:

### Solution report:

Write a report (300–400 words) outlining:

- The key problems identified in the case study.
- The proposed AI/ML solution, including model selection and preprocessing steps.
- The deployment strategy, including environment, integration, and monitoring.
- A plan for addressing system errors, retraining, and scaling the solution.

### Code implementation (optional):

If you wish to go deeper, you can implement a prototype of the predictive maintenance system using a sample dataset. This could include:

- Preprocessing drone sensor data.
- Implementing a time-series analysis or regression model to predict failures.
- Visualizing model predictions and drone health.

## Conclusion

This hands-on activity provides a real-world application of AI/ML systems in an operational setting. By working through the deployment and maintenance of a predictive maintenance system, you gain valuable experience in problem-solving, model deployment, and systems engineering. Make sure to consider both technical and operational challenges when crafting your solutions.

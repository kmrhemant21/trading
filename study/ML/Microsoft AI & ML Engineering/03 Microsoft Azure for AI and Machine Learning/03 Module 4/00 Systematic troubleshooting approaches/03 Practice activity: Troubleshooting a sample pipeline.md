# Practice activity: Troubleshooting a sample pipeline

## Introduction
In this hands-on activity, you will troubleshoot a sample machine learning pipeline using a troubleshooting agent. 

> Note: you may refer to the troubleshooting techniques and tools covered in previous courses to guide your process. Additionally, ensure you have Python 3.8+ installed and the required libraries (ml_pipeline and troubleshooting_agent) set up in your environment.

Troubleshooting is a crucial skill for identifying issues in model deployment and maintaining reliable systems. You will use the agent to debug different stages of the pipeline systematically, identify root causes of common issues, and apply appropriate fixes.

By the end of this activity, you will be able to:
- Apply different troubleshooting techniques to diagnose and fix issues in a machine learning pipeline.

## Step-by-step guide to troubleshoot a sample pipeline
This reading will guide you through the following steps:

1. Step 1: examine pipeline logs
2. Step 2: validate data integrity
3. Step 3: debug model training code
4. Step 4: review resource allocation
5. Step 5: verify deployment settings
6. Step 6: test in a staging environment
7. Step 7: implement logging and monitoring

### Step 1: Examine pipeline logs
Logs are your first clue to identifying the problem.

Azure Machine Learning Studio provides detailed logging for each pipeline step. Focus on errors or warnings to pinpoint the issue. Read and understand this example code: 

```python
# Example of accessing logs from Azure ML pipeline
from azureml.core import Workspace, Experiment

# Connect to your workspace
ws = Workspace.from_config()

# Create example experiment
experiment_name = 'sample_pipeline' #replace with your pipeline name
experiment = Experiment(ws, experiment_name)

# Access the run details
for run in experiment.get_runs():
    print(f"Run ID: {run.id}, Status: {run.status}")
    print(run.get_details())
```

### Step 2: Validate data integrity
Ensure the data used at each stage meets expected formats and values.

Use validation scripts to identify missing or corrupted data before retrying the pipeline:

```python
import pandas as pd

# Create and save a sample dataset
data = {
    "customer_id": [1, 2, 3, 4, 5],
    "membership_level": ["Bronze", "Silver", "Gold", "Silver", "Bronze"]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
csv_filename = "customer_data.csv"
df.to_csv(csv_filename, index=False)

# Load the dataset
incoming_data = pd.read_csv(csv_filename)

# Validate the data
def validate_data(df):
    # Check for null values in 'customer_id'
    if df['customer_id'].isnull().any():
        print("Validation Error: 'customer_id' column contains null values.")
    else:
        print("No null values in 'customer_id' column.")

    # Check that 'membership_level' contains only allowed values
    allowed_values = {"Bronze", "Silver", "Gold"}
    invalid_values = set(df['membership_level']) - allowed_values
    if invalid_values:
        print(f"Validation Error: 'membership_level' contains invalid values: {invalid_values}")
    else:
        print("All values in 'membership_level' are valid.")

# Run validation
validate_data(incoming_data)
```

### Step 3: Debug model training code
Run the training code step-by-step in a Jupyter Notebook to isolate issues:

```python
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a LinearRegression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output the results
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")
```

### Step 4: Review resource allocation
Check resource usage in Jupyter Notebooks:

- Ensure sufficient memory and CPU are allocated.
- Upgrade your compute instance or expand your compute cluster if necessary.

### Step 5: Verify deployment settings
Ensure the deployment configuration includes all necessary dependencies and that the model is containerized correctly. 

This is just example code; a real-world deployment may look different. Read and understand this code, but don't run it: 

```python
# Example of deploying a model to Azure Kubernetes Service (AKS) 
from azureml.core import Workspace, Model 
from azureml.core.webservice import AksWebservice, AksCompute 
from azureml.core.model import InferenceConfig 
 
# Connect to your workspace 
ws = Workspace.from_config() 
 
# Load the model 
model = Model(ws, 'my_model') 
 
# Define inference configuration 
inference_config = InferenceConfig(entry_script='score.py', environment='myenv') 
 
# Define deployment configuration 
aks_config = AksWebservice.deploy_configuration(cpu_cores=1, memory_gb=1) 
 
# Deploy the model 
service = Model.deploy(workspace=ws, name='my-aks-service', models=[model], 
                   	inference_config=inference_config, deployment_config=aks_config, 
                   	deployment_target=AksCompute(ws, 'aks-compute')) 
service.wait_for_deployment (show_output=True)
```

### Step 6: Test in a staging environment
Another step to take in real-world troubleshooting is testing in a staging environment to identify issues before they're pushed to production.  

- Validate changes in a staging environment before applying them to production.
- Azure ML Studio allows you to create a staging endpoint to test updates safely.

### Step 7: Implement logging and monitoring
Add logging at each step to provide detailed insights for future troubleshooting:

```python
import logging

# Configure logging settings
logging.basicConfig(filename='pipeline_logs.log', level=logging.INFO)

# Log pipeline events
logging.info("Pipeline step completed successfully.")
```
Integrate tools such as Azure Application Insights for monitoring pipeline performance.

## Real-world scenario
Imagine that you are working with a pipeline used for predicting customer churn. After deployment, you notice that the model's predictions are inconsistent. By using the troubleshooting agent, you identify that model drift has occurred due to a change in customer behavior patterns. You retrain the model with updated data and apply performance profiling to reduce latency, ensuring that the updated pipeline provides accurate and timely predictions.

## Conclusion
In this activity, you learned:

- How to identify and diagnose pipeline issues using logs.
- Techniques to validate data integrity.
- Debugging strategies for model training.
- Resource optimization methods to prevent allocation issues.
- Steps for verifying deployment configurations.
- The importance of testing in a staging environment.
- How to implement robust logging and monitoring systems.

By applying these strategies, you can ensure the reliability and efficiency of your machine learning pipelines, paving the way for successful deployments and sustained performance.

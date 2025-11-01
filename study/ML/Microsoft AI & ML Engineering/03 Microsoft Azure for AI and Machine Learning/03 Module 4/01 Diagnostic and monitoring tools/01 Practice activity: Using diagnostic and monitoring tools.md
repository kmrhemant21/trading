# Practice activity: Using diagnostic and monitoring tools

## Introduction
Have you ever struggled to identify the root cause of an issue in your machine learning pipeline, only to end up with more questions than answers? Diagnostic and monitoring tools can provide the clarity needed to identify, troubleshoot, and resolve these issues. This guide demonstrates how to set up and use these tools to enhance model performance and reliability.

By the end of this activity, you will be able to:

- Set up a diagnostic or monitoring tool to troubleshoot a machine learning pipeline.
- Use the selected tool to identify and resolve data quality, performance, or integration issues.
- Gain practical knowledge of implementing monitoring and diagnostic tools for production-level pipelines.

## Step-by-step guide to using diagnostic and monitoring tools
This reading will guide you through the following steps:

1. Step 1: Validate data quality with Pandas.
2. Step 2: Monitor performance with Azure Application Insights.
3. Step 3: Document and resolve issues.
4. Step 4: Apply incremental deployment strategies.

### Step 1: Validate data quality with Pandas
Use Pandas to check for missing or invalid values in your dataset:

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

### Step 2: Monitor performance with Azure Application Insights
Set up metrics collection for the deployed machine learning model to track performance metrics such as latency, throughput, and resource utilization. Don't run this code, only read it and understand it. If you want to run it, you'll need to replace {service_name} with the name of a deployed service in your workspace. 

```python
from azureml.core import Workspace

# Connect to Azure ML Workspace
workspace = Workspace.from_config()

# Access the deployed service
service_name = 'my-model-service'
service = workspace.webservices[service_name]

# Enable Application Insights if not already enabled
if not service.app_insights_enabled:
    service.update(enable_app_insights=True)
    print(f"Application Insights enabled for service: {service_name}")
else:
    print(f"Application Insights is already enabled for service: {service_name}")

# Check the Application Insights link
print(f"Application Insights URL: {service.scoring_uri}")
```

### Step 3: Document and resolve issues
Maintain a detailed record of issues detected during diagnostics:

Document issues: missing values, high latency, API integration errors.

Log details: include identified issues, steps taken to address them, and results of those actions.

```
Issues Identified:
- Missing values in 'customer_id'.
- High latency during predictions.

Steps Taken:
- Implemented data validation to ensure completeness.
- Upgraded compute resources to reduce latency.

Results:
- Validation passed without errors.
- Latency reduced by 30%.
```

Validate that fixes have been implemented successfully by rerunning diagnostics and confirming that no new issues arise.

### Step 4: Apply incremental deployment strategies 
Use canary deployment to release changes to a subset of users and monitor their impact. This is only example code; read it and understand it, but do not run it:

```python
# Apply canary deployment to a limited number of users
# This is pseudocode; it shows the business logic, but isn’t a full implementation
canary_deployment_successful = service.canary_deploy()
if canary_deployment_successful:
    print("Canary deployment successful. Proceeding to full deployment.")
else:
    print("Canary deployment failed. Investigate issues before full deployment.")
```
Monitor the changes during deployment, and proceed to full deployment only if no issues are detected.

## Conclusion
In this activity, you learned how to:

- Set up diagnostic and monitoring tools to manage machine learning pipelines effectively.
- Identify and resolve common issues such as data quality and integration errors.
- Leverage proactive monitoring to improve model performance and reliability.
- Apply deployment strategies such as canary deployment to minimize risk.

By applying these techniques, you can ensure the reliability and efficiency of your machine learning models in production environments.

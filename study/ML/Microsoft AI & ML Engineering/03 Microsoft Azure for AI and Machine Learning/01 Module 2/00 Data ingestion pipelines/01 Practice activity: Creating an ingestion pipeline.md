# Practice activity: Creating an ingestion pipeline

## Introduction
Imagine you're working for a health care company that collects patient data from multiple clinics. Pipelines in Microsoft Azure may be easier for model training purposes than wrangling complex code, making them a perfect solution for newer or less experienced AI/ML engineers. Consolidating, cleaning, and storing that data for analysis can be daunting without a streamlined process. This hands-on walkthrough will demonstrate the process of creating an ingestion pipeline step-by-step using Microsoft Azure services to extract, transform, and load data efficiently. Follow along to gain practical experience and observe best practices in action.

By the end of this activity, you will be able to:

- Set up a pipeline.
- Implement data extraction, transformation, and loading processes.
- Use Azure monitoring tools to ensure data reliability.

## Step-by-step guide to building an ingestion pipeline
This reading will guide you through the following steps:

1. Step 1: Access your workspace.
2. Step 2: Create a new pipeline.
3. Step 3: Explore the pipeline.
4. Step 4: Run the pipeline.
5. Step 5: Evaluate the pipeline results.

### Step 1: Access your workspace
- Navigate to [https://ml.azure.com](https://ml.azure.com) and sign in if prompted.
- Select your workspace and click to enter it.

### Step 2: Create a new pipeline
- Under Assets, navigate to Pipelines and select New Pipeline.
- Begin by creating a basic regression pipeline for demonstration purposes.
- Click to create the regression pipeline.
- Pipelines in Microsoft Azure are used to transform raw data into trained models and evaluate those models.
- The pipeline contains blocks chained together to process raw automobile price data into an evaluated model.

### Step 3: Explore the pipeline
Double-click on blocks to view more details:

- Data ingestion: consumes a dataset URI such as "azureml.data.automobile.price.raw.versions.3."
- Model training: shows run settings, compute target, and component information.
- Evaluation: uses testing data to assess the model's performance.

Notice that the pipeline uses a linear regression model, combining training data from a split dataset and evaluating it.

### Step 4: Run the pipeline
Click Configure and Submit.

Set up the experiment:
- Select the button Create New.
- Experiment name: "car-price-data-pipeline."
- Leave the job display name as is.
- Click Review and Submit.

Resolve any errors:
- If the Submit button is disabled, check the runtime settings.
- Select a compute instance already running in your workspace.
- Once configured, click Submit. A notification will confirm submission.

### Step 5: Evaluate the pipeline results
View the details of the pipeline job upon completion.

Navigate to the Evaluate Model block and double-click it:
- Select Output and Logs to review the logs.
- Select Metrics to assess performance metrics such as:
    - Coefficient of determination
    - Mean absolute error
    - Relative absolute error

**Coefficient of determination (R²)** 

$$R^2 = 1 - \frac{SS_{residual}}{SS_{total}}$$

where $SS_{residual}$ is the sum of squared differences between actual and predicted values, and $SS_{total}$ is the sum of squared differences between actual values and their mean. 

**Mean absolute error (MAE)** 

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

where $y_i$ represents the actual values, $\hat{y}_i$ represents the predicted values, and $n$ is the number of data points. 

**Relative absolute error (RAE)** 

$$RAE = \frac{\sum_{i=1}^{n} |y_i - \hat{y}_i|}{\sum_{i=1}^{n} |y_i - \bar{y}|}$$
where yi  represents the actual values, ŷi  represents the predicted values, and ӯ  is the mean of the actual values. 

## Real-world scenario
Consider a health care company. A pipeline can:

- Centralize data: consolidate data from multiple clinics into a single system.
- Transform data: clean and preprocess patient data for analysis.
- Load data into models: train machine learning models for predictive insights. 

This streamlined approach improves accessibility, supports timely decision-making, and enhances patient outcomes.

## Conclusion
In this activity, you learned how to:

- Build a strong data ingestion pipeline using Azure services.
- Connect to various sources securely, handle sensitive data, and preprocess it for analysis.
- Train and evaluate a regression model using the pipeline.

Now you're ready to set up your own data pipeline using Azure tools. Experiment with different data sources and models to create a flexible and scalable solution. This is a critical skill for managing real-world machine learning projects.

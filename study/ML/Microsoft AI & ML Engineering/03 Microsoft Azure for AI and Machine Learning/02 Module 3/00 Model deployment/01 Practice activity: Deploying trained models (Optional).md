# Practice activity: Deploying trained models (Optional)

## Introduction
Imagine you've developed a model capable of making impactful predictions or optimizing critical decisions. However, the true power of your model lies in its deployment—ensuring that it's accessible, scalable, and ready to deliver real-time insights.

In this activity, we'll guide you through transforming your trained model into a fully operational service on Azure Machine Learning, so it's ready to provide tangible value to end-users and applications.

By the end of this activity, you will be able to:

- Configure and navigate the Azure Machine Learning workspace to manage and deploy machine learning models.
- Deploy a trained model as a scalable service by creating an inference configuration and selecting appropriate Azure compute resources.
- Test and validate the deployed model endpoint, ensuring reliable predictions for real-world applications.

## Step-by-step guide to model deployment
This reading will guide you through the following steps:

1. Step 1: Access your workspace.
2. Step 2: Open the deployment example.
3. Step 3: Initialize your environment.
4. Step 4: Train and register your model.
5. Step 5: Set up a custom environment.
6. Step 6: Deploy the model.
7. Step 7: Clean up deployment.

### Step 1: Access your workspace
- Navigate to `https://ml.azure.com` and sign in if prompted.
- Enter your workspace.

### Step 2: Open the deployment example
- Go to Notebooks under the authoring section.
- Switch to the Samples tab of the File Explorer.
- Navigate to SDK v1 > how-to-use-azureml > deployment > deploy-to-cloud.
- Open the "model-register-and-deploy.ipynb" notebook.
- Clone this file and its associated files into your workspace under a folder named "Cloned From samples" by selecting "Clone this notebook."

### Step 3: Initialize your environment
- Ensure you are using the Python 3.8 Azure ML kernel.
- Run the first cell to import AzureML.core and verify the correct SDK version.
- Initialize your workspace by running the provided code cell to create a workspace object from your existing configuration.

### Step 4: Train and register your model
- Use Scikit-Learn to train a small model on the diabetes dataset:
    - Run the code to train the model.
    - Save and register input and output datasets by running the provided code.
- Register the trained model using the "model.register" function:
    - Include metadata such as description, tags, and framework information.
    - Example: "Ridge regression model to predict diabetes progression."

### Step 5: Set up a custom environment
- Avoid using the default environment to ensure compatibility and security.
- Create a custom environment with specific package versions, including:
    - "pip," "Azure Machine Learning Defaults," "Inferred Schema," "Joblib," and specific versions of "DIL," "NumPy," and "Scikit-Learn."
- Define the "score.py" script:
    - Include methods for initialization and running the model.
    - Example: "model.predict" is used to process input data and return predictions.

### Step 6: Deploy the model
- Use "inference.config" and "model.deploy" to deploy your model.
- This step may take several minutes. Monitor output logs for any errors or troubleshooting assistance.
- Upon successful deployment, test the service using sample input payloads.
- Run the provided code cell to submit input data and observe the output predictions. If you encounter an error, try adding "import json" at the beginning of the code cell.

### Step 7: Clean up deployment
- Use "service.delete" to remove the deployed service.
- Cleaning up ensures that unnecessary charges are avoided and resources are freed.

## Real-world scenario
Suppose you've built a recommendation model for an e-commerce platform to suggest products to customers in real-time. By deploying this model as an API you can ensure:

- **Scalability**: use a service such as Azure Kubernetes Service to handle high volumes of requests during peak traffic.
- **Monitoring**: set up alerts and monitoring to track performance and ensure reliability.
- **Proactive action**: retrain the model as necessary to address changing user behaviors or data drift.

## Conclusion
In this activity, you learned to:

- Configure the Azure Machine Learning workspace and navigate deployment resources.
- Train, register, and deploy a machine learning model using custom environments.
- Test and validate deployed models for reliable, real-world predictions.

Proper deployment practices maximize the value and performance of machine learning models. Take what you've learned here and try deploying a different model to Azure. Experiment with different compute targets, such as Azure Kubernetes Service, to gain experience managing large-scale deployments. Mastering these steps is essential for real-world machine learning success.

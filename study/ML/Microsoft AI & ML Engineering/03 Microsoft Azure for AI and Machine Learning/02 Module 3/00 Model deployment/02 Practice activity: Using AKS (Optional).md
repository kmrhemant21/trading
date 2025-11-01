# Practice activity: Using AKS (Optional)

## Introduction
Are you ready to scale your machine learning models seamlessly? Deploying models in a production environment doesn't have to be complicated. Azure Kubernetes Service provides a powerful, scalable solution to manage deployments with ease. Whether you're new to deploying machine learning models or looking to refine your skills, this activity will show you how to bring your models to life in real-world applications.

> **Note**: You will need 12 virtual CPUs to do this activity. The free Azure account only has 10. If you don't have enough virtual CPUs to complete this activity, feel free to skip it.

By the end of this activity, you will be able to:

* Set up an Azure Machine Learning workspace.
* Deploy models to Azure Kubernetes Service.
* Test deployed models for reliability and performance.

## Step-by-step guide to deploy models with Azure Kubernetes Service
This reading will guide you through the following steps:

1. Step 1: Access your workspace.
2. Step 2: Open the deployment notebook.
3. Step 3: Import and initialize.
4. Step 4: Register a model.
5. Step 5: Define the deployment environment.
6. Step 6: Write the entry script.
7. Step 7: Create inference configuration.
8. Step 8: Provision the Azure Kubernetes Service cluster.
9. Step 9: Deploy the model.
10. Step 10: Clean up resources.

### Step 1: Access your workspace
1. Go to [https://ml.azure.com](https://ml.azure.com) and sign in if prompted.
2. Enter your Azure Machine Learning workspace.

### Step 2: Open the deployment notebook
1. Navigate to Notebooks under the authoring section.
2. Switch to the Samples tab in the File Explorer.
3. Go to SDK v1 > how-to-use-azureml > deployment > production-deploy-to-aks.
4. Open the "production-deploy-to-aks.ipynb" notebook.
5. Clone the notebook and its dependencies to your workspace under a directory such as "dweaver."
6. Ensure that your compute instance is running and attach it.
7. Use the Python 3.8 Azure ML kernel for compatibility.

> **Note**: The deployment notebook provides a guided and structured way to deploy models to Azure Kubernetes Service, with preconfigured scripts that streamline the deployment process for ease of use.

### Step 3: Import and initialize
1. Import necessary libraries and verify the Azure Machine Learning SDK version by running the provided cells.
2. Load the workspace configuration:
3. Print workspace details (name, resource group, location, and subscription ID) to confirm proper setup.

> **Note**: Initializing your environment ensures that all dependencies and configurations are correctly set up, reducing potential errors during later steps.

### Step 4: Register a model
1. Register the trained model:
    * Example: "sklearn-regression-model.pkl" (ridge regression model for diabetes prediction).
    * Include metadata such as description and framework information during registration.

> **Note**: Model registration is a critical step to organize and track models in your workspace. This metadata helps to ensure consistency and traceability for production deployments.

### Step 5: Define the deployment environment
1. Create a custom environment:
    * Specify Conda dependencies such as "numpy," "scikit-learn," and "scipy."
    * Include Pip dependencies such as "Azure ML Defaults" and "Inference Schema."
    * Optionally, use a custom docker image for further customization.

> **Note**: Defining the environment ensures that all necessary libraries and dependencies are available for the model to function correctly in production.

### Step 6: Write the entry script
1. Create a "score.py" script:
    * Define the initialization method to load the registered model using JobLib.
    * Define the "run" method to:
      * Parse input data (JSON to NumPy array).
      * Predict using the model.
      * Return predictions in list form.
      * Handle exceptions effectively.

> **Note**: The entry script defines how your deployed model processes incoming requests, ensuring seamless integration with external applications or APIs.

### Step 7: Create inference configuration
1. Use the "InferenceConfig" class to specify the entry script and environment.
2. Run the provided cells to finalize the inference configuration.

> **Note**: The inference configuration links the model, entry script, and environment, providing a blueprint for deployment to Azure Kubernetes Services.

### Step 8: Provision the Azure Kubernetes Services cluster
Optional: skip provisioning if you already have an Azure Kubernetes Services cluster. If you do not have a cluster, you will not be able to complete Steps 8 and 9. Know that these steps are optional.

Note that Azure Kubernetes Services provisioning by default may require up to 12 virtual CPUs, potentially exceeding quotas.

In production, upgrade quotas or use existing clusters to avoid deployment issues.

Optional settings include deploying Azure Kubernetes Services in a virtual network and enabling SSL for secure communication.

> **Note**: Provisioning an Azure Kubernetes Services cluster provides a scalable and secure environment to handle high volumes of inference requests in production.

### Step 9: Deploy the model
Optional: skip deploying unless you already have an Azure Kubernetes Services cluster. If you do not have a cluster, you will not be able to complete Step 9. Know that this step is optional.

1. Use "akswebservice.deployConfiguration" and "model.deploy" to deploy the model to Azure Kubernetes Services.
2. Monitor the logs and wait for deployment to complete.
3. Test the deployed web service:
    * Send test input data to the cluster.
    * Validate predictions.

> **Note**: Deploying the model to Azure Kubernetes Services makes it accessible as a web service, enabling real-time predictions and integration with applications.

### Step 10: Clean up resources
1. Delete the deployed service and model using appropriate cleanup commands.
2. This ensures no unnecessary charges or resource usage.

> **Note**: Cleaning up resources prevents unnecessary costs and ensures efficient use of cloud infrastructure by releasing unused resources.

## Real-world scenario
Suppose that you are developing a customer support automation model for a large enterprise. Use these concepts:

* **Validation**: Deploy the model to Azure Container Instances for internal testing.
* **Scalability**: Transition to Azure Kubernetes Services for production-level deployment to handle high traffic.
* **Reliability**: Monitor performance metrics and set alerts for critical thresholds.

By leveraging Azure Container Instances for quick testing and Azure Kubernetes Services for large-scale production, you can ensure high availability and robust performance.

## Conclusion
In this activity, you learned how to:

* Set up an Azure Machine Learning workspace for deployment.
* Deploy models to Azure Kubernetes Service.
* Test and validate deployed models.

Azure Machine Learning and Azure Kubernetes Services provide the tools needed to manage deployments of any scale. Try deploying a model to both Azure Container Instances and Azure Kubernetes Services to see how different compute targets impact performance. Experiment with deployment configurations to gain deeper insights into optimizing your machine learning workflows for real-world success.

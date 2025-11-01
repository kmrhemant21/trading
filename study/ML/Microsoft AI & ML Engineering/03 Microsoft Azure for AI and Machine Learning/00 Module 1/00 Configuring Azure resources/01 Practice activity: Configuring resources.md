# Practice activity: Configuring resources

## Introduction
This hands-on activity will guide you through the process of configuring resources in Azure for AI/ML projects. You will learn how to create a compute instance and register a datastore in your Azure Machine Learning workspace. Completing this activity will give you practical experience in setting up and managing the key components needed for your machine learning workflows.

> Note: this activity is similar in nature to the workspace setup activity. If you have not yet tried the workspace setup activity, now is a good time to try it.

By the end of this activity, you will be able to:

- Create and configure a compute instance in Azure Machine Learning to execute ML tasks.
- Register and authenticate a datastore to manage and access data securely in your Azure Machine Learning workspace.
- Apply best practices for scaling, access control, and secure data storage in Azure.

## The step-by-step process to create a compute instance
This reading will guide you through the following steps:

1. Access the Azure Machine Learning studio.
2. Create a compute instance.
3. Create and configure a Jupyter notebook.
4. Run Python code in Azure Machine Learning.

### Step 1: Access the Azure Machine Learning studio
Navigate to `ml.azure.com` to open the Azure Machine Learning studio, your central hub for managing machine learning workflows.

If it's your first visit, Azure may prompt you to create a workspace. Select "Create workspace" from the top right and proceed with the following configurations:

- Assign a unique name to your workspace. You can also set a friendly name for easier identification.
- Choose an active Azure subscription, and create a new resource group to organize related resources. If unsure, use the default naming conventions provided by Azure.
- Select the geographic region closest to your location for optimal performance. Click "Create" to initiate workspace setup.

Workspace creation takes a few minutes. Once completed, click the workspace name to enter the dashboard and explore its features.

### Step 2: Create a compute instance
Within the workspace dashboard, navigate to the "Notebooks" section found under the "Authoring" tab in the left-side menu. This section allows you to manage and execute scripts for your machine learning projects.

Select "Create compute" to initiate the setup of a compute instance. This virtual machine is essential for running code and experiments.

Configure your compute instance by specifying the following:

- Assign a meaningful name, such as "testworkspaceinstance42," to easily identify it later.
- Choose a virtual machine type, such as CPU, for tasks requiring moderate computational power.
- Retain the preselected virtual machine size, which provides sufficient resources for most standard machine learning tasks.

Review your configuration, click "Review + Create," and then confirm by clicking "Create." You can monitor its creation progress in the "Manage -> Compute" section of your workspace. This process may take several minutes to complete.

### Step 3: Create and configure a Jupyter notebook
Once your compute instance is active, proceed to the "+ Files" button within the Notebooks section. Select "Create new file" to initialize your first notebook.

Assign a filename, such as "test.ipynb," and click "Create" to generate the notebook.

Verify that your compute instance is attached. If not automatically linked, select it manually from the instance dropdown menu available in the interface.

Set the notebook kernel to "Python 3.8 - Azure ML" or the most recent version supported by Azure Machine Learning to ensure compatibility with Azure's machine learning libraries.

### Step 4: Run Python code in Azure Machine Learning
Inside your Jupyter notebook, create a new code cell and enter the following Python snippet. Then, execute the code by pressing Shift+Enter.

**Code example**
```python
import tensorflow as tf
print("TensorFlow Version: " + tf.__version__)
```
This command verifies your TensorFlow installation and displays the version information.

Note that when using a CPU-only compute instance, certain warnings about GPU functionality may appear. These warnings can be safely ignored for tasks that do not require GPU acceleration.

If you need to modify code in earlier cells, restart the kernel and rerun all cells to apply changes effectively. This ensures that your notebook's state is consistent and reflects updated scripts.

## Conclusion
After completing these steps, you should have a working compute instance and a connected datastore. Completing this activity has equipped you with the practical skills needed to set up and manage compute and datastore resources in Azure Machine Learning. These configurations form the backbone of scalable, secure, and efficient machine learning workflows. 

Now that you have completed this activity, reflect on how these resources will be used in your upcoming projects. Practice configuring additional instances and datastores to gain confidence in managing Azure Machine Learning resources effectively.

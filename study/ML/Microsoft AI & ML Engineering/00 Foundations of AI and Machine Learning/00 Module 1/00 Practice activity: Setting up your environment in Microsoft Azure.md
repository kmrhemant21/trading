Practice activity: Setting up your environment in Microsoft Azure
===============================================================

### Introduction

In this reading, you’ll set up your environment in Microsoft Azure, building a solid foundation for all testing, application, and deployment tasks you’ll be working on throughout the course. Follow each step to ensure your environment is configured accurately, setting you up for success as you dive into hands-on exercises and apply new concepts.

---

## Step-by-step guide for setting up your environment in Microsoft Azure

Creating your environment in Azure consists of the following five phases that will include various steps: 

- **Phase 1:** Create an Azure account
- **Phase 2:** Set up a workspace
- **Phase 3:** Create a compute instance
- **Phase 4:** Execute code in your Jupyter Notebook
- **Phase 5:** Identify common pitfalls with Notebooks

---

### Phase 1: Create an Azure account

**Step 1: Sign up for Azure**  
If you don't already have an Azure account, visit the [Azure Portal](https://portal.azure.com), and sign up for a free account.

> **IMPORTANT:** your free account will include a $200 credit that expires after 30 days. If you do not complete this program within 30 days, you’ll need to upgrade to a pay-as-you-go account to complete the program.  
> Students at eligible institutions may be eligible for a $100 credit that expires after 12 months.

**Step 2: Access the Azure Portal**  
Once you’ve signed in, you’ll be directed to the Azure Portal dashboard.

---

### Phase 2: Set up a workspace

**Step 1: Access the Azure Machine Learning Studio**  
Go to [https://ml.azure.com](https://ml.azure.com).

**Step 2: Create a new workspace**  
Azure may prompt you to create a new workspace upon your first time visiting ml.azure.com. If it does not, click “Create workspace” near the top right.

- Choose a name for your workspace. Optionally, choose a friendly name.
- Select your existing Azure subscription.
- Create a new resource group; the default name is fine.
- Select the region that’s geographically closest to you.
- Click “Create.”

> *Azure ML Studio screen showing options to create a workspace with fields for name, subscription, resource group, and region.*

**Step 3: Enter the workspace**  
It will take a few minutes for Azure to create your workspace. Once it’s finished, click on the workspace to enter it.

**Step 4: Enter the Notebooks section**  
In the left panel, under “Authoring,” is the “Notebooks” section. Click on it.

> *Azure ML Studio menu with 'Notebooks' circled under 'Authoring'.*

---

### Phase 3: Create a compute instance

**Step 1:** In the Notebooks section, click “Create compute”  
> *Azure ML Studio "Notebooks" section is open with an option to create a compute instance for running Jupyter Notebooks.*

**Step 2: Define the required settings**

- Give your compute instance a name.
- Select virtual machine type CPU.
- Keep the preselected virtual machine size.
- Click “Review + Create.”

**Step 3: Create the compute instance**  
Click “Create.”

---

### Phase 4: Execute code in your Jupyter Notebook

**Step 1: Create a Jupyter Notebook**

- Click the “+ Files button”
- Select “Create new file”
- Change the file name to `test.ipynb`
- Click “Create”

**Step 2: Attach a compute instance**  
If you just created your first compute instance, you’ll have to wait for Azure to finish creating it before you can proceed. 

The instance should automatically attach. If it doesn’t, select your instance from the drop-down menu.

> *Compute instance dropdown in Azure ML Studio showing "testinstance1 - Running" with specs: 4 cores, 32 GB RAM, 64 GB disk.*

**Step 3: Select the appropriate kernel**  
The kernel selection menu is on the top right. Select “Python 3.8 - Azure ML” or the latest version of the Azure ML kernel. It’ll take a moment to become active.

**Step 4: Execute code**  
Type this code into the code cell in your notebook:

```python
import tensorflow as tf
print("TensorFlow Version: " + tf.__version__)
```

Press `Shift+Enter` to run the code and proceed to the next cell. You should see some error messages followed by your TensorFlow version.

---

### Phase 5: Identify common pitfalls with Notebooks

**Pitfall 1: Error messages**  
When using libraries like TensorFlow, you will get error messages you can safely ignore because they pertain to GPU functionality and you are using a CPU-only instance:

> *Code output from running a Python script. Displays multiple warnings related to TensorFlow, CUDA, and cuDNN plugins.*

**Pitfall 2: Modifications to previous cells don’t take effect until you run those cells**  
Modifications to code in cells don’t immediately take effect. If you change code in a previous cell, it’s recommended to click the “Restart kernel and run all cells” button:

> *Azure ML Studio, Jupyter notebook file 'test.ipynb' open, with 'Run Cell' button circled in red.*

**Pitfall 3: Wrong kernel**  
New notebooks will use the Python 3.10 - SDK v2 kernel by default. Make sure you change your kernel to Python 3.8 - Azure ML or the latest version of the Azure ML kernel.

---

## Conclusion

After following the steps above, your Azure environment should now be set up and ready to use.

If you need further help, see the following documents or consult Microsoft Copilot:

- [Quickstart: Get started with Azure Machine Learning - Azure Machine Learning | Microsoft Learn](https://learn.microsoft.com/azure/machine-learning/quickstart-create-resources)
- [Tutorial: Create workspace resources - Azure Machine Learning | Microsoft Learn](https://learn.microsoft.com/azure/machine-learning/tutorial-1st-experiment-sdk-setup)

---

### Note: Stopping Your Compute Instance in Azure

Once you have finished working with your Azure environment, it is important to stop the compute instance to avoid unnecessary charges.

To do this:

1. Navigate to the Compute section in the Azure Machine Learning Studio.
2. Locate your active compute instance.
3. Click the Stop button.

This will pause the instance and prevent additional costs from accruing. If you no longer need the instance, you can delete it by selecting Delete instead.
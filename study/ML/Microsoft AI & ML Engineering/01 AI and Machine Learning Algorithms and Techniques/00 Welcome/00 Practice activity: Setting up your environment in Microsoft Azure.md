# Practice Activity: Setting Up Your Environment in Microsoft Azure

## Introduction

In this reading, you’ll set up your environment in Microsoft Azure, building a solid foundation for all testing, application, and deployment tasks you’ll be working on throughout the course. Follow each step to ensure your environment is configured accurately, setting you up for success as you dive into hands-on exercises and apply new concepts.

---

## Step-by-Step Guide for Setting Up Your Environment in Microsoft Azure

Creating your environment in Azure consists of the following five phases:

1. **Create an Azure account**
2. **Set up a workspace**
3. **Create a compute instance**
4. **Execute code in your Jupyter Notebook**
5. **Identify common pitfalls with Notebooks**

---

### Phase 1: Create an Azure Account

**Step 1: Sign up for Azure**  
If you don't already have an Azure account, visit the [Azure Portal](https://portal.azure.com) and sign up for a free account.

> **IMPORTANT:**  
> Your free account includes a \$200 credit that expires after 30 days. If you do not complete this program within 30 days, you’ll need to upgrade to a pay-as-you-go account to continue.  
> Students at eligible institutions may be eligible for a \$100 credit that expires after 12 months.

**Step 2: Access the Azure Portal**  
Once signed in, you’ll be directed to the Azure Portal dashboard.

---

### Phase 2: Set Up a Workspace

**Step 1: Access the Azure Machine Learning Studio**  
Go to [https://ml.azure.com](https://ml.azure.com).

**Step 2: Create a New Workspace**  
- Azure may prompt you to create a new workspace on your first visit. If not, click **Create workspace** near the top right.
- Choose a name for your workspace (optionally, a friendly name).
- Select your existing Azure subscription.
- Create a new resource group (the default name is fine).
- Select the region closest to you.
- Click **Create**.

**Step 3: Enter the Workspace**  
Wait a few minutes for Azure to create your workspace. Once finished, click on the workspace to enter it.

**Step 4: Enter the Notebooks Section**  
In the left panel, under **Authoring**, click **Notebooks**.

---

### Phase 3: Create a Compute Instance

**Step 1:** In the Notebooks section, click **Create compute**.

**Step 2: Define the Required Settings**  
- Name your compute instance.
- Select virtual machine type **CPU**.
- Keep the preselected virtual machine size.
- Click **Review + Create**.

**Step 3:** Click **Create** to launch the compute instance.

---

### Phase 4: Execute Code in Your Jupyter Notebook

**Step 1: Create a Jupyter Notebook**  
- Click the **+ Files** button.
- Select **Create new file**.
- Name the file `test.ipynb`.
- Click **Create**.

**Step 2: Attach a Compute Instance**  
Wait for your compute instance to finish creating. It should attach automatically; if not, select your instance from the drop-down menu.

**Step 3: Select the Appropriate Kernel**  
At the top right, select **Python 3.8 - Azure ML** or the latest Azure ML kernel. Wait for it to become active.

**Step 4: Execute Code**  
Type the following code into a code cell:

```python
import tensorflow as tf
print("TensorFlow Version: " + tf.__version__)
```

Press `Shift+Enter` to run the code. You should see some error messages (related to GPU functionality) followed by your TensorFlow version.

---

### Phase 5: Identify Common Pitfalls with Notebooks

- **Pitfall 1: Error Messages**  
    When using libraries like TensorFlow, you may see error messages about GPU functionality. These can be safely ignored if you are using a CPU-only instance.

- **Pitfall 2: Modifications to Previous Cells Don’t Take Effect Until You Run Those Cells**  
    If you change code in a previous cell, click **Restart kernel and run all cells** to ensure changes take effect.

- **Pitfall 3: Wrong Kernel**  
    New notebooks may default to **Python 3.10 - SDK v2**. Change the kernel to **Python 3.8 - Azure ML** or the latest Azure ML kernel.

---

## Conclusion

After following these steps, your Azure environment should be set up and ready to use.

For further help, see:

- [Quickstart: Get started with Azure Machine Learning - Azure Machine Learning | Microsoft Learn](https://learn.microsoft.com/azure/machine-learning/quickstart-create-resources)
- [Tutorial: Create workspace resources - Azure Machine Learning | Microsoft Learn](https://learn.microsoft.com/azure/machine-learning/tutorial-create-resources)

---

### Note: Stopping Your Compute Instance in Azure

Once you have finished working, stop your compute instance to avoid unnecessary charges:

1. Navigate to the **Compute** section in Azure Machine Learning Studio.
2. Locate your active compute instance.
3. Click the **Stop** button.

This will pause the instance and prevent additional costs. If you no longer need the instance, you can delete it by selecting **Delete**.


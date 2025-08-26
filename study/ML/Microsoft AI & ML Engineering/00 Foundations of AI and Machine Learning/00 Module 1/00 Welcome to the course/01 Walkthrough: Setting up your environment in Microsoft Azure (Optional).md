# Walkthrough: Setting up your environment in Microsoft Azure (Optional)

By now, you have set up a lab environment in Microsoft Azure. If you haven't, see "**Practice activity: Setting Up Your Environment in Microsoft Azure**" for step-by-step instructions. This reading provides a more general overview, as well as an explanation as to the rationale behind some of the steps you took to set up your environment, and explains how these steps are used by professionals in the AI/ML industry.

By the end of this walkthrough, you will be able to:

- Understand the process of setting up an AI/ML environment in Azure.
- Apply professional cloud management practices.
- Install and configure essential AI/ML tools.

---

## Overview of setting up your environment in Microsoft Azure

### An Azure account

**What it does:**  
Creating an Azure account is your entry point into the Azure ecosystem. This account gives you access to Azure's vast array of cloud services, including those necessary for AI/ML development. By signing up, you get a centralized dashboard (Azure Portal) in which you can manage all your resources, such as virtual machines (VMs), databases, and networking components.

**Professional use:**  
Professionals use Azure accounts to deploy and manage scalable applications in the cloud. Having an Azure account is essential for accessing the tools and services required for AI/ML projects, from initial data processing to model deployment.

---

### Resource groups

**What it does:**  
A resource group in Azure is a logical container for resources that share the same life cycle, such as VMs, storage accounts, and databases. By organizing resources into groups, you can manage and monitor them collectively, apply access controls, and track costs more effectively.

**Professional use:**  
In a professional setting, resource groups help teams organize resources for different projects or environments (e.g., development, testing, production). This organizational structure simplifies resource management and groups related assets together, making it easier to manage and scale AI/ML solutions.

---

### Virtual machines

**What it does:**  
A VM is a software-based emulation of a physical computer. In Azure, a VM allows you to run an isolated environment in which you can install and configure the software needed for AI/ML tasks, such as Python, Jupyter Notebooks, and ML libraries.

**Professional use:**  
Professionals use VMs to create development environments tailored to specific projects. For AI/ML engineers, VMs provide the flexibility to experiment with different tools, test code, and run ML models without impacting their local machine or production systems. Azure’s VMs are scalable, meaning you can adjust computing resources based on your workload’s demands.

---

### SSH access

**What it does:**  
Secure Shell (SSH) access allows you to connect to your VM securely from your local machine. It encrypts the connection between your computer and the VM, ensuring that your data and commands are secure.

**Professional use:**  
SSH is a fundamental tool for professionals who need to manage and operate their VMs remotely. By using SSH, AI/ML engineers can interact with their cloud-based environments as if they were sitting directly at the machine, allowing them to run scripts, install software, and troubleshoot issues from anywhere.

---

### Essential tools and libraries

**What it does:**  
Python is the primary programming language used for ML, and libraries such as NumPy, pandas, and Scikit-learn provide the tools necessary for data manipulation, analysis, and model development. Jupyter Notebook offers an interactive environment for writing and running code, making it easier to visualize data and share results.

**Professional use:**  
Professionals rely on these tools to conduct data analysis, develop ML models, and iterate quickly on their experiments. Python's rich ecosystem of libraries makes it the go-to language for AI/ML development. Jupyter Notebooks are widely used in the industry for developing and documenting ML workflows, especially in collaborative environments.

---

### Save as a template

**What it does:**  
Saving your environment as an Azure Resource Manager (ARM) template allows you to capture the configuration of your lab environment in a reusable format. You can use this template to recreate the environment quickly, ensuring consistency across different projects or team members.

**Professional use:**  
Professionals use ARM templates to automate the deployment of complex environments. By saving a VM configuration as a template, AI/ML teams can ensure that their environments are reproducible, which is critical for collaborative projects and scaling solutions across multiple instances or regions. It also allows for quick recovery if an environment needs to be rebuilt.

---

## Conclusion

Setting up your environment in Azure is a crucial step in your journey to becoming proficient in AI/ML deployment. Each step in this process mirrors the real-world practices of AI/ML professionals, from organizing resources and managing environments to ensuring security and scalability. Understanding the purpose and application of each step will help you work more effectively and prepare you for the tasks you’ll face as an AI/ML engineer in the industry.

---
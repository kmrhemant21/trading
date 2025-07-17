# Walkthrough: Setting up your environment in Microsoft Azure (Optional)

By now, you should have set up a lab environment in Microsoft Azure. If not, refer to the **Practice activity: Setting Up Your Environment in Microsoft Azure** for step-by-step guidance. This overview explains the rationale behind each setup step and how these practices are used by AI/ML professionals.

## Learning Objectives

By the end of this walkthrough, you will be able to:

- Understand the process of setting up an AI/ML environment in Azure.
- Apply professional cloud management practices.
- Install and configure essential AI/ML tools.

---

## Overview of Setting Up Your Environment in Microsoft Azure

### 1. Azure Account

**What it does:**  
An Azure account is your entry point to the Azure ecosystem, providing access to a wide range of cloud services for AI/ML development. The Azure Portal serves as a centralized dashboard to manage resources like virtual machines (VMs), databases, and networking.

**Professional use:**  
Professionals use Azure accounts to deploy and manage scalable cloud applications. Access to these tools and services is essential for all stages of AI/ML projects, from data processing to model deployment.

---

### 2. Resource Groups

**What it does:**  
A resource group is a logical container for resources sharing the same lifecycle (e.g., VMs, storage, databases). It enables collective management, access control, and cost tracking.

**Professional use:**  
Resource groups help teams organize resources by project or environment (development, testing, production), simplifying management and scaling of AI/ML solutions.

---

### 3. Virtual Machines

**What it does:**  
A VM is a software-based emulation of a physical computer. In Azure, VMs provide isolated environments for installing and configuring AI/ML tools like Python, Jupyter Notebooks, and ML libraries.

**Professional use:**  
VMs allow professionals to create tailored development environments, experiment with tools, and run ML models without affecting local or production systems. Azure VMs are scalable to match workload demands.

---

### 4. SSH Access

**What it does:**  
SSH (Secure Shell) enables secure, encrypted connections to your VM from your local machine.

**Professional use:**  
SSH is essential for remote management of VMs. AI/ML engineers use SSH to run scripts, install software, and troubleshoot from anywhere, as if working directly on the machine.

---

### 5. Essential Tools and Libraries

**What it does:**  
Python is the primary language for ML, with libraries like NumPy, pandas, and scikit-learn for data manipulation and model development. Jupyter Notebook offers an interactive coding environment for visualization and sharing.

**Professional use:**  
These tools are industry standards for data analysis and ML model development. Jupyter Notebooks are especially valued for collaborative workflows and documentation.

---

### 6. Save as a Template

**What it does:**  
Saving your environment as an Azure Resource Manager (ARM) template captures your configuration in a reusable format, enabling quick recreation and consistency.

**Professional use:**  
ARM templates automate deployment of complex environments, ensuring reproducibility and scalability across projects and teams. They also facilitate quick recovery if an environment needs to be rebuilt.

---

## Conclusion

Setting up your environment in Azure is a foundational skill for AI/ML deployment. Each step reflects real-world practices—organizing resources, managing environments, ensuring security, and enabling scalability. Mastering these steps prepares you for professional tasks as an AI/ML engineer.
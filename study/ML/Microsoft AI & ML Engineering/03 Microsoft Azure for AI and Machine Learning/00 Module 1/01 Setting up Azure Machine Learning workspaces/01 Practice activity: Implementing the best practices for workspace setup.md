# Practice activity: Implementing the best practices for workspace setup

## Introduction
In this hands-on activity, you will implement the best practices for setting up an Azure Machine Learning workspace. This exercise is designed to give you practical experience in setting up and configuring the key components of the workspace. By following these steps, you will gain the foundational skills needed to manage machine learning projects in Azure effectively.

**Note**: this activity is similar in nature to the workspace setup activity. If you have not yet tried to do this, now is a good time to try.

By the end of this activity, you will be able to:
- Set up an Azure Machine Learning workspace using the Azure portal.
- Configure the workspace to ensure consistency, reproducibility, and seamless teamwork.

## Step-by-step process for best practices in workspace setup
This reading will guide you through the following steps:

1. Step 1: Structure your workspace.
2. Step 2: Set up a virtual environment.
3. Step 3: Install dependencies.
4. Step 4: Implement version control.
5. Step 5: Configure Jupyter notebooks.
6. Step 6: Maintain documentation.

### Step 1: Structure your workspace
Begin by creating a structured directory for your project. Consistency in file organization not only helps you stay organized but also enables collaborators to navigate the workspace effortlessly.

- **data/**: store raw datasets and any processed versions of the data.
- **notebooks/**: place all Jupyter notebooks used for data exploration and prototyping.
- **scripts/**: include Python scripts for tasks, such ase data preprocessing, model training, and evaluation, that are executed outside notebooks.
- **models/**: save trained model files and their versions.
- **logs/**: keep execution and performance logs to track script runs and debug issues.

Create the following folders/directories in your workspace by using the "Add files" button and then clicking "Create new folder": {data,notebooks,scripts,models,logs}.

Alternatively, open a terminal on your compute instance and run the following command to create this directory structure:

```
mkdir -p ml_project/{data,notebooks,scripts,models,logs}
```

This command generates a consistent project directory in one step, establishing an organized workspace foundation. 

### Step 2: Set up a virtual environment
Open the terminal in the"Notebooks" section of Azure Machine Learning.

Create a virtual environment to isolate project dependencies. Virtual environments prevent conflicts between package versions used in different projects.

```
conda create --name new_ml_env
conda activate new_ml_env
```

Activate the environment, and ensure that all project dependencies remain isolated for consistency across collaborators.

### Step 3: Install dependencies
Use a requirements.txt file to document all necessary packages and their versions. Copy and paste this code into the terminal, line-by-line. This creates a document called "requirements.txt" with a different Python package on each line. 

```
cat > requirements.txt <<EOL
pandas
scikit-learn
matplotlib
Tensorflow
```

Then, on a new line, type "EOL" without the quotes, then press enter.

Install dependencies using the file to ensure consistency in package versions across all environments.

```
conda install --file requirements.txt
```

This step guarantees that every collaborator works with the same versions of required libraries. 

### Step 4: Implement version control
Initialize a Git repository in your project directory to enable version control. This tracks changes, facilitates collaboration, and prevents data loss. 

```
git init
git add .
git commit -m "Initial commit"
```

If you see any error messages during this step, refer to earlier activities in this course about setting up a code repository. 

Use Git to manage the history of your files, revert to earlier versions when needed, and collaborate efficiently. For guidance on creating a repository, refer to the introductory screencasts in Module 1 of this course.

### Step 5: Configure Jupyter notebooks
Install Jupyter within your virtual environment to ensure that it uses the appropriate dependencies:

```
conda install jupyter
pip install ipykernel
python -m ipykernel install --user --name=new_ml_env --display-name "ML Project Env"
```

This adds the virtual environment to Jupyter as a kernel and ensures that your notebooks use the correct package versions, avoiding compatibility issues.

Open a new notebook and confirm that the kernel displays "ML Project Env." If it doesn't, select it from the kernel selection dropdown menu. 

### Step 6: Maintain documentation
Create a README.md file to describe your project, dependencies, and setup instructions. This serves as a guide for new collaborators. To do this, create a new file, call it readme.md, and select file type "Other." You can click on the file to view it if it hasn't already opened, then click the "Markdown editor" button near the top left of the viewer to edit it. 

Add a brief overview, instructions for setting up the environment, and descriptions of directory contents. Put this in the left column of the markdown editor. 

```
# ML Project
## Directory Structure
- `data/`: Raw and processed datasets
- `notebooks/`: Jupyter notebooks
- `scripts/`: Python scripts
- `models/`: Trained model files
- `logs/`: Execution logs

## Setup Instructions
Create a virtual environment.

`conda create --name new_ml_env`

Activate the environment.

`conda activate new_ml_env`

Install dependencies.

`conda install --file requirements.txt`
```

Save the file as README.md to provide collaborators with clear and concise setup instructions. 

There's a save button near the top left of the markdown editor. 

## Conclusion
This activity guides you through the process of setting up an Azure Machine Learning workspace, which centralizes your machine learning resources, enabling efficient project management and scalability. By familiarizing yourself with these setup practices, you will be well prepared to organize and optimize your resources, providing a strong foundation for your AI/ML projects on Azure.

Take this opportunity to practice setting up your own Azure Machine Learning workspace. Understanding the best practices for configuring these resources will ensure that your projects are scalable, organized, and easy to manage, providing a solid foundation for advanced AI/ML endeavors.

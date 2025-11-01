# Practice activity: Authenticating to Azure Machine Learning

## Introduction
Imagine you're working on a sensitive AI/ML project and need to access resources from within a Python script. How does Azure Machine Learning ensure it's really you accessing those resources and not an attacker? Authentication is your first line of defense, ensuring only authorized users can access critical resources. Mastering Azure ML authentication is key to safeguarding your data and maintaining secure workflows.

By the end of this activity, you will be able to:

- Authenticate to an Azure ML workspace using various methods.
- Identify the advantages and use cases of each authentication type.
- Securely manage secrets during remote runs.

## Options for authenticating in Azure Machine Learning
This reading will guide you through the following methods:

- Interactive login authentication
- Azure CLI authentication
- Managed service identity (MSI) authentication
- Service principal authentication
- Token authentication

## Step-by-step authentication instructions
### Step 1: Open the authentication notebook
Access the notebook:

1. Go to [https://azure.com](https://azure.com) and sign in if prompted.
2. Open your Azure ML workspace.
3. Navigate to Notebooks and switch to the Samples tab.
4. Go to SDK v1 > how-to-use-azureml > manage-azureml-service > authentication-in-azureml.
5. Clone the authentication-in-azureml.ipynb notebook and its dependencies into your workspace.

Prepare the environment:

- Ensure your compute instance is running.
- Set the kernel to Python 3.8-AzureML.

**Note:** This notebook provides a guided, hands-on approach to understanding and implementing Azure ML authentication methods, saving you time and effort in learning these techniques from scratch.

### Step 2: Use an authentication method
#### Method 1: Interactive login authentication
Interactive login is the default method and ideal for quick access during development.

**Why use this method?**

- Best for developers who need a simple and quick setup for development.

**Steps:**

1. Import the required module:

```python
from azureml.core import Workspace
```

2. Authenticate:

```python
ws = Workspace.from_config()
```

3. Troubleshoot (if needed):
    - Specify the Subscription ID, Resource group, and Workspace name. These details are available in the dropdown near your user icon on the Azure portal.

#### Method 2: Azure CLI authentication
Azure CLI integrates seamlessly with command-line tools, making it a preferred method for users familiar with CLI.

**Why use this method?**

- Ideal for users comfortable with CLI tools and managing Azure resources through the command line.

**Steps:**

1. Install Azure CLI:
    - Follow the [Azure CLI installation guide](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli).

2. Log in using Azure CLI:

```bash
az login
```

3. Authenticate in your script:

```python
from azureml.core.authentication import AzureCliAuthentication
cli_auth = AzureCliAuthentication()
ws = Workspace.from_config(auth=cli_auth)
```

#### Method 3: Managed service identity (MSI) authentication
MSI eliminates the need for passwords or secrets, making it highly secure.

**Why use this method?**

- Best for securing applications running on Azure infrastructure.

**Steps:**

1. Ensure your environment supports MSI:
    - This method works on Azure VMs or services with MSI enabled.

2. Use the ManagedIdentityCredential class:

```python
from azure.identity import ManagedIdentityCredential
from azureml.core import Workspace

credential = ManagedIdentityCredential()
ws = Workspace(subscription_id="your_subscription_id",
resource_group="your_resource_group", workspace_name="your_workspace_name",credential=credential)
```

#### Method 4: Service principal authentication
Service principal authentication is suitable for automated workflows or CI/CD pipelines.

**Why use this method?**

- Ideal for automated workflows requiring strict access control.

**Steps:**

1. Set up a Service Principal:
    - Register a new application in Azure Active Directory.
    - Record the Application ID, Tenant ID, and Client Secret.

2. Assign permissions:
    - Grant the Service Principal access to your Azure ML workspace.

3. Authenticate in your script:

```python
from azureml.core.authentication import ServicePrincipalAuthentication

svc_pr = ServicePrincipalAuthentication(
     tenant_id="your_tenant_id",
     service_principal_id="your_application_id",
     service_principal_password="your_client_secret"
)
ws = Workspace.from_config(auth=svc_pr)
```

#### Method 5: Token authentication
Token authentication offers granular control and is suitable for integrations with external systems.

**Why use this method?**

- Flexible for integrations requiring token-based access.

**Steps:**

1. Generate a token externally using tools compatible with Azure ML.

2. Use the token in your script:

```python
from azureml.core.authentication import InteractiveLoginAuthentication

token_auth = InteractiveLoginAuthentication(token="your_token_here")
ws = Workspace.from_config(auth=token_auth)
```

### Step 3: Handle secrets in remote runs
Securely managing secrets is a best practice as it protects your applications from unauthorized access.

**Steps:**

1. Set up Azure Key Vault:
    - Link your Azure ML workspace to a Key Vault.
    - Add secrets (e.g., API keys, passwords) as key-value pairs.

2. Access secrets in your script:

```python
from azureml.core import Workspace

ws = Workspace.from_config()
secret = ws.get_default_keyvault().get_secret(name="your_secret_name")
```

## Conclusion
In this activity, you learned how to:

- Authenticate to an Azure ML workspace using multiple methods.
- Recognize the benefits and use cases of each authentication type.
- Manage secrets securely during remote runs.

Now it's your turn. Ensure your Azure ML applications handle authentication properly, safeguard your sensitive data, and lock down access to authorized individuals only. Mastering these techniques will help you secure your workflows and enhance your ML projects.

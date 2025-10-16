# Course assignment: Producing a troubleshooting agent

> **Disclaimer**: Be aware that disabling the content filter can allow unintended outputs, including inappropriate or harmful content. This step should be limited to testing phases in secure environments. Always re-enable the filter before deploying to production to ensure safe interactions.  

## Introduction
Picture this—you're creating powerful AI solutions in Microsoft Machine Learning Studio, but you have a problem with your deployment. How do you resolve it? You can build your own personal troubleshooting agent to help you with these problems.

By the end of this activity, you will be able to:

* Navigate to and demonstrate proficiency with Azure deployments for an LLM-based troubleshooting agent.
* Deploy an LLM-based troubleshooting agent using serverless endpoints.

## Step-by-step guide: Create a troubleshooting agent
This reading will guide you through the following steps:

1. Step 1: Open Azure Machine Learning Studio.
2. Step 2: Access the endpoints menu.
3. Step 3: Select a model.
4. Step 4: Configure deployment settings.
5. Step 5: Test your deployment.
6. Step 6: Interact with your agent.

### Step 1: Open Azure Machine Learning Studio
1. Open a web browser and navigate to `ml.azure.com`.
2. Sign in to your Azure account.

### Step 2: Access the endpoints menu
1. On the left-hand menu, locate Endpoints and click on it.
2. Under Endpoints, select Serverless Endpoints.
3. Click the Create button to start setting up a new serverless endpoint.

### Step 3: Select a model
> If you have the free subscription for Azure: Use the Search Models field to find Phi-4-Mini-Instruct  

1. Use the Search Models field to find your desired model.
2. For this example, type Llama 2 70B and select it from the results.
3. Once the model is selected, click Subscribe and Deploy to proceed.

### Step 4: Configure deployment settings
1. In the deployment settings, disable the Content Filter option.
2. Enter a custom name for your endpoint (e.g., llm-troubleshooting-agent).
3. Click Deploy to initiate the deployment process.

**Note**: Deployment may take several minutes. Wait until the Provisioning State and Endpoint State indicate success.

### Step 5: Test your deployment
1. Once the deployment is complete, navigate to the Test tab at the top of the screen.
2. Paste the following prompt into the input field:

**Example prompt**
```
You are an intelligent troubleshooting agent. Users will pose questions to you asking for basic technical support regarding hardware or software support, and you will respond with a best guess as to the solution and appropriate remediation steps. Your tone should be supportive, empathetic, apologetic, and professional. If necessary, ask the user for more specifics about their device—manufacturer, operating system, or other pertinent details you require to provide the best solution. If the first 
```
3. Press Enter to test the agent's capabilities.

### Step 6: Interact with your agent
1. Describe a technical issue to the agent and observe its response.

**Example input**
```
I need help creating a new compute instance in Azure Machine Learning Studio.
```
2. The troubleshooting agent will provide a detailed step-by-step solution to the query.

## Conclusion
Deploying, customizing, and testing a model in Azure Machine Learning Studio is a straightforward process. In this summative activity, you:

* Learned how to deploy an LLM-based troubleshooting agent using serverless endpoints.
* Experimented with customizing and testing a model to address specific use cases.

Now it's your turn. Explore different models and prompts to create troubleshooting agents tailored to your unique requirements.

# A practical guide: Deploying AI/ML models

**Disclaimer:**

The code provided is intended to help you understand the concept and flow of the process described. Do not attempt to run the code as-is, as it may be incomplete or require additional configurations. Use this material as a reference, not as a deployment-ready solution.

## Introduction

You’ve built your model, refined it, and now it's time for the big moment—deploying it to make real-world impact. Taking an AI/ML model from development to deployment is where all your hard work starts to deliver value.

In this reading, we’ll walk through the process of deploying ML models using the tools and platforms we’ve discussed throughout the course. Deployment is the critical step that takes your model from development to real-world application, where it can start delivering value. We’ll cover the deployment process using Microsoft Azure Kubernetes Service (AKS) and Azure App Services, complete with code samples to guide you through the setup.

By the end of this reading, you’ll be able to: 

- Deploy machine learning models using AKS and Azure App Services.
- Practice packaging models in Docker containers, creating and managing Azure resources for deployment, and exposing your models as accessible services.

---

## Deploying a model with AKS

AKS is a managed container orchestration service that simplifies the deployment, management, and scaling of containerized applications using Kubernetes. It’s an ideal platform for deploying ML models that need to handle high traffic and require scaling.

### Step 1: Prepare your model

First, ensure that your ML model is packaged into a Docker container. This container should include the model itself, any necessary dependencies, and the code to serve predictions. 

**Dockerfile example:**
```
# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
```

### Step 2: Push your Docker image to Azure Container Registry (ACR)

Build your Docker image and push it to ACR so that AKS can easily access it.  

**Build and push commands:**
```
# Log in to Azure
az login

# Create a resource group
az group create --name myResourceGroup --location eastus

# Create an ACR instance
az acr create --resource-group myResourceGroup --name myRegistry --sku Basic

# Log in to the ACR instance
```

### Step 3: Create an AKS cluster

Now, set up an AKS cluster where your containerized model will be deployed.  
```
# Create an AKS cluster
az aks create --resource-group myResourceGroup --name myAKSCluster --node-count 1 --enable-addons monitoring --generate-ssh-keys
```

### Step 4: Deploy the model to AKS

Create a Kubernetes deployment to run your model in the AKS cluster.

**Deployment YAML example:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
    name: my-model-deployment
spec:
    replicas: 2
    selector:
        matchLabels:
            app: my-model
    template:
```

Apply the deployment:
```
kubectl apply -f deployment.yaml
```

### Step 5: Expose the deployment as a service

Finally, expose your deployment to the internet.

**Service YAML example:**
```yaml
apiVersion: v1
kind: Service
metadata:
    name: my-model-service
spec:
    type: LoadBalancer
    ports:
    - port: 80
    selector:
        app: my-model
```

Apply the service:
```
kubectl apply -f service.yaml
```

### Step 6: Access your deployed model

Once the service is created, you’ll receive an external IP address where your model is accessible.  

Check service status:
```
kubectl get service my-model-service
```
Use the external IP to send requests to your deployed model.

---

## Deploying a model with Azure App Services

Azure App Services is a fully managed platform for building, deploying, and scaling web apps. It’s a great choice for deploying ML models that need to be accessed via HTTP requests, such as APIs.

### Step 1: Prepare your model

Similar to the AKS deployment, package your model in a Docker container, or prepare a Python Flask app that serves your model predictions.

**Example Flask app (app.py):**
```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
```

### Step 2: Create an App Service in Azure

Set up an App Service in Azure where your model will be deployed.  

**App Service creation command:**
```
# Create an App Service plan
az appservice plan create --name myAppServicePlan --resource-group myResourceGroup --sku B1 --is-linux

# Create the Web App
az webapp create --resource-group myResourceGroup --plan myAppServicePlan --name myModelApp --deployment-container-image-name myRegistry.azurecr.io/my-model:latest
```

### Step 3: Configure continuous deployment

You can set up continuous deployment from ACR to Azure App Services to ensure that updates to your model are automatically deployed.  

**Continuous deployment setup:**
```
az webapp deployment container config --name myModelApp --resource-group myResourceGroup --enable-cd true
```

### Step 4: Access your deployed model

After deployment, your model will be accessible via the web app’s URL.

Get the web app URL:
```
az webapp show --name myModelApp --resource-group myResourceGroup --query defaultHostName
```
Use this URL to send HTTP requests to your model.

---

## Conclusion

Deploying your AI/ML models is the final step in bringing your solutions to life. Whether you choose AKS for its scalability and flexibility or Azure App Services for its ease of use and integration, the key is to ensure that your deployment strategy aligns with your project’s requirements. With the code samples and steps provided here, you should be well equipped to deploy your models effectively and start delivering real-world value.

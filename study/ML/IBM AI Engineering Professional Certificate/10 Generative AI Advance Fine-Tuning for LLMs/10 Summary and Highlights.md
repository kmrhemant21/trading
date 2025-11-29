# Summary and Highlights

Congratulations! You have completed this module. At this point in the course, you know the following:

## Direct Preference Optimization (DPO) Overview

**Direct Preference Optimization (DPO)** is a reinforcement learning technique designed to fine-tune models based on human preferences more directly and efficiently than traditional methods.

## Key Concepts

- **Data Collection**: DPO involves collecting data on human preferences by showing users different outputs from the model and asking them to choose the better one.

- **Model Architecture**: DPO involves three models:
    - The reward function (uses an encoder model)
    - The target decoder
    - The reference model

- **Problem Simplification**: In DPO, you can convert a complex problem into a simpler objective function that is more straightforward to optimize.

## Implementation Steps

### Two Main Steps for Fine-tuning with DPO:
1. **Data collection**
2. **Optimization**

### Detailed Steps with Hugging Face:

**Step 1: Data preprocessing**
- Reformat
- Define and apply the process function
- Create the training and evaluation sets

**Step 2: Create and configure the model and tokenizer**

**Step 3: Define training arguments and DPO trainer**

**Step 4: Plot the model's training loss**

**Step 5: Load the model**

**Step 6: Inferencing**

## Technical Details

- DPO leverages a closed-form optimal policy as a function of the reward to reformulate the problem
- Subtracting the reward model for two samples eliminates the need for the partition function

### Loss Function:

$$L_{DPO}(\pi_\theta) = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

**Where:**
- $\pi_\theta$ is the policy being optimized
- $\pi_{ref}$ is the reference policy
- $y_w$ is the preferred output
- $y_l$ is the less preferred output
- $\beta$ is a temperature parameter
- $\sigma$ is the sigmoid function
# Practice activity: Comparing fine-tuning techniques

**Disclaimer**: The time required for this activity may vary based on your hardware (e.g., slower GPUs or CPUs), the size and complexity of the dataset, and your familiarity with techniques like LoRA and QLoRA. Beginners may take longer to complete certain steps, especially when creating detailed reports or working with larger datasets.

## Introduction

In this activity, you will apply and compare the different fine-tuning techniques we've covered so far: traditional fine-tuning, Low-Rank Adaptation (LoRA), and Quantized Low-Rank Adaptation (QLoRa). You will evaluate each approach's performance and resource efficiency and learn how to choose the right method based on the task and hardware constraints.

This activity compares how traditional fine-tuning, LoRA, and QLoRA perform in terms of computational cost, memory usage, and task performance. You will fine-tune the same pretrained model using these three techniques and compare the results based on training time, memory consumption, and model accuracy on a task-specific dataset.

By the end of this activity, you will be able to:

- Apply different fine-tuning techniques to a pretrained model.
- Compare the computational efficiency and model performance of each fine-tuning technique.
- Evaluate how to choose the best fine-tuning technique based on hardware.

## Step-by-step guide to compare techniques

Create a new Jupyter Notebook. You can call it "fine_tuning_comparion". Make sure you have the appropriate Python kernel selected.

The remaining of this reading will guide you through the following steps:

1. Step 1: Prepare your dataset
2. Step 2: Apply traditional fine-tuning
3. Step 3: Fine-tune with LoRA
4. Step 4: Fine-tune with QLoRA
5. Step 5: Compare and analyze results

### Step 1: Prepare your dataset

The first step is to prepare your dataset for fine-tuning. You will use the same dataset for all three techniques to ensure a fair comparison.

**Instructions**
- Load a sample dataset from scikit-learn and inspect its structure.
- Apply preprocessing steps such as cleaning and tokenizing (use the BERT tokenizer) the full dataset before splitting it into sets. This ensures consistent preparation across the entire dataset and avoids data leakage.
- Split the dataset into training, validation, and test sets.

**Code example**
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('your_dataset.csv')

# Split dataset into training, validation, and test sets
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")
print(f"Test set size: {len(test_data)}")
```

### Step 2: Apply traditional fine-tuning

In the first part of the activity, you will apply traditional fine-tuning to the pretrained model. This involves updating all the parameters of the model during training.

**Instructions**
- Load a pretrained model (e.g., BERT or GPT).
- Fine-tune the entire model on the task-specific dataset.
- Record the training time, memory usage, and evaluation metrics.

**Code example**
```python
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results_traditional',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
)

# Fine-tune the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
)

# Start fine-tuning
trainer.train()
```

**Record**
- Training time: initialize a timer before each training block to print a message when training is complete.
- Memory usage: use monitoring tools (e.g., nvidia-smi) to track memory consumption.
- Model performance: print each performance metric (accuracy, precision, recall, F1 score) to the console after each block to compare between methods.

### Step 3: Fine-tune with LoRA

Next, you will fine-tune the model using LoRA. This technique updates only low-rank matrices added to certain layers while freezing the rest of the model's parameters.

**Instructions**
- Import the necessary libraries, ensure the correct kernel is selected and tokenize the data set.
- Install the peft package.
- Initialize the BERT model and then define a LoRa configuration, define the rank of the update matrices and the alpha scaling factor.
- Apply LoRA to specific layers of the pretrained model (e.g., attention layers).
- Create a LoRa model using peft and set the number of training epochs.
- Initialize a data collator, initialize a trainer, and then you can train the model.
- Fine-tune only the low-rank matrices while keeping the other parameters frozen.
- Record the same metrics as in traditional fine-tuning.

**Code example**
```python
from lora import LoRALayer

# Apply LoRA to specific layers (e.g., attention layers)
for name, module in model.named_modules():
    if 'attention' in name:
        module.apply(LoRALayer)

# Freeze the rest of the model
for param in model.base_model.parameters():
    param.requires_grad = False

# Fine-tune the LoRA-enhanced model
trainer.train()
```

**Record**
- Training time: initialize a timer before each training block to print a message when training is complete.
- Memory usage: track memory consumption as you did in the previous step.
- Model performance: print each performance metric (accuracy, precision, recall, F1 score) to the console after each block to compare between methods.

### Step 4: Fine-tune with QLoRA

Finally, you will apply QLoRA to the pretrained model. QLoRA quantizes the model's parameters to reduce memory usage even further and fine-tunes only low-rank matrices.

**Instructions**
- Import the necessary libraries, ensure the correct kernel is selected and tokenize the data set.
- Quantize the pretrained model to reduce the memory footprint.
- Apply LoRA to the quantized model's layers.
- Fine-tune the model and record the metrics.

**Code example**
```python
from qlora import QuantizeModel

# Quantize the model to reduce memory usage
quantized_model = QuantizeModel(model, bits=8)

# Apply LoRA to specific layers in the quantized model
for name, module in quantized_model.named_modules():
    if 'attention' in name:
        module.apply(LoRALayer)

# Fine-tune the QLoRA-enhanced model
trainer.train()
```

**Record**
- Training time: initialize a timer before each training block to print a message when training is complete.
- Memory usage: track memory usage with monitoring tools.
- Model performance: print each performance metric (accuracy, precision, recall, F1 score) to the console after each block to compare between methods.

### Step 5: Compare and analyze results

After completing the fine-tuning process using all three techniques, you will compare the results. The goal is to analyze the trade-offs between model performance, training time, and memory usage for each technique.

**Questions to consider**
- Which technique was the fastest to train?
- Which technique used the least memory?
- How did the model performance compare across the different techniques?
- Based on your results, which technique would you recommend for scenarios with limited computational resources?

**Deliverables**
By the end of this activity, you should produce:
- A report summarizing the results of the three fine-tuning techniques, including training time, memory usage, and model performance metrics.
- An analysis of which technique is the most efficient and why.
- A reflection on how these techniques can be applied in real-world scenarios.

## Conclusion

By comparing traditional fine-tuning, LoRA, and QLoRA, you will gain a deeper understanding of the trade-offs between computational efficiency and model performance. This activity will help you decide when to use each technique based on the specific needs of your task and hardware constraints.

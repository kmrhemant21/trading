# Practice activity: Fine-tuning an LLM

## Introduction
In this activity, you will fine-tune a pretrained large language model (LLM) on a task-specific dataset. Fine-tuning allows you to adapt the general capabilities of an LLM to a specific task, such as text classification, summarization, or sentiment analysis. By the end of this activity, you will have hands-on experience with fine-tuning techniques and be able to evaluate the model's performance on your custom task.

By the end of the activity, you will be able to: 

- Set up a fine-tuning environment using a pretrained LLM.
- Fine-tune the model on your task-specific dataset.
- Evaluate the performance of the fine-tuned model using appropriate metrics.

## Step-by-step guide to fine-tuning an LLM
This reading will guide you through the following steps:

1. Step 1: Set up the environment
2. Step 2: Prepare the dataset
3. Step 3: Fine-tune the LLM
4. Step 4: Evaluate the fine-tuned model
5. Step 5: Optional—experiment with hyperparameters

### Step 1: Set up the environment
Before starting the fine-tuning process, make sure you have the necessary environment set up. You will need access to a pretrained LLM (such as GPT, BERT, or T5) and a task-specific dataset.

**Instructions**
1. Ensure that you have the necessary libraries installed (e.g., Hugging Face Transformers, PyTorch, or TensorFlow).
2. Download or load the pretrained model from the Hugging Face Model Hub or any other repository.

**Code example**
```python
# Install necessary libraries
!pip install transformers datasets

# Import relevant modules
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load a pretrained model and tokenizer (e.g., BERT for sequence classification)
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

Note: Depending on your task, you can select other pretrained models such as GPT-2 for text generation or T5 for summarization.

### Step 2: Prepare the dataset
In this step, you will prepare your dataset for fine-tuning. Depending on your task, you can reuse the same datasets from previous activities, such as those used in classification or sentiment analysis tasks, or choose a new dataset. If you prefer to explore new datasets, consider using a dataset from platforms such as Hugging Face's dataset library, Kaggle, or another dataset source. For consistency with the previous lesson, we will split the dataset into training, validation, and test sets. The validation set helps to monitor the model's performance during training, and the test set is used to evaluate the final performance after training is complete.

In this activity, you will split the dataset into training and test sets only to simplify the process.

**Instructions**
1. Load your dataset, ensuring that it is properly formatted for the task (e.g., text and labels for classification).
2. Split your dataset into training and test sets.
3. Preprocess the text using the tokenizer of the pretrained model.

**Code example**
```python
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the IMDb dataset
dataset = load_dataset('imdb')

# Convert dataset to Pandas DataFrame
df = pd.DataFrame(dataset['train'])

# Perform train-test split
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Tokenize the dataset
def preprocess_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

# Convert back to Hugging Face dataset
from datasets import Dataset
train_data = Dataset.from_pandas(train_data)
test_data = Dataset.from_pandas(test_data)

# Apply preprocessing
train_data = train_data.map(preprocess_function, batched=True)
test_data = test_data.map(preprocess_function, batched=True)
```

### Step 3: Fine-tune the LLM
Now, you will fine-tune the pretrained LLM on your dataset. This process involves training the model on the task-specific data and optimizing its parameters for your task.

**Instructions**
1. Set up the training arguments, such as learning rate, number of epochs, and batch size.
2. Fine-tune the model on the training data while validating it on the validation set.

**Code example**
```python
import os
import torch
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification

# Disable parallelism warning and MLflow logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MLFLOW_TRACKING_URI"] = "disable"
os.environ["HF_MLFLOW_LOGGING"] = "false"

# Ensure CPU usage if no GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a smaller, faster model like DistilBERT
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model.to(device)

# Use a subset of the dataset to speed up training
train_data = train_data.select(range(1000))  # Select 1000 samples for training
test_data = test_data.select(range(200))     # Select 200 samples for evaluation

# Set up training arguments for faster training
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="steps",
    eval_steps=500,
    learning_rate=2e-5,
    per_device_train_batch_size=8,   
    num_train_epochs=1,              
    weight_decay=0,                  
    logging_steps=500,               
    save_steps=1000,                 
    save_total_limit=1,              
    gradient_accumulation_steps=1,   
    fp16=False,                      
    report_to="none",                
)

# Define the Trainer for fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
)

# Fine-tune the model
trainer.train()
```

### Step 4: Evaluate the fine-tuned model
Once the model is fine-tuned, it's essential to evaluate its performance on the test set using appropriate metrics, such as accuracy, precision, recall, and F1 score.

**Instructions**
1. Use the test dataset to evaluate the model's performance.
2. Calculate the evaluation metrics and analyze the results.

**Code example**
```python
# Evaluate the model
results = trainer.evaluate()

# Print evaluation results
print(f"Accuracy: {results['eval_accuracy']}")
```

### Step 5: Optional—experiment with hyperparameters
To further improve the performance of your fine-tuned model, you can experiment with different hyperparameters, such as learning rate, batch size, and number of epochs. You can also apply techniques such as hyperparameter search to find the best configuration for your task.

**Optional task**
- Adjust the learning rate or batch size and rerun the fine-tuning process to see whether performance improves.

## Conclusion
Fine-tuning a pretrained LLM allows you to leverage the power of existing models while customizing them for your specific task. Through this activity, you'll gain hands-on experience in fine-tuning, which is a critical skill in applying LLMs to real-world tasks.

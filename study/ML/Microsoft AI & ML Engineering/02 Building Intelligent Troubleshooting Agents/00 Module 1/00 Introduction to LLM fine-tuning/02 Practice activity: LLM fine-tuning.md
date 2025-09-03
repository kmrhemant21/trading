# Practice Activity: LLM Fine-tuning

> **Disclaimer:**  
> Resource-intensive tasks such as model training are involved in these activities. If you are using outdated hardware or systems with limited processing power, tasks might take 30 to 90 minutes. For a smoother experience, consider using cloud-based resources or modern hardware optimized for machine learning workloads.

## Introduction

In this activity, you’ll practice fine-tuning a large language model (LLM) using a simulated task. You will learn to prepare a dataset, configure the model, adjust hyperparameters, and deploy the fine-tuned model for sentiment analysis. Detailed commentary is provided in each code snippet to explain the steps and real-world applications.

**By the end of this activity, you will be able to:**

- Prepare and split task-specific datasets.
- Set up an Azure environment for LLM fine-tuning.
- Fine-tune a pretrained model for sentiment classification.
- Evaluate and deploy the model for real-time sentiment analysis.
- Interpret evaluation metrics such as accuracy and F1 score.

## Scenario

Imagine you work for a healthcare organization tasked with analyzing patient feedback surveys. Your objective is to fine-tune an LLM to automatically identify key sentiments and flag negative feedback for review by healthcare specialists.

## Step-by-Step Process for Fine-Tuning LLMs

### Step 1: Prepare the Dataset

**Dataset Collection:**  
Gather a dataset of anonymized patient feedback, categorized by sentiment—positive, neutral, and negative. Preprocess the data by cleaning, tokenizing, and splitting it.

**Code Example: Data Cleaning and Tokenization**

```python
import pandas as pd
import re
from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Create a noisy dataset
data_dict = {
    "text": [
        "  The staff was very kind and attentive to my needs!!!  ",
        "The waiting time was too long, and the staff was rude. Visit us at http://hospitalreviews.com",
        "The doctor answered all my questions...but the facility was outdated.   ",
        "The nurse was compassionate & made me feel comfortable!! :) ",
        "I had to wait over an hour before being seen.  Unacceptable service! #frustrated",
        "The check-in process was smooth, but the doctor seemed rushed. Visit https://feedback.com",
        "Everyone I interacted with was professional and helpful.  "
    ],
    "label": ["positive", "negative", "neutral", "positive", "negative", "neutral", "positive"]
}

# Convert dataset to a DataFrame
data = pd.DataFrame(data_dict)

# Clean the text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

data["cleaned_text"] = data["text"].apply(clean_text)

# Convert labels to integers
label_map = {"positive": 0, "neutral": 1, "negative": 2}
data["label"] = data["label"].map(label_map)

# Tokenize the cleaned text
data['tokenized'] = data['cleaned_text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

# Pad or truncate to fixed length (e.g., 128 tokens)
data['padded_tokenized'] = data['tokenized'].apply(
    lambda x: x + [tokenizer.pad_token_id] * (128 - len(x)) if len(x) < 128 else x[:128]
)

# Preview cleaned and labeled data
print(data[['cleaned_text', 'label', 'padded_tokenized']].head())
```

### Step 2: Split the Dataset

Divide the dataset into training, validation, and test sets. This helps in training, hyperparameter tuning, and unbiased evaluation respectively.

**Code Example: Data Splitting**

```python
from sklearn.model_selection import train_test_split

# Split data: 70% training, 15% validation, 15% test
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print(f"Training Size: {len(train_data)}, Validation Size: {len(val_data)}, Test Size: {len(test_data)}")
```

### Step 3: Set Up the Environment

Fine-tune the model in an environment equipped with GPU/TPU resources. We will use a pretrained BERT model configured for sentiment classification.

**Additional GPU Setup Instructions:**

- **Cloud Environments:** Use platforms like Google Colab or AWS SageMaker.
- **Local Environments:** Install necessary libraries and configure CUDA for GPU.

**Code Example: Loading the Pretrained Model and Preparing Datasets**

```python
from datasets import Dataset

# Convert DataFrame to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)
test_dataset = Dataset.from_pandas(test_data)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["cleaned_text"], padding="max_length", truncation=True, max_length=128)

# Tokenize the datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Remove unnecessary columns
columns_to_remove = ["text", "cleaned_text"]
train_dataset = train_dataset.remove_columns(columns_to_remove)
val_dataset = val_dataset.remove_columns(columns_to_remove)
test_dataset = test_dataset.remove_columns(columns_to_remove)

# Convert labels to integers if necessary
train_dataset = train_dataset.map(lambda x: {"label": int(x["label"])})
val_dataset = val_dataset.map(lambda x: {"label": int(x["label"])})
test_dataset = test_dataset.map(lambda x: {"label": int(x["label"])})

# Print a sample to confirm input_ids exist
print(train_dataset[0])
```

### Step 4: Configure Hyperparameters

Define hyperparameters such as learning rate and batch size to control the training process.  
A low learning rate (typically between 1e-5 and 5e-5) is crucial for gradual updates while fine-tuning.

**Code Example: Defining Training Parameters**

```python
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# Load pre-trained BERT model configured with 3 labels
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

training_args = TrainingArguments(
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    output_dir='./results',
    evaluation_strategy="epoch",
    logging_strategy="epoch",  
    logging_dir='./logs',  
    save_strategy="epoch",  
    load_best_model_at_end=True 
)

# Explanation:  
# The 'evaluation_strategy' dictates that the model is evaluated after every training epoch.
```

### Step 5: Fine-Tune the Model

Train the model using the prepared datasets while monitoring its progress.

**Code Example: Fine-Tuning Process**

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset.with_format("torch", columns=["input_ids", "attention_mask", "label"]),
    eval_dataset=val_dataset.with_format("torch", columns=["input_ids", "attention_mask", "label"])
)

# Start training
trainer.train()
```

### Step 6: Evaluate the Model

Evaluate the fine-tuned model on the test set using metrics like accuracy and F1 score.

**Code Example: Model Evaluation**

```python
from sklearn.metrics import accuracy_score, f1_score

# Prepare test dataset
test_dataset = test_dataset.with_format("torch", columns=["input_ids", "attention_mask", "label"])

# Generate predictions
predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)
labels = test_dataset["label"]

# Calculate metrics
accuracy = accuracy_score(labels, preds)
f1 = f1_score(labels, preds, average="weighted")

print(f"Accuracy: {accuracy}, F1 Score: {f1}")

# Explanation:  
# High F1 scores indicate balanced performance across all classes, which is essential for sentiment analysis.
```

### Step 7: Deploy the Model

Save and deploy the fine-tuned model for real-time sentiment analysis.

**Code Example: Saving and Deploying the Model**

```python
# Save the model and tokenizer
model.save_pretrained("./fine_tuned_bert")
tokenizer.save_pretrained("./fine_tuned_bert")
print("Model saved successfully!")
```

## Conclusion

This activity covers the complete process of fine-tuning an LLM for sentiment analysis. You learned how to clean and tokenize data, split datasets, configure a GPU environment, set hyperparameters, train and evaluate the model, and finally deploy it. With these skills, you can adapt LLMs for a variety of specialized applications across different industries.

# Walkthrough: Model and dataset selection (Optional)

## Introduction
Preprocessing datasets is a critical step in ensuring high-quality input for machine learning models. Cleaning and organizing the data, tokenizing text, and handling missing values are essential for preparing data to be used in fine-tuning tasks. This walkthrough will guide you through the steps needed to preprocess raw text data and structure it for a fine-tuning task.

By the end of this walkthrough, you will be able to:

- Clean and preprocess raw text data for machine learning tasks.
- Apply tokenization and text normalization techniques.
- Prepare your dataset for fine-tuning in a structured way.

## Step-by-step guide to preprocessing data
This reading will guide you through the following steps:

1. Step 1: Data Preprocessing
2. Step 2: Clean the text 
3. Step 3: Tokenize
4. Step 4: Handle missing data
5. Step 5: Prepare the data for fine-tuning
6. Step 6: Split the data

### Step 1: Data Preprocessing
 Before diving into the cleaning and tokenization processes, it's essential to import and organize the raw data into a structured format. We begin by loading the dataset, defining necessary labels, and preparing the initial dataset.  

```python
import pandas as pd
import torch
data = pd.read_csv('customer_data.csv')

# Define mapping for labels
label_mapping = {'Bronze': 0, 'Silver': 1, 'Gold': 2}  # Assign numbers to each category

# Convert membership_level to numeric labels
data['label'] = data['membership_level'].map(label_mapping)

# Convert labels to PyTorch tensor
labels = torch.tensor(data['label'].tolist())
data['cleaned_text'] = ["Hello, I am a Bronze member!", 
                        "Silver membership offers perks.", 
                        "Gold members get premium benefits.", 
                        "Silver members enjoy discounts.", 
                        "Bronze is the starting tier."]
```

### Step 2: Clean the text 
Text cleaning is the first step in preparing your dataset. It involves removing unwanted characters, URLs, and excess whitespace to ensure uniformity and cleanliness in the data. Text is also changed to lowercase to maintain consistency across all data points.

#### Example code
```python
import re

# Function to clean the text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Apply cleaning function to your dataset
data['cleaned_text'] = data['text'].apply(clean_text)
print(data['cleaned_text'].head())
```

#### Explanation
Cleaning the text by removing unnecessary characters and formatting it ensures that the data is consistent, making it easier for the model to understand.

### Step 3: Tokenize
Tokenization is the process of converting text into individual tokens that a machine-learning model can understand. We use the tokenizer corresponding to the pretrained model (e.g., BERT) for this. This ensures that the data is properly formatted and ready for fine-tuning.

#### Example code
```python
from transformers import BertTokenizer

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the cleaned text
tokens = tokenizer(
    data['cleaned_text'].tolist(), padding=True, truncation=True, return_tensors='pt', max_length=128
)

print(tokens['input_ids'][:5])  # Check the first 5 tokenized examples
```

#### Explanation
Tokenization converts the cleaned text into a format suitable for fine-tuning the model, ensuring that the input is ready for training.

### Step 4: Handle missing data
Missing data is common in real-world datasets. You can handle it either by removing incomplete entries or by imputing missing values. This step is critical to preventing errors during the training process.

#### Example code
```python
# Check for missing data
print(data.isnull().sum())

# Option 1: Drop rows with missing data
data = data.dropna()

# Option 2: Fill missing values with a placeholder
data['cleaned_text'].fillna('missing', inplace=True)
```

#### Explanation
Handling missing data ensures that your dataset is complete, which prevents training interruptions or biases introduced by missing information.

### Step 5: Prepare the data for fine-tuning
After cleaning and tokenizing your text, the next step is to prepare the data for fine-tuning. This involves structuring the tokenized data and labels into a format suitable for training, such as PyTorch DataLoader objects.

#### Example code
```python
import torch
from torch.utils.data import TensorDataset, DataLoader

# Create PyTorch tensors from the tokenized data
input_ids = tokens['input_ids']
attention_masks = tokens['attention_mask']
labels = torch.tensor(data['label'].tolist())

# Create a DataLoader for training
dataset = TensorDataset(input_ids, attention_masks, labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

print("DataLoader created successfully!")
```

#### Explanation
Organizing your data into DataLoader objects is necessary for model training, allowing the model to process the data in batches efficiently.

### Step 6: Split the data
Before training, it's important to split your data into training, validation, and test sets. The training set is used to train the model, the validation set helps to tune model hyperparameters, and the test set is used for final evaluation to ensure that the model generalizes well to unseen data.

#### Example code
```python
from sklearn.model_selection import train_test_split

# First, split data into a combined training + validation set and a test set
train_val_inputs, test_inputs, train_val_labels, test_labels = train_test_split(
    input_ids, labels, test_size=0.1, random_state=42
)

# Now, split the combined set into training and validation sets
train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    train_val_inputs, train_val_labels, test_size=0.15, random_state=42
)

# Create DataLoader objects for training, validation, and test sets
train_dataset = TensorDataset(train_inputs, train_labels)
val_dataset = TensorDataset(val_inputs, val_labels)
test_dataset = TensorDataset(test_inputs, test_labels)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)
test_dataloader = DataLoader(test_dataset, batch_size=16)

print("DataLoader objects for training, validation, and test sets created successfully!")
```

#### Explanation
The train_test_split method from the sklearn.model_selection module splits your data into training and validation (or test) sets. Here's a breakdown of how it works:

- input_ids and labels: these are the inputs and labels you are splitting.
- test_size=0.1: this indicates that 10 percent of the data will be set aside for the test set.
- random_state=42: this ensures the split is reproducible—using the same random state will produce the same split every time.

In this case, we first split the data into two sets:

- train_val_inputs and test_inputs: a combined set of training + validation data and a test set.

Then, we further split the train_val_inputs into train_inputs and val_inputs to get a separate validation set.

This process allows us to train, validate, and test data.

## Conclusion
Following this walkthrough, you've cleaned, tokenized, and structured your dataset for fine-tuning. With clean and well-prepared data, your model will have the best chance of achieving high performance during fine-tuning. You can use these preprocessing steps in your machine-learning projects to ensure optimal results.

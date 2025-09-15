# Practice activity: Preparing a dataset for fine-tuning

## Introduction

Have you ever wondered how to turn raw data into something your machine learning model can use? In this guide, we'll cover how to take unstructured text data, clean it up, and prepare it for fine-tuning, turning it into a powerful asset for your AI projects.

By the end of this reading, you'll be able to:

- Efficiently pre-process your own data, ensuring it's ready for training and optimized for success in real-world tasks.
- Prepare a dataset for fine-tuning.
- Clean the raw data, tokenize the text, handle missing data, and structure it into a training-ready input for a fine-tuning task.

## Instructions for preparing a dataset for fine-tuning

Create a new Jupyter notebook. You can call it "preparing_for_fine_tuning". Make sure you have the appropriate Python kernel selected.

The remaining of this reading will guide you through the following steps:

1. Step 1: Importing your dataset
2. Step 2: Clean the text
3. Step 3: Handle missing data
4. Step 4: Tokenization
5. Step 5: Structure data for fine-tuning
6. Step 6: Split the dataset

### Step 1: Import data set

You can download the Tweet emotion intensity dataset from Hugging Face into your environment.

Import the file and print out the first few lines of it.

The following code snippet will help you load the dataset:

#### Code example for loading dataset

```python
# Install modules
# A '!' in a Jupyter Notebook runs the line in the system's shell, and not in the Python interpreter

# Import necessary libraries
import pandas as pd
import random

# Load dataset 
# you can download this dataset from https://huggingface.co/datasets/stepp1/tweet_emotion_intensity/tree/main
data = pd.read_csv('data/tweet_emotion_intensity.csv')

# Preview the data
print(data.head())
```

Once the dataset is loaded, you will use it throughout the rest of the guide to implement each of the steps involved in preparing a dataset for fine-tuning an LLM.

#### Types of data

**Labeled data**: the dataset must include labeled data for supervised learning tasks such as sentiment analysis. In our case, the IMDB dataset is labeled with sentiment classes, such as "positive" or "negative" reviews.

**Unlabeled data**: unlabeled data can be used in unsupervised learning tasks or semi-supervised learning models. For this example, however, we will focus on labeled data for fine-tuning.

#### Sources of data

**Public datasets**: many open-source datasets are available for various natural language processing (NLP) tasks. Here are a few examples:

- IMDB Movie Reviews: a large dataset for sentiment analysis, labeled as positive or negative reviews
- SQuAD: a dataset for question-answering tasks
- AG News: a dataset for text classification (e.g., categorizing news articles into topics)

**Proprietary data**: you might need proprietary datasets if you're working on a specific task in a specialized domain. For example, a healthcare LLM model might use electronic health records (EHRs) data, while a retail model could rely on customer feedback or transaction data.

Starting with a clear and well-organized dataset sets a solid foundation for the fine-tuning process. Following this example with the IMDB dataset will help you practice the steps outlined in this guide. 

### Step 2: Clean the text

This step is cleaning the raw text data to remove unnecessary characters, such as URLs, special symbols, or HTML tags, and to normalize the text by converting it to lowercase. 

Make a new column called cleanedText that is equal to the data in the Tweet column that has had this cleanedText function applied to it.

#### Code example for text cleaning

```python
import re # Import the `re` module for working with regular expressions

# Function to clean the text
def clean_text(text):
    text = text.lower() # Convert all text to lowercase for uniformity
    text = re.sub(r'http\S+', '', text) # Remove URLs from the text
    text = re.sub(r'<.*?>', '', text) # Remove any HTML tags from the text
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation, keep only words and spaces
    return text # Return the cleaned text

# Assume `data` is a pandas DataFrame with a column named 'text'
# Apply the cleaning function to each row of the 'text' column
data['cleaned_text'] = data['tweet'].apply(clean_text)

# Print the first 5 rows of the cleaned text to verify the cleaning process
print(data['cleaned_text'].head())
```

### Step 3: Handle missing data

We now handle missing or incomplete data in your dataset. You can either remove rows with missing data or fill them with placeholders, ensuring the dataset is complete for training. 

#### Code example for handling missing data

```python
# Check for missing values in the dataset
print(data.isnull().sum()) # Print the count of missing values for each column

# Option 1: Remove rows with missing data in the 'cleaned_text' column
data = data.dropna(subset=['cleaned_text']) # Drop rows where 'cleaned_text' is NaN (missing)

# Option 2: Fill missing values in 'cleaned_text' with a placeholder
data['cleaned_text'].fillna('unknown', inplace=True) # Replace NaN values in 'cleaned_text' with 'unknown'
```

### Step 4: Tokenization

After cleaning the text, we tokenize it. Tokenization splits the text into individual words or subwords that can be used by the model. We will use the BERT tokenizer to ensure compatibility with the Brie-trained model you are fine-tuning. 

#### Code example for tokenization

```python
from transformers import BertTokenizer

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the cleaned text
tokens = tokenizer(
    data['cleaned_text'].tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt'
)

print(tokens['input_ids'][:5])  # Preview the first 5 tokenized examples
```

We load the BERT tokenizer from the Transformers library from HuggingFace, and tokenize the cleanedText that we defined earlier. Then, we can print the tokens of the input IDs. The words have been converted into these numbered tokens. 

In certain cases, especially when data is limited, data augmentation techniques can be applied to generate new training examples by modifying the original dataset.

- Paraphrasing: rewriting sentences in different ways while preserving the meaning
- Backtranslation: translating text into another language and back again to create variation
- Synonym replacement: replacing certain words in the text with their synonyms

#### Code example for synonym replacement (augmentation)

The following example demonstrates how to implement synonym replacement using the nltk library. It randomly replaces words in the text with their synonyms to create new variations of sentences. This method can be applied when paraphrasing or backtranslation is not feasible.

```python
# Import necessary modules
import random # Random module for generating random numbers and selections
from nltk.corpus import wordnet # NLTK's WordNet corpus for finding synonyms

# Define a function to find and replace a word with a synonym
def synonym_replacement(word):
# Get all synsets (sets of synonyms) for the given word from WordNet
    synonyms = wordnet.synsets(word)

# If the word has synonyms, randomly choose one synonym, otherwise return the original word
    if synonyms:
# Select a random synonym and get the first lemma (word form) of that synonym
        return random.choice(synonyms).lemmas()[0].name()

# If no synonyms are found, return the original word
    return word

# Define a function to augment text by replacing words with synonyms randomly
def augment_text(text):
# Split the input text into individual words
    words = text.split() # Split the input text into individual words

# Replace each word with a synonym with a probability of 20% (random.random() > 0.8)
    augmented_words = [
    synonym_replacement(word) if random.random() > 0.8 else word 
# If random condition met, replace
for word in words] # Iterate over each word in the original text

# Join the augmented words back into a single string and return it
    return ' '.join(augmented_words)

# Apply the text augmentation function to the 'cleaned_text' column in a DataFrame
# Create a new column 'augmented_text' containing the augmented version of 'cleaned_text'
data['augmented_text'] = data['cleaned_text'].apply(augment_text)
```

##### Explanation of code

**synonym_replacement**: this function uses the nltk library's wordnet to retrieve synonyms of a given word. If synonyms are available, it randomly selects one. If not, the original word is returned.

**augment_text**: this function iterates through each word in the text, replacing it with a synonym based on a random probability (here, a 20 percent chance for each word).

**Applying augmentation**: we apply the augment_text function to the cleaned text in the dataset, creating a new column, augmented_text, which contains the augmented text samples.

### Step 5: Structure the data for fine-tuning

You can fine-tune your model once the dataset is cleaned and tokenized. The next step is structuring the data for fine-tuning. 

Import Torch, TensorDataset and DataLoader. We will convert the tokens into PyTorch tensors. We will define a mapping function that sets the tweet sentiment intensity from high to 1, from medium to 0.5, and from low to 0. Then, we will apply that function to each item in sentiment_intensity, and then we will drop any rows where sentiment_intensity is none, where sentiment_intensity was something other than high, medium, or low. Finally, we will convert the sentiment_intensity column to a tensor.

#### Code example for DataLoaders

```python
import torch # Import PyTorch library
from torch.utils.data import TensorDataset, DataLoader # Import modules to create datasets and data loaders

# Convert tokenized data into PyTorch tensors
input_ids = tokens['input_ids'] # Extract input IDs from the tokenized data
attention_masks = tokens['attention_mask'] # Extract attention masks from the tokenized data

# Define a mapping function
def map_sentiment(value):
    if value == "high":
        return 1
    elif value == "medium":
        return 0.5
    elif value == "low":
        return 0
    else:
        return None  # Handle unexpected values, if any

# Apply the function to each item in 'sentiment_intensity'
data['sentiment_intensity'] = data['sentiment_intensity'].apply(map_sentiment)

# Drop any rows where 'sentiment_intensity' is None
data = data.dropna(subset=['sentiment_intensity']).reset_index(drop=True)

# Convert the 'sentiment_intensity' column to a tensor
labels = torch.tensor(data['sentiment_intensity'].tolist())
```

Following these steps, your dataset will be appropriately cleaned, tokenized, and structured for fine-tuning. A well-prepared dataset is crucial for achieving high performance and ensuring your model generalizes well to new data.

### Step 6: Split the Dataset

Finally, we split the dataset into training, validation, and test sets. This ensures that your model is trained on one portion of the data while its performance is monitored and tested on unseen examples. This includes organizing the tokenized data into PyTorch TensorDataset objects, ready for training.

#### Code example 

```python
from sklearn.model_selection import train_test_split # Import function to split dataset

# First split: 15% for test set, the rest for training/validation
train_val_inputs, test_inputs, train_val_masks, test_masks, train_val_labels, test_labels = train_test_split(
    input_ids, attention_masks, labels, test_size=0.15, random_state=42
)

# Second split: 20% for validation set from remaining data
train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = train_test_split(
    train_val_inputs, train_val_masks, train_val_labels, test_size=0.2, random_state=42
)

# Create TensorDataset objects for each set, including attention masks
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
val_dataset = TensorDataset(val_inputs, val_masks, val_labels)
test_dataset = TensorDataset(test_inputs, test_masks, test_labels)

# Create DataLoader objects
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)
test_dataloader = DataLoader(test_dataset, batch_size=16)

print("Training, validation, and test sets are prepared with attention masks!")
```

## Conclusion

A well-prepared dataset is the foundation of a successful fine-tuning process for large language models. By carefully collecting, cleaning, and tokenizing the data, you ensure that your model learns from high-quality inputs and generalizes well to unseen data. Additionally, using augmentation techniques when appropriate can further improve model performance. These steps will set you on the right path toward achieving optimal results in your fine-tuning efforts.
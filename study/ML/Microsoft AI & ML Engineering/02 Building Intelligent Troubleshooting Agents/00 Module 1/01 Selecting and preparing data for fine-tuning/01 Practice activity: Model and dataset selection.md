# Practice activity: Model and dataset selection

## Introduction
Selecting the right model and dataset is crucial for ensuring the success of a fine-tuning task. Your choices at this stage will directly affect the model's performance and ability to generalize well on new data. This section will guide you through the key considerations and practical steps for choosing the appropriate model and dataset for fine-tuning.

By the end of this activity, you will be able to:

- Define the task requirements and select an appropriate model architecture (e.g., text classification, text generation, or question answering).
- Choose a suitable pretrained language model for fine-tuning based on task complexity, model size, and resource constraints.
- Select, curate, and prepare task-specific datasets that align with the model's goals, ensuring proper data quality, size, and balance.
- Preprocess datasets by cleaning, tokenizing, and balancing the data for optimal fine-tuning results.
- Evaluate the initial model performance and refine the dataset as necessary before beginning fine-tuning.

## Step-by-step process for model and dataset selection
This reading will guide you through the following steps:

1. Step 1: Define the task and requirements
2. Step 2: Choose the pretrained model
3. Step 3: Select the dataset
4. Step 4: Preprocess the dataset
5. Step 5: Evaluate and refine

### Step 1: Define the task and requirements
Before selecting a model or dataset, it's essential to clearly understand your task and its specific requirements. The task type will influence the model architecture and how the dataset is prepared.

#### Task type
Different tasks require different model architectures and may also require different approaches to dataset preparation. For example:

- **Text classification**: this task categorizes text into predefined labels (e.g., spam detection, sentiment analysis). Models such as BERT and RoBERTa are well suited for these tasks due to their contextual solid understanding. For text classification, you need a dataset labeled with categories, and you must ensure that the dataset is balanced across the categories to avoid biased predictions.

- **Text generation**: this task allows the model to generate new text (e.g., language translation, summarization). GPT-based models are ideal for these use cases because of their natural language generation capabilities. When preparing the dataset for text generation, it's crucial to ensure that the input–output pairs are aligned correctly, and long sequences may need to be truncated or padded appropriately.

- **Question answering**: this task allows the model to extract or generate answers from text (e.g., reading comprehension). Models such as BERT and T5 are well suited for sentence-level or passage-level understanding. For question-answering tasks, you need to ensure the dataset is properly formatted, often in a question-answer pair structure with relevant context passages included. The dataset should also be cleaned to remove irrelevant information that may confuse the model.

#### Task complexity
For more complex tasks, such as translating legal texts or analyzing medical records, you'll need a model capable of handling domain-specific language. Fine-tuning an existing pretrained model is typically more effective than training a model from scratch. When dealing with specialized tasks, the dataset must be curated to include domain-specific terminology and balanced to cover various cases within the field.

### Step 2: Choose the pretrained model
One of the most important decisions in the process is choosing a pretrained model for fine-tuning. The selected model should have a robust understanding of language and be adaptable to the task at hand.

#### Commonly used pretrained models
- **Bidirectional encoder representations from transformers (BERT)**: BERT is highly effective for such tasks as text classification, named entity recognition (NER), and question answering. It uses bidirectional attention to understand context from both sides of a word.

- **Generative pretrained transformer (GPT-3)**: GPT-3 is one of the most powerful models for text generation tasks, such as summarization or language translation. It's particularly effective at producing human-like text based on input prompts.

- **Robustly optimized BERT pretraining approach (RoBERTa)**: RoBERTa builds upon BERT but is optimized for better performance in such tasks as sentiment analysis, classification, and text completion.

- **Text-to-text transfer transformer (T5)**: T5 converts all NLP tasks into a text-to-text format, making it highly versatile for translation, summarization, and classification.

#### Model size
When selecting a model, consider the available computational resources and the complexity of your task. Larger models often perform better but require significantly more memory and processing power. Smaller models can be more practical for less resource-intensive tasks or when working with smaller datasets.

#### Quantifying task size
**Small tasks**: these typically involve datasets with fewer than 10,000 examples, short text inputs (e.g., less than 200 tokens), and relatively straightforward tasks, such as fundamental sentiment analysis or spam classification. Smaller models like BERT-base (110M parameters) or DistilBERT (66M parameters) are sufficient for such tasks. Within a few hours, these models can be fine-tuned with limited resources—such as a single GPU with 8–16GB of VRAM.

The following are examples of small tasks:

- Classifying customer reviews as positive or negative
- Detecting spam in emails with short message lengths
- Recognizing named entities in a few hundred documents

**Large tasks**: these involve more complex datasets with at least 100,000 examples, longer text sequences (at least 500 tokens), and tasks requiring deep contextual understanding, such as question answering, machine translation, or text generation. For these tasks, larger models such as GPT-3 (175B parameters) or T5-large (770M parameters) are necessary. Fine-tuning these models requires high-end computational resources, such as multiple GPUs or TPUs with at least 32GB VRAM, and can take days or weeks, depending on the dataset size.

The following are examples of large tasks:

- Generating long-form text, such as articles or stories, from input prompts
- Answering questions based on passages from legal or medical documents
- Translating paragraphs of text between languages

By understanding the task size, you can better gauge the appropriate model size, balancing performance and resource efficiency. For example, a small task like classifying tweets doesn't require the computational heft of GPT-3, whereas generating detailed, coherent answers to complex questions might.

#### Model availability
Before proceeding, ensure that the pretrained model you plan to fine-tune is available in your chosen framework, such as Hugging Face Transformers or TensorFlow Hub. These platforms provide access to the most popular pretrained models, prebuilt tokenizers, model configurations, and usage guides. This simplifies integration by allowing you to focus on fine-tuning rather than managing tokenization or model architecture from scratch.

#### Understanding model options
Each model has various input options and configurable parameters, such as num_labels in the example below, which affects how the model performs classification (e.g., the number of categories it can classify). Understanding these options is essential to fine-tune the model for your specific task effectively. For example, in a classification task, setting num_labels correctly ensures the model knows how many categories to classify into.

Refer to the [documentation](https://huggingface.co/transformers/) of the different models in the Hugging Face library to explore other methods, options, and parameters. This will give you detailed information on the available models, how to configure them, and the impact of different parameters on model performance. Note that we are using BERT in this example for its reliability and relative universality. Other models exist, and your employer may ask you to deploy one that is either commercially available or proprietary to the company.

**Code example (loading a pretrained model)**

```python
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained BERT model and tokenizer for classification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Model and tokenizer are now ready for fine-tuning
```

`num_labels=3`: this parameter specifies the number of labels (or classes) the model will classify into. For example, in a sentiment analysis task, `num_labels=2` might be used for binary classification (positive/negative), while a task with more categories would require a higher value.

Note that you will often find yourself in a situation in which content that does not fall into a binary is required. For example, email classification's ultimate output of "spam" or "not spam" nonetheless requires a detailed examination before num_labels classifies mail into its ultimate location. While this might seem as if the appropriate num_labels would be "2," in this example, know that you can examine and classify items more than once — perhaps into "spam," "not spam," and "further review required."

To find additional configuration options and better understand how each parameter affects the model, learners can explore the respective model documentation in Hugging Face or TensorFlow Hub. These sites provide detailed examples and explanations for fine-tuning various models.

### Step 3: Select the dataset
Choosing the correct dataset is just as critical as selecting the model. A well-curated dataset ensures that the model learns effectively and generalizes well to new tasks.

#### Task-specific datasets
The dataset should be aligned with the task requirements. For example, if you are fine-tuning a model for sentiment analysis, a dataset like the IMDB movie reviews dataset labeled for positive and negative sentiments would be appropriate. For a question-answering task, the SQuAD dataset provides excellent training material.

To help you explore more available datasets, such platforms as [Hugging Face Datasets](https://huggingface.co/datasets) offer an extensive repository of task-specific datasets that you can easily integrate into your workflow. This repository includes datasets for various NLP tasks, such as text classification, generation, and translation. 

In addition to Hugging Face, here are other resources for finding datasets:

- **Kaggle Datasets**: Kaggle provides many publicly available datasets across various domains. You can filter datasets based on task, size, and other criteria. [Kaggle Datasets](https://www.kaggle.com/datasets)

- **Google Dataset Search**: Google features a search engine for datasets covering various fields, from finance to health care and beyond. [Google Dataset Search](https://datasetsearch.research.google.com/)

- **UCI Machine Learning Repository**: University of California, Irvine's (UCI) repository offers a collection of datasets for machine learning tasks, including NLP and classification tasks. [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)

Exploring these repositories allows you to find task-specific datasets that meet your fine-tuning needs. Make sure the dataset is suitable for the task and properly labeled to ensure successful model fine-tuning.

#### Dataset size
The size of the dataset affects how well the model will fine-tune. Larger datasets generally allow the model to learn more comprehensive patterns. However, when carefully curated and balanced, smaller datasets can also yield strong results.

If you're working with a smaller dataset, consider whether it needs to be augmented to increase variety and improve model performance. Common augmentation techniques include paraphrasing (rewording the text while preserving the meaning) and backtranslation (translating the text to another language and back again to generate alternative expressions).

To help you implement dataset augmentation, here are some resources and code examples:

- **Hugging Face Datasets** provide built-in support for data transformations and augmentations. You can explore their documentation for augmentation techniques: [Hugging Face Dataset Documentation](https://huggingface.co/docs/datasets/index)

- **nlpaug** is a popular Python library for augmenting text datasets using such methods as paraphrasing, backtranslation, and synonym replacement. You can find the documentation and examples here: [nlpaug Documentation](https://github.com/makcedward/nlpaug)

**Code example for augmenting data using backtranslation**

```python
from nlpaug.augmenter.word import BackTranslationAug

# Initialize the backtranslation augmenter (English -> French -> English)
back_translation_aug = BackTranslationAug(from_model_name='facebook/wmt19-en-de', to_model_name='facebook/wmt19-de-en')

# Example text to augment
text = "The weather is great today."

# Perform backtranslation to create augmented text
augmented_text = back_translation_aug.augment(text)

print("Original text:", text)
print("Augmented text:", augmented_text)
```

These resources and examples provide a starting point for augmenting datasets, helping to improve model generalization and performance when working with smaller datasets.

#### Balancing the dataset
Class imbalance can significantly affect model performance. The model may become biased if certain labels are overrepresented in the dataset. To address this, you can:

- Undersample the majority class to balance the dataset.
- Oversample the minority class or generate synthetic examples using data augmentation techniques.
- Use class weighting during model training to give more importance to underrepresented classes.

**Code example (loading a dataset for fine-tuning)**
Before we dive into the code, it's important to ensure learners understand how to properly split datasets for training, testing, and validation. Most datasets come presplit (as in the IMDB dataset example), but in cases in which a dataset isn't presplit, you can use such functions as train_test_split from libraries such as Scikit-Learn to manually create these splits.

If you're unfamiliar with dataset splitting or want more control over how it's done, the Scikit-Learn documentation provides a thorough guide: [Scikit-Learn: train_test_split documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

Here's an updated version of the code that includes a manual splitting example:

```python
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Load the IMDB movie reviews dataset for sentiment analysis
dataset = load_dataset('imdb')

# Split the dataset into training and validation sets (if not presplit)
train_data, val_data = train_test_split(dataset['train'], test_size=0.2)

# Convert the data into the format required for tokenization
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_train = train_data.map(tokenize_function, batched=True)
tokenized_val = val_data.map(tokenize_function, batched=True)
```

#### Key concepts and options
**Dataset splitting**
When using `train_test_split`, you can specify how much data goes into training versus validation (e.g., `test_size=0.2` allocates 20 percent of the data to validation and 80 percent to training). If the dataset is already split (as in the original example with IMDB), you don't need to do this manually.

**Tokenization options**
The tokenizer method in the code has several input options, such as `padding='max_length'` and `truncation=True`, which ensure that all sequences have the same length. Here's a brief explanation:

- **Padding**: ensures that all input sequences are of equal length by padding shorter sequences with extra tokens.
- **Truncation**: ensures that input sequences exceeding the maximum length are truncated to fit within the model's requirements.

To explore additional input options for such methods as tokenizer, learners can refer to the [Hugging Face Transformers documentation](https://huggingface.co/transformers/main_classes/tokenizer.html), which provides detailed explanations of all available parameters.

### Step 4: Preprocess the dataset
After selecting the dataset, it's essential to preprocess the text to make it suitable for model input. Proper preprocessing ensures the data is in the right format for fine-tuning the model. Don't worry about the specifics of data cleaning at the moment (other than broad strokes); a more detailed discussion and examples follow this element.

- **Text cleaning**: ensure the text is free from noise, such as special characters, HTML tags, or unnecessary whitespace. Depending on the model and task, text normalization (e.g., lowercasing) may also be required.

- **Tokenization**: use the tokenizer corresponding to the pretrained model to break down the text into input tokens. For most transformer models, tokenization also includes padding sequences to a fixed length and truncating long sequences to ensure consistent input size.

**Code example (tokenizing and preparing the dataset for training)**

```python
# Tokenize the dataset
tokenized_train = tokenizer(
    train_data['text'], padding=True, truncation=True, return_tensors="pt"
)
tokenized_val = tokenizer(
    val_data['text'], padding=True, truncation=True, return_tensors="pt"
)
```

### Step 5: Evaluate and refine
Once you've selected the model and dataset, evaluate the initial performance of the dataset before starting fine-tuning.

- **Initial evaluation**: run the pretrained model on the dataset to get a baseline performance. This helps to measure the improvements made after fine-tuning.

- **Refinement**: ensure that the dataset is balanced correctly and formatted. Before fine-tuning, you should iterate on data cleaning or dataset balancing.

## Conclusion
Selecting and preparing the right model and dataset are crucial steps that significantly impact the success of a fine-tuning task. By clearly defining the task, choosing an appropriate pretrained model, and ensuring the dataset is high-quality and well-prepared, you lay a solid foundation for effective fine-tuning. Proper handling of imbalanced data, careful tokenization, and thorough preprocessing ensure the model learns relevant patterns and generalizes new data well. Evaluating the model's baseline performance before fine-tuning helps maximize the benefits of the process, leading to more accurate and efficient outcomes. These careful preparations are essential for developing reliable and high-performing AI/ML systems tailored to specialized tasks.

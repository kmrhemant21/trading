# Summary and Highlights

Congratulations! You have completed this lesson. At this point in the course, you know that:

## Tokenization and Data Loading

- **Tokenization and data loading** are part of the data preparation activities for natural language processing (NLP).

- **Tokenization** breaks a sentence into smaller pieces or tokens.

- **Tokenizers** are essential tools that break down text into tokens. These tokens can be words, characters, or subwords, making complex text understandable to computers. Examples of tokenizers are:
    - Natural Language Toolkit (NLTK)
    - spaCy

## Types of Tokenization

- **Word-based tokenization** preserves the semantic meaning, though it increases the model's overall vocabulary.

- **Character-based tokenization** has smaller vocabularies but may not convey the same information as entire words.

- **Subword-based tokenization** allows frequently used words to stay unsplit while breaking down infrequent words.

### Subword Tokenization Algorithms
- WordPiece
- Unigram
- SentencePiece

## Special Tokens

You can add special tokens such as:
- `<bos>` at the beginning of a tokenized sentence
- `<eos>` at the end of a tokenized sentence

## PyTorch Data Handling

### Data Sets
A **data set** in PyTorch is an object that represents a collection of data samples. Each data sample typically consists of:
- One or more input features
- Corresponding target labels

### Data Loaders
A **data loader** helps you prepare and load data to train generative AI models. Benefits include:
- Output data in batches instead of one sample at a time
- Seamless integration with the PyTorch training pipeline
- Simplified data augmentation and preprocessing

#### Key Data Loader Parameters
- **Data set**: The source to load from
- **Batch size**: Determines how many samples per batch
- **Shuffle**: Whether to shuffle the data for each epoch
- **Iterator interface**: Makes it easy to iterate over batches of data during training

PyTorch provides a dedicated `DataLoader` class for this functionality.

### Collate Functions
A **collate function** is employed in the context of data loading and batching in machine learning, particularly when dealing with variable-length data such as:
- Text sequences
- Time series
- Sequences of events

Its primary purpose is to prepare and format individual data samples into batches that machine learning models can efficiently process.
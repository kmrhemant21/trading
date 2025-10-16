# Practice activity: Implementing sentiment analysis

## Introduction
Have you ever wondered how companies analyze customer reviews, track social media sentiment, or monitor brand reputation? Sentiment analysis makes this possible by automatically classifying text as positive, negative, or neutral, helping businesses to understand emotions at scale. In this activity, you'll learn how to implement a basic sentiment analysis model and see its impact on real-world applications. By the end of this activity, you will be able to create a sentiment analysis system that processes text and outputs sentiment classification.

By the end of this activity, you will:

- Describe how to implement a sentiment analysis model using a pre-trained pipeline.
- Input both predefined and custom text for a sentiment classification.
- And interpret the results and evaluate sentiment analysis in practical applications

## Step-by-step process to implement sentiment analysis
Create a new Jupyter notebook. You can call it "sentiment_analysis". Make sure you have the appropriate Python 3.8 Azure ML kernel selected.

The remaining of this reading will guide you through the following steps:

This reading will guide you through the following steps:

1. Step 1: Set up the environment
2. Step 2: Import libraries and load the pretrained model
3. Step 3: Input text for sentiment analysis
4. Step 4: Test with custom input
5. Step 5: Reflect on the results
6. Step 6: Enhance the sentiment analysis system

### Step 1: Set up the environment
**Instructions**  
Begin by setting up your Python development environment. If you don't have Python installed, download and install it from 
python.org. You will also need to install the transformers library from Hugging Face, which provides access to pretrained NLP models.

Open a terminal and install the necessary libraries using the following commands:

**Example setup**
```python
!pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
!pip install tokenizers==0.15.2
!pip install transformers==4.36.2
```

**Explanation**  
The transformers library contains pretrained models for various NLP tasks, including sentiment analysis. We'll be using a pretrained model in this activity to simplify the implementation.

### Step 2: Import libraries and load the pretrained model
**Instructions**  
After setting up the environment, import the necessary libraries and load a pretrained sentiment analysis model. For this activity, we'll use the BERT-based sentiment analysis model provided by Hugging Face's transformers library.

**Code example**
```python
from transformers import pipeline

# Initialize sentiment analyzer with specific model to avoid downloading issues
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

print("Sentiment analysis model loaded successfully!")
```

**Explanation**  
The pipeline function from the transformers library allows you to quickly load a pretrained model. In this case, we're initializing the sentiment analysis pipeline, which automatically loads a BERT-based model trained for this task. BERT is ideal for this application, but you should know that other language models may be used instead, though specific implementation for each will not be covered in this element.

### Step 3: Input text for sentiment analysis
**Instructions**  
Now that the sentiment analysis model is loaded, we'll input some sample text to classify. The model will process the text and return the sentiment (positive, negative, or neutral) along with a confidence score.

You can input any text for analysis, but here we'll start with some predefined examples.

**Code example**
```python
# Sample texts for sentiment analysis
texts = [
    "I love this product! It's amazing.",
    "The service was terrible and I'm very disappointed.",
    "It's okay, not great but not bad either."
]

# Analyze the sentiment of each text
for text in texts:
    result = sentiment_analyzer(text)
    print(f"Text: {text}")
    print(f"Sentiment: {result[0]['label']}")
    print(f"Confidence: {result[0]['score']:.2f}")
    print()  # Empty line for readability
```

**Explanation**  
In this example, the model processes each text and returns the sentiment label (positive, negative, or neutral) along with a confidence score that indicates how certain the model is about the classification.

### Step 4: Test with custom input
**Instructions**  
Now, let's allow the user to input their own text for sentiment analysis. You will modify the code to accept custom text input from the user.

**Code example update**
```python
# Accept user input for custom sentiment analysis
custom_text = input("Enter a sentence for sentiment analysis: ")

# Analyze the sentiment
result = sentiment_analyzer(custom_text)

print(f"\nSentiment: {result[0]['label']}")
print(f"Confidence: {result[0]['score']:.2f}")
```

**Explanation**  
With this update, users can type their own sentences and see the sentiment classification in real time, giving them an interactive way to experiment with the model.

### Step 5: Reflect on the results
**Instructions**  
Once you have tested the model with both predefined and custom input, take some time to reflect on the results. Consider how accurate the sentiment classifications are and whether the model handles more nuanced text (such as sarcasm or mixed sentiment) effectively.

**Reflection questions**
- How accurate was the model in classifying the sentiment of the text samples?
- Did the model correctly handle text with mixed or neutral sentiment?
- How could the sentiment analysis system be improved to handle more complex or ambiguous text?

**Example reflection**  
If the model misclassifies certain sentences or struggles with nuanced language, consider adding more training data or fine-tuning to enhance its performance. Once you're comfortable with BERT, you might explore other models with varying parameters and capabilities to address sentiment classification challenges.

### Step 6: Enhance the sentiment analysis system
**Instructions**  
For additional practice, try enhancing the system by adding more features, such as:

- Allowing the model to process longer paragraphs instead of single sentences.
- Storing the sentiment results in a file for further analysis.
- Integrating sentiment analysis into an application, such as a customer feedback system or social media monitoring tool.

**Example enhancement**
```python
# Allow the model to process a longer paragraph of text
long_text = """
The product is good overall, but there are some issues with battery life. 
I wish it lasted longer. However, the design is sleek, and I’m happy with the performance so far.
"""
result = sentiment_analyzer(long_text)
for res in result:
    print(f"Sentiment: {res['label']}, Confidence: {res['score']:.2f}")
```

**Explanation**  
By processing longer pieces of text, the model can handle more complex data, giving you deeper insights into overall sentiment, especially in use cases such as product reviews or opinion analysis.

## Conclusion
In this activity, you implemented a basic sentiment analysis system using a pretrained model from the transformers library. You learned how to load the model, input text, and classify it as positive, negative, or neutral. Sentiment analysis is a powerful tool for understanding emotions in text and can be applied to numerous real-world tasks, such as customer feedback analysis, brand monitoring, and social media tracking.

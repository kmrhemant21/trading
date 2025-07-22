# Apply Pretrained LLMs for Specific Tasks

## Introduction

This notebook demonstrates text generation, summarization, question answering, and chatbot interactions using `google/flan-t5-small`, a small transformer model optimized for efficiency. This model is suitable for learning but has limitations in coherence and depth compared to larger models like OpenAI's `text-davinci-003` (GPT-3). For real-world applications, it is recommended to use more powerful models like GPT-3 or GPT-4 via the OpenAI API, which provides better fluency, context understanding, and reasoning.

By the end of this reading, you will be able to:

- Deploy `google/flan-t5-small` for tasks like text generation and summarization.
- Create a basic conversational AI.

> **Note**  
> For the sake of demonstration, we are using a small, free model. If you need higher-quality responses, consider using GPT-3 or GPT-4 with the OpenAI API, but be aware of API key restrictions. For student exercises, the `flan-t5-small` model is a good starting point, and larger models like `flan-t5-base` can be used for better quality while still remaining open-source.

---

## Set Up the Environment

First things first, let’s set up our environment. Depending on the LLM you’re using, you’ll need to ensure that the required libraries and APIs are installed and accessible. For this demonstration, we'll be using `flan-t5-small`. Let’s start by installing the necessary packages.

```bash
# Install dependencies
!pip install transformers
```

With the library installed, we’re now ready to start deploying our pre-trained LLM for specific tasks.

---

## Simple Text Generation

Let’s start with a simple text generation task. We’ll generate a continuation of a given text prompt. This is particularly useful for applications like content creation, chatbots, and even creative writing.

```python
# Simple completion
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
```

As you can see, `flan-t5-small` generates a continuation of our prompt. The `max_length` parameter controls the length of the generated text, allowing you to fine-tune the output to your needs.

---

## Summarization

Next, let’s use `flan-t5-small` for summarization. This is an invaluable tool for condensing long articles, reports, or any other text into a concise summary.

```python
# Summarization example
text = """NASA's Perseverance rover successfully landed on Mars as part of the Mars Exploration Program.
It is designed to search for signs of ancient life, collect rock samples, and prepare for future missions."""
summary = generator(f"Summarize: {text}", max_length=50, min_length=20, do_sample=False)
print(summary[0]["generated_text"])
```

Here, `flan-t5-small` has distilled the key points from a paragraph into a brief summary. This feature is especially useful in contexts like news aggregation, where quick and accurate summaries are needed.

---

## Question and Answer

Another powerful application of LLMs is answering questions based on a given context. This capability is ideal for developing AI-powered customer support systems, educational tools, and more.

```python
# Question and answer example
question = (
    "The capital of France is Paris. "
    "The Eiffel Tower is located in Paris.\n\n"
    "Question: Where is the Eiffel Tower located?"
)

response = generator(question, max_length=50)
print(response[0]["generated_text"])
```

`flan-t5-small` quickly answers the question based on the context provided. This type of task showcases the model's ability to understand and process information in a way that’s useful for real-time applications that use context for Retrieval Augmented Generation (RAG).

---

## Basic Conversational AI

Finally, let’s create a basic conversational AI. This model can be the backbone of chatbots used in customer service, virtual assistants, and more.

```python
# Basic chatbot
chatbot_prompt = "You are a friendly AI assistant. Answer the user’s question with a helpful response."
messages = [{"role": "user", "content": "Tell me a fact about the Sun."}]
response = generator(f"{chatbot_prompt} {messages[-1]['content']}", max_length=50)
print(response[0]["generated_text"])
```

This script allows us to maintain a conversation history to generate more contextually relevant responses. This is a great starting point for building more sophisticated conversational agents.

---

## Limitations of `flan-t5-small` and When to Use Larger Models

- The responses may sometimes be generic or incorrect.
- The model has a limited ability to track conversation history.
- GPT-3 (`text-davinci-003`) or GPT-4 via OpenAI API provides more accurate, fluent, and coherent text generation.
- **API Key Requirement**: OpenAI’s models require an API key, making them less ideal for classroom settings where students cannot use a personal API key.

---

## Conclusion

And there you have it! We’ve explored how to use an open-source LLM like `flan-t5-small` for a variety of tasks, from text generation and summarization to question answering and conversational AI. These examples are just the beginning—these models can be adapted and fine-tuned for countless applications. Now go ahead, deploy `flan-t5-small` (or a more powerful LLM) for tasks like text generation and summarization, and start building your own conversational AI using these powerful tools.
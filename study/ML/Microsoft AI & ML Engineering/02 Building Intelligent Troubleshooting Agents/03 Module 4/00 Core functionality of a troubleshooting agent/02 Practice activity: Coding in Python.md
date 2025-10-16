# Practice activity: Coding in Python

## Introduction
Imagine being able to solve technical issues in real time, without relying on customer service. In this activity, you will learn how to build a simple troubleshooting agent using Python. The agent will process user queries, identify problems, and suggest solutions based on natural language processing (NLP) techniques. You will implement key components such as tokenization, part-of-speech (POS) tagging, named entity recognition (NER), and sentiment analysis to build the agent's core functionality.

By the end of this activity, you will:

- Implement a basic troubleshooting agent using Python.
- Use NLP techniques such as tokenization, POS tagging, and NER to process user queries.
- Incorporate sentiment analysis to detect user frustration and guide problem prioritization.

## Step-by-step process to code a troubleshooting agent in Python
Create a new Jupyter notebook. Make sure you have the appropriate Python 3.8 Azure ML kernel selected.

The remaining of this reading will guide you through the following steps:

1. Step 1: Setting up your environment
2. Step 2: Tokenizing the user query
3. Step 3: Applying part-of-speech tagging
4. Step 4: Extracting key entities with named entity recognition
5. Step 5: Analyzing sentiment for prioritization
6. Step 6: Building a knowledge base for problem-solving
7. Step 7: Implementing decision-making logic
8. Step 8: Testing Your troubleshooting agent

### Step 1: Setting up your environment
**Instructions**  
Start by setting up your Python environment. You will need to install the following libraries:

- nltk for tokenization and POS tagging
- spaCy for NER
- transformers for sentiment analysis

Install the required libraries using the following commands:

```python
pip install nltk
pip install spacy
pip install transformers
```

**Explanation**  
These libraries provide essential tools for text processing, enabling you to build an agent that can understand, analyze, and respond to user queries.

### Step 2: Tokenizing the user query
**Instructions**  
Tokenization is the first step in processing the user's input. This involves splitting the query into individual words (tokens) so that each word can be processed separately.

**Code example**
```python
import nltk
from nltk.tokenize import word_tokenize

# Sample user query
query = "My laptop is overheating after the update."

# Tokenize the query
tokens = word_tokenize(query)
print(tokens)
```

**Explanation**  
Tokenizing the query breaks it down into smaller units, allowing the agent to analyze each word and understand the overall context of the problem.

### Step 3: Applying part-of-speech tagging
**Instructions**  
After tokenization, apply POS tagging to assign grammatical roles to each token. This will help the agent understand the structure of the sentence and how the words relate to each other.

**Code example**
```python
# Apply POS tagging to the tokens
tagged_tokens = nltk.pos_tag(tokens)
print(tagged_tokens)
```

**Explanation**  
POS tagging identifies the role of each word in the sentence, such as whether it's a noun, verb, or adjective, which helps the agent to comprehend the query more effectively.

### Step 4: Extracting key entities with named entity recognition
**Instructions**  
NER identifies important entities such as product names, dates, or locations that may be essential for diagnosing the problem. This step helps the agent focus on the most relevant parts of the query.

**Code example**
```python
import spacy

# Load the pre-trained model for NER
nlp = spacy.load("en_core_web_sm")

# Apply NER to the query
doc = nlp(query)

# Extract and print entities
for ent in doc.ents:
    print(ent.text, ent.label_)
```

**Explanation**  
NER enables the agent to identify key elements of the query, such as "laptop" (device) and "update" (event), so that it can provide more relevant troubleshooting steps.

### Step 5: Analyzing sentiment for prioritization
**Instructions**  
Sentiment analysis helps to detect the user's emotional state. This is important for prioritizing queries where the user expresses frustration or urgency.

**Code example**
```python
from transformers import pipeline

# Initialize sentiment analysis pipeline
sentiment_analyzer = pipeline('sentiment-analysis')

# Analyze the sentiment of the query
result = sentiment_analyzer(query)
print(result)
```

**Explanation**  
Sentiment analysis allows the agent to prioritize cases where the user is frustrated, ensuring that those queries are handled with greater urgency.

### Step 6: Building a knowledge base for problem-solving
**Instructions**  
Create a simple knowledge base where common problems are mapped to their solutions. The troubleshooting agent will match the user's query to the closest issue in the knowledge base and suggest a solution.

**Code example**
```python
# Sample knowledge base
knowledge_base = {
    "overheating": "Check your cooling system, clean the fans, and ensure proper ventilation.",
    "slow performance": "Close unnecessary applications, restart your system, and check for malware."
}

# Function to retrieve solutions
def get_solution(issue):
    return knowledge_base.get(issue, "No solution found for this issue.")

# Example usage
print(get_solution("overheating"))
```

**Explanation**  
The knowledge base contains predefined solutions for common issues, which the agent can use to quickly provide helpful troubleshooting steps.

### Step 7: Implementing decision-making logic
**Instructions**  
Add decision-making logic to guide the troubleshooting agent. The agent will attempt to match the user's query to an issue in the knowledge base and return the relevant solution. If no match is found, it can ask the user for more details.

**Code example**
```python
def troubleshoot(query):
    if "overheating" in query.lower():
        return get_solution("overheating")
    elif "slow" in query.lower():
        return get_solution("slow performance")
    else:
        return "Can you provide more details about the issue?"

# Example usage
response = troubleshoot(query)
print(response)
```

**Explanation**  
The decision-making logic allows the agent to ask for more information, if needed, or directly retrieve a solution from the knowledge base based on the query.

### Step 8: Testing your troubleshooting agent
**Instructions**  
Test the complete agent by running several troubleshooting queries. Use different inputs to ensure that the agent can handle a variety of issues and provide relevant solutions.

**Test cases**
- Input: "My laptop is overheating after the update."
  - Expected output: "Check your cooling system, clean the fans, and ensure proper ventilation."
- Input: "The computer is running slow."
  - Expected output: "Close unnecessary applications, restart your system, and check for malware."
- Input: "I have a problem with my printer."
  - Expected output: "Can you provide more details about the issue?"

**Explanation**  
Testing the agent with multiple queries ensures that it can handle different types of problems, ask appropriate follow-up questions, and provide the correct solutions.

## Conclusion
In conclusion, you've built a troubleshooting agent using Python, incorporating key NLP techniques such as tokenization, POS tagging, NER, and sentiment analysis. The agent uses a simple knowledge base and decision-making logic to assist users in diagnosing and resolving technical issues. In future lessons, you can expand this system by integrating machine learning models or enhancing the decision-making logic for more complex troubleshooting scenarios.

# Overview: sentiment analysis

## Introduction
Sentiment analysis is a natural language processing (NLP) technique that identifies the emotional tone of a text, categorizing it as positive, negative, or neutral. By enabling machines to understand emotional context, sentiment analysis provides valuable insights for such applications as customer feedback analysis, social media monitoring, and brand sentiment tracking.

To power such tasks, machines often employ such advanced models as long short-term memory (LSTM) networks. LSTMs are a specialized type of recurrent neural network (RNN) designed to remember important information across long sequences. They achieve this through memory cells and gated units—input, forget, and output gates—that control the flow of information, deciding what to retain or discard. This makes LSTMs particularly effective for language processing, time-series forecasting, and sequential data analysis, in which understanding context over time is critical. Together, sentiment analysis and LSTM networks form a robust combination for extracting meaningful insights from textual or sequential data.

By the end of this reading, you will be able to:

- Explain the fundamental concepts of AI and ML and their differences.
- Describe the key components of an AI/ML system, including algorithms, data, and models.
- Understand the process of training, testing, and deploying machine learning models.
- Identify common AI/ML applications across various industries.

## Principles of sentiment analysis
Explore the following principles:

- Text classification
- Polarity detection
- Subjectivity vs. objectivity

### 1. Text classification
At its core, sentiment analysis is a classification task. The goal is to categorize a given text into predefined sentiment categories—most commonly positive, negative, and neutral. Depending on the complexity, some systems may use more granular categories such as "very positive" or "very negative."

**Example**: sentiment analysis would classify the sentence "I love this product!" as positive, while it would label "I'm disappointed with the service" as negative.

### 2. Polarity detection
Polarity refers to the positivity or negativity of a text, which sentiment analysis models quantify by assigning a polarity score to each sentence or phrase. This score typically ranges from strongly negative to strongly positive, with a neutral score in the middle. For example, if sentiment analysis classifies a sentence as positive, it might receive a score like +0.8, indicating strong positivity. Similarly, it could rate a negative sentence as −-0.7, reflecting strong negativity. Neutral sentences would receive scores close to 0. These scores directly translate to discrete sentiment classes: a score above a certain threshold (e.g., +0.5) would classify the sentence as positive, a score below -−0.5 as negative, and anything near 0 as neutral. This allows for finer granularity in classifying text sentiment based on the intensity of the sentiment detected. 

Note that some sentiment analysis tools may choose a different scale through which to demonstrate either positive or negative sentiment, meaning a score of "+0.8," for example, may represent on one tool the same score as "+8,", "+80," or "+80%" on another.

**Example**: in a review such as "The food was great, but the service was terrible," sentiment analysis can assign a polarity score to each part of the sentence. For "the food was great," the model might assign a positive polarity score of +0.9, indicating strong positivity. In contrast, for "the service was terrible," the model would likely assign a negative polarity score of −0.8, reflecting strong negativity. The model computes these scores based on the presence of words like "great" (which contributes positively) and "terrible" (which contributes negatively). By breaking down the sentence into components and assigning individual polarity scores, you can understand the overall sentiment of the review as a mix of both positive and negative sentiments. The final step could aggregate these scores to provide an overall polarity, balancing the positive and negative sentiments, potentially resulting in an overall neutral score close to 0. Note that not all sentiment analyses use the same scale (and as such, a score of +100 and a score of +9 might, on two different tools, equal "maximum positive sentiment").

### 3. Subjectivity vs. objectivity
Sentiment analysis can determine whether a sentence is subjective or objective by analyzing the language used and the presence of opinionated or factual terms. Subjective sentences often contain personal opinions, emotions, or judgments, using words that reflect a personal perspective, such as "I feel," "wonderful," or "terrible." In contrast, objective sentences present factual information, using neutral language that can be verified, such as "The event took place on Tuesday" or "The car has four doors." Sentiment analysis typically achieves this distinction by examining the presence of opinion-based language, sentiment-laden words, and contextual cues that suggest personal bias versus neutral descriptions. By scoring sentences on a subjectivity scale, sentiment analysis models can classify whether a text is primarily subjective or objective, giving insight into the tone and reliability of the information provided. Note that sentiment analysis is notoriously bad at identifying sarcasm or satire and likely will not pick up on either.

**Example**: "The movie was fantastic!" is a subjective statement with strong positive sentiment, while "The movie was released in 2022" is an objective, neutral statement.

## Techniques used in sentiment analysis
Explore the following techniques:

- Rule-based approaches
- Machine learning approaches
- Deep learning and neural networks
- Pretrained models

### 1. Rule-based approaches
Early sentiment analysis systems used rule-based methods to classify text. These systems rely on manually created rules, such as dictionaries of positive and negative words, to determine a text's sentiment. Rule-based methods are simple but often lack the ability to handle nuances in language or context.

**Example**: a rule-based system may flag the word "great" as positive and "bad" as negative, but it could struggle with more complex phrases like "not bad," which carries a positive connotation despite the presence of a negative word.

### 2. Machine learning approaches
Machine learning has largely replaced rule-based systems in modern sentiment analysis. In these systems, a model is trained on a labeled dataset of text examples with corresponding sentiment labels. The model learns to recognize patterns in the data and can then classify new text based on those learned patterns.

**Example**: a sentiment analysis model trained on thousands of product reviews can predict the sentiment of a new review, even if it uses unfamiliar words or phrases.

### 3. Deep learning and neural networks
Advanced sentiment analysis systems often rely on deep learning techniques such as neural networks to train sentiment classifiers. Key models in this domain include long short-term memory (LSTM) networks—a type of recurrent neural network (RNN) designed to retain long-term dependencies in sequential data—and transformers. Both LSTMs and transformers excel at capturing the complexities of human language, including context, sarcasm, and idiomatic expressions. LSTMs, in particular, manage information flow through memory cells and gated units, utilizing three gates—input, forget, and output—to retain or discard data as needed. This architecture enables LSTMs to perform well in such tasks as language processing, time-series forecasting, and sequential data analysis, in which understanding context over time is crucial.

**Example**: a neural network-based sentiment analysis model can understand that the sentence "I couldn't be happier with this purchase!" is positive, even though the word "couldn't" usually indicates a negative sentiment.

### 4. Pretrained models
Pretrained models such as bidirectional encoder representations from transformers (BERT) and generative pretrained transformers (GPT) have revolutionized sentiment analysis by providing a strong foundation for understanding language. These models are fine-tuned for sentiment analysis tasks, allowing them to achieve high accuracy even on complex or subtle language.

**Example**: a BERT-based sentiment analysis system can detect positive or negative sentiments even in long and context-heavy texts, such as detailed customer reviews or social media posts.

## Applications of sentiment analysis
Explore the following applications:

- Customer feedback analysis
- Social media monitoring
- Brand sentiment tracking
- Political sentiment and opinion mining
- Market research

### 1. Customer feedback analysis
Companies use sentiment analysis to gauge customer satisfaction by analyzing feedback from reviews, surveys, and social media posts. This helps businesses understand what customers like or dislike about their products or services and make data-driven decisions to improve them.

**Example**: a retail company may use sentiment analysis to scan online reviews and identify common complaints about shipping delays, allowing it to address operational inefficiencies.

### 2. Social media monitoring
Sentiment analysis is widely used to monitor social media platforms like X, Facebook, and Instagram. Businesses and marketers can monitor the public's feelings about their brand, products, or campaigns in real time and make necessary adjustments.

**Example**: a brand might monitor tweets about a new product launch to gauge the public's reaction and adjust its marketing strategy based on positive or negative feedback.

### 3. Brand sentiment tracking
Sentiment analysis can help companies monitor and protect their brand's reputation by detecting trends in public perception. If sentiment shifts from positive to negative, companies can address the issues causing the shift proactively.

**Example**: a food company might track sentiment around new product lines. If customer sentiment turns negative, the company can investigate whether the product has a problem or if it needs to adjust its marketing messages.

### 4. Political sentiment and opinion mining
In politics, sentiment analysis is used to assess public opinion on policies, political candidates, and events. This information helps political analysts and campaign teams understand voter sentiment and plan campaigns accordingly.

**Example**: during an election campaign, political analysts might use sentiment analysis to study social media posts and news articles to determine how the public feels about a candidate's performance in a recent debate.

### 5. Market research
Companies often use sentiment analysis to conduct market research and gain insights into how customers feel about competitors or industry trends. This helps them stay ahead of the competition and adjust their strategies based on customer sentiment.

**Example**: a tech company might analyze sentiment around a competitor's product launch to see whether it's well received and adjust its own product road map accordingly.

## Challenges in sentiment analysis
Explore the following challenges:

- Handling sarcasm and irony
- Contextual understanding
- Multilingual sentiment analysis

### 1. Handling sarcasm and irony
Detecting sarcasm or irony is one of the most difficult challenges in sentiment analysis, as these forms of expression can completely reverse the meaning of a sentence. While humans can often easily detect sarcasm, machines still struggle with this task.

**Example**: the sentence "Oh, great! Another software update!" is sarcastic, but a machine might misinterpret it as positive because of the word "great."

### 2. Contextual understanding
Sentiment can change depending on the context in which a word or phrase is used. Sentiment analysis models need to consider the context to avoid misclassifying text.

**Example**: the word "cold" in "cold coffee" is likely negative, but in "cold weather," it may be neutral or even positive, depending on the context.

### 3. Multilingual sentiment analysis
Conducting sentiment analysis across different languages adds another layer of complexity. Each language has its own nuances, idioms, and cultural expressions, making it challenging to build sentiment analysis models that work well for multiple languages.

**Example**: a model trained in English may struggle to analyze sentiments in a language like Japanese, in which expressions of politeness can obscure the true sentiment behind a statement.

## Conclusion
Sentiment analysis is a powerful tool for understanding people's feelings about products, services, brands, and events. It provides actionable insights from customer feedback, social media monitoring, and market research to drive decision-making. As NLP technologies continue to evolve, sentiment analysis will become even more accurate, nuanced, and capable of handling the complexities of human language.

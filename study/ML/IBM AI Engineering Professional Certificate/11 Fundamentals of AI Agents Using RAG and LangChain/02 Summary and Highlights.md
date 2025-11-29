# Reading: Summary and Highlights

Congratulations! You have completed this lesson. At this point in the course, you know that:

## Key Concepts

- **RAG** is an AI framework that helps optimize the output of large language models or LLMs.

- **RAG combines** retrieved information and generates natural language to create responses.

- **RAG consists** of two main components: 
    - The **retriever** - the core of RAG
    - The **generator** - which functions as a chatbot

## RAG Process

In the RAG process:

1. The **retriever** encodes user-provided prompts and relevant documents into vectors, stores them in a vector database, and retrieves relevant context vectors based on the distance between the encoded prompt and documents.

2. The **generator** then combines the retrieved context with the original prompt to produce a response.

## Technical Components

### Dense Passage Retrieval (DPR)
- The **DPR Context Encoder** and its tokenizer focus on encoding potential answer passages or documents
- This encoder creates embeddings from extensive texts, allowing the system to compare these with question embeddings to find the best match

### Facebook AI Similarity Search (Faiss)
- **Faiss** is a library developed by Facebook AI Research that offers efficient algorithms for searching through large collections of high-dimensional vectors
- Faiss is essentially a tool to calculate the distance between the question embedding and the vector database of context vector embeddings

### DPR Question Encoder
- The **DPR question encoder** and its tokenizer focus on encoding the input questions into fixed-dimensional vector representations, grasping their meaning and context to facilitate answering them
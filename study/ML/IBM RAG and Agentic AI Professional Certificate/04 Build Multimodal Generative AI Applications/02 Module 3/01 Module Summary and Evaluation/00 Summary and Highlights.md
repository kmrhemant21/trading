# Summary and Highlights

Congratulations! You have completed this module. At this point, you know that: 

## Multimodal Retrieval-Augmented Generation (MM-RAG)

Combines multimodal inputs, such as text + images or videos, with retrieval-augmented generation, fetching relevant data to enhance LLM responses.

### Pattern follows three steps: 

1. **Multimodal data retrieval**
2. **Contrastive learning for embeddings**
3. **Generative models informed by multimodal context**

### Pipeline has four steps:

1. **Data indexing**: Diverse data, such as text, images, audio, and video, is converted into embeddings and stored in a vector database for efficient retrieval.
2. **Data retrieval**: User query is embedded, and semantically relevant multimodal data is fetched from the vector database.
3. **Augmentation**: Retrieved data is combined with the original query to enrich the context for generation.
4. **Response generation**: Multimodal response is generated using the augmented input, blending information from all modalities.

## Multimodal Chatbots and QA Systems

Advanced AI systems that process and respond to multiple data types, such as text, images, audio, and video.

- Can see, read, and understand the world more like humans do.

### Key features: 

- **Multiple input modalities**
- **Integrated understanding**
- **Contextual response generation**

### Basic implementation steps: 

1. Set up the environment and import libraries.
2. Initialize the model.
3. Prepare an image for processing.
4. Create the multimodal query function.
5. Use the multimodal QA function.

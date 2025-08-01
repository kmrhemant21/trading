# Reading: Summary and Highlights

## Image Captioning with Meta’s Llama 

Combining computer vision with natural language processing creates powerful tools for understanding visual content.

Three main stages of the image captioning process with a multimodal large language model (LLM) are:

1. **Input processing**  
    Input processing receives and prepares the image and optional text prompt.

2. **Image validation and encoding**  
    Image validation and encoding validate and convert the image into a format (e.g., Base64) suitable for the model.

3. **Multimodal LLM processing**  
    Multimodal LLM processing combines visual and textual information to generate a descriptive caption.

Core components of the image captioning system to produce captions tailored to prompts are:

- Visual encoders
- Text embedding
- Fusion layers
- Language generation tools

Implementing an image captioning system using Meta’s Llama 4 Maverick model via IBM watsonx involves:

- Importing libraries and authenticating access
- Encoding images and preparing prompts
- Sending combined image-text messages to the model
- Extracting descriptive text from the model’s response

## Text-to-Video Generation with OpenAI’s Sora

Sora is a multimodal, diffusion-based transformer model developed by OpenAI that can generate high-quality video from text or image inputs. 

For accurate results, you must craft a structured prompt and include essential elements, such as scene context, visual details, and motion required in your clip.

### Steps for Text-to-Video Generation:

1. Open your browser and go to [sora.openai.com](https://sora.openai.com) to access the official Sora interface.
2. If not logged in, click **“Log In”** or **“Sign Up”** for a new OpenAI account.
3. After logging in, you’ll land on the **“Explore page,”** where you can browse others’ videos for inspiration.
4. Use the composer at the bottom to enter your text prompt describing the video you want.
5. Before creating, review your settings:
    - **Choose Type:** Video
    - Set aspect ratio, resolution, duration, number of variations, and style preset
    - Options depend on your OpenAI subscription tier
6. Click **“Create video”** to submit your request; processing takes 30 seconds to a few minutes.
7. Finished videos appear under **“My Videos”**:
    - Hover to **“Preview”**
    - Click to open in a lightbox and use the arrow keys to view variations
8. Select a variation to refine using the editing toolbar:
    - **“Edit”** storyboard or recut clips
    - Use **“Remix”** to describe changes in natural language
    - Use **“Blend”** to merge with another video
    - Use **“Loop”** to create seamless repeats
9. After editing, a new variation is added to your set.

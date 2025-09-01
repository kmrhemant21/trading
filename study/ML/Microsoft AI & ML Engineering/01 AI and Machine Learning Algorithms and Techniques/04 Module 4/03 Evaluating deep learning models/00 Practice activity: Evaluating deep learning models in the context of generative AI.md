# Practice activity: Evaluating deep learning models in the context of generative AI

## Introduction
In this practice activity, you will evaluate the performance of deep learning models within the context of GenAI, specifically focusing on generative adversarial networks (GANs) and autoencoders. 

By the end of this practice activity, you will:
- Compare the performance of these models in tasks related to data generation and reconstruction.
- Use appropriate evaluation metrics for generative models.
- Analyze the results to understand which model is best suited for various GenAI tasks.

## Step-by-step instructions

### Step 1: Set up the environment
Ensure that TensorFlow is installed. If it isn't, you can install it with the following command:

```python
pip install tensorflow
```

Import the necessary libraries to build, train, and evaluate generative models:

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
```

### Step 2: Evaluate an autoencoder

#### Objective
You will evaluate an autoencoder by measuring how accurately it can reconstruct images from the MNIST dataset. The evaluation metric will be MSE.

#### Steps
1. Load the MNIST dataset
```python
# Load MNIST dataset
(X_train, _), (X_test, _) = tf.keras.datasets.mnist.load_data()

# Normalize and flatten images
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
```

2. Define the autoencoder architecture
```python
# Define the encoder
encoder = models.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu')  # Bottleneck layer
])

# Define the decoder
decoder = models.Sequential([
    layers.Input(shape=(64,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(784, activation='sigmoid')  # Reconstructed output
])

# Build the autoencoder model
autoencoder = models.Sequential([encoder, decoder])
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=20, batch_size=256, validation_data=(X_test, X_test))
```

3. Evaluate the autoencoder's reconstruction performance
```python
# Predict reconstructed images
reconstructed_images = autoencoder.predict(X_test)

# Calculate the mean squared error
mse = np.mean(np.square(X_test - reconstructed_images))
print(f'Autoencoder Reconstruction MSE: {mse}')
```

#### Evaluation metric
MSE measures how well the autoencoder can reconstruct the original images. Lower MSE values indicate better reconstruction quality.

### Step 3: Evaluating a GAN

#### Objective
You will evaluate the performance of a GAN by examining its ability to generate realistic images from random noise. The evaluation metric will be a visual inspection of the generated images, combined with the discriminator accuracy in distinguishing real from fake images.

#### Steps
1. Define the GAN architecture
```python
# Define the generator
def build_generator():
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(100,)),
        layers.Dense(784, activation='sigmoid')  # Output: 28x28 flattened image
    ])
    return model

# Define the discriminator
def build_discriminator():
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(1, activation='sigmoid')  # Output: Probability (real or fake)
    ])
    return model
```

2. Compile the discriminator and GAN
```python
generator = build_generator()
discriminator = build_discriminator()

discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Stack the generator and discriminator
gan = models.Sequential([generator, discriminator])
discriminator.trainable = False
gan.compile(optimizer='adam', loss='binary_crossentropy')
```

3. Train the GAN

> Note: The training process for the GAN in Step 3 may take a significant amount of time to complete, depending on hardware capabilities and dataset size. If the training duration is too long, consider adjusting the number of epochs to a lower value to balance performance and efficiency. Reducing epochs can speed up execution while still providing meaningful results. Feel free to experiment with different values to find an optimal balance.

```python
# Training GAN
epochs = 10000
batch_size = 64
half_batch = batch_size // 2

for epoch in range(epochs):
    # Real images
    idx = np.random.randint(0, X_train.shape[0], half_batch)
    real_images = X_train[idx]
    real_labels = np.ones((half_batch, 1))

    # Fake images
    noise = np.random.normal(0, 1, (half_batch, 100))
    fake_images = generator.predict(noise)
    fake_labels = np.zeros((half_batch, 1))

    # Train the discriminator
    d_loss_real = discriminator.train_on_batch(real_images, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)

    # Train the generator (via GAN model)
    noise = np.random.normal(0, 1, (batch_size, 100))
    gan_labels = np.ones((batch_size, 1))  # Try to fool the discriminator
    g_loss = gan.train_on_batch(noise, gan_labels)

    # Every 1000 epochs, print losses and visualize generated images
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Discriminator Loss: {d_loss_real[0]}, Generator Loss: {g_loss}")
        # Generate and display images
        generated_images = generator.predict(np.random.normal(0, 1, (10, 100)))
        for i in range(10):
            plt.subplot(2, 5, i+1)
            plt.imshow(generated_images[i].reshape(28, 28), cmap='gray')
        plt.show()
```

4. Evaluate the GAN's performance
   - **Discriminator accuracy**: track the accuracy of the discriminator during training. High accuracy means the discriminator is good at distinguishing real from fake images.
   - **Visual inspection**: observe the quality of the generated images every 1000 epochs. Over time, the generated images should become more realistic.

### Step 4: Compare the models
- **Autoencoder**: use the MSE score to evaluate the model's ability to reconstruct the MNIST images.
- **GAN**: use a combination of discriminator accuracy and visual inspection to evaluate how realistic the generated images are.

## Deliverables
By the end of this actiivity, you should develop:

1. Your code for building, training, and evaluating the autoencoder and GAN.
2. A brief report (300–400 words) discussing:
   - The performance of each model.
   - How the autoencoder's MSE reflects its reconstruction ability.
   - The quality of the GAN's generated images and how well the discriminator learned to distinguish real from fake data.

## Conclusion
This activity allows you to explore the effectiveness of deep learning models in GenAI tasks. By evaluating an autoencoder for image reconstruction and a GAN for generating realistic images, you gain insight into how these models perform in different GenAI applications.

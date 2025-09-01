# Walkthrough: Evaluating deep learning models in the context of generative AI (Optional)

## Introduction

In this walkthrough, you are guided through the evaluation of deep learning models in the context of generative AI, focusing on autoencoders and generative adversarial networks (GANs). This walkthrough provides a detailed explanation of the proper solution for each model, including code implementation, evaluation metrics, and expected results.

By the end of this walkthrough, you'll be able to:

- Evaluate the performance of autoencoders and GANs in generating and reconstructing images. 
- Gain hands-on experience with model implementation, training processes, and evaluation metrics, enabling you to assess the effectiveness of these models in generative AI applications.

## Evaluating an autoencoder

### Objective

You evaluated an autoencoder's performance by measuring its ability to reconstruct images from the Modified National Institute of Standards and Technology (MNIST) dataset using mean squared error (MSE) as the evaluation metric.  

### Solution walkthrough

#### Step 1: Load and preprocess the data

You loaded the MNIST dataset and normalized the pixel values to be between 0 and 1. The data was also flattened to prepare it for the fully connected layers of the autoencoder.

```python
# Load MNIST dataset
(X_train, _), (X_test, _) = tf.keras.datasets.mnist.load_data()

# Normalize and flatten the images
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
```

#### Step 2: Define the autoencoder architecture

The autoencoder consists of an encoder that compresses the input into a lower-dimensional representation (latent space) and a decoder that reconstructs the input from this compressed form.

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
    layers.Dense(784, activation='sigmoid')  # Reconstruct the original input
])

# Build the autoencoder
autoencoder = models.Sequential([encoder, decoder])
autoencoder.compile(optimizer='adam', loss='mse')
```

#### Step 3: Train the autoencoder

You trained the autoencoder to minimize the MSE between the original and reconstructed images.

```python
# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=20, batch_size=256, validation_data=(X_test, X_test))
```

#### Step 4: Evaluate the autoencoder

After training, you evaluated the model's performance by calculating the MSE on the test data. A lower MSE indicates better reconstruction accuracy.

```python
# Predict reconstructed images
reconstructed_images = autoencoder.predict(X_test)

# Calculate the Mean Squared Error
mse = np.mean(np.square(X_test - reconstructed_images))
print(f'Autoencoder Reconstruction MSE: {mse}')
```

### Expected outcome

The autoencoder should achieve a low MSE (typically around 0.01 to 0.03), indicating that it successfully reconstructs the input images from their compressed latent representations.

The MSE score quantifies how close the reconstructed images are to the original ones. Lower MSE values mean the model is performing well.

## Evaluating a generative adversarial network (GAN)

### Objective

You evaluated the GAN's performance by observing its ability to generate realistic images from random noise and tracking the discriminator accuracy over time.

### Solution walkthrough

#### Step 1: Define the GAN architecture

The GAN consists of two models:

- Generator: produces fake images from random noise
- Discriminator: tries to distinguish between real and fake images

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

#### Step 2: Compile and train the GAN

The discriminator was compiled using binary cross-entropy loss. The GAN model combines both the generator and discriminator to train the generator to produce better fake images.

```python
# Build the models
generator = build_generator()
discriminator = build_discriminator()

# Compile the discriminator
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Stack the generator and discriminator
gan = models.Sequential([generator, discriminator])
discriminator.trainable = False  # Freeze the discriminator during GAN training

gan.compile(optimizer='adam', loss='binary_crossentropy')
```

#### Step 3: Train the GAN

The discriminator was trained alternately on real and fake images. The generator was trained to fool the discriminator.

```python
# Training the GAN
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

    # Train the generator
    noise = np.random.normal(0, 1, (batch_size, 100))
    gan_labels = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, gan_labels)

    # Print losses every 1000 epochs and visualize generated images
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Discriminator Loss: {d_loss_real[0]}, Generator Loss: {g_loss}")
        # Generate and display images
        generated_images = generator.predict(np.random.normal(0, 1, (10, 100)))
        for i in range(10):
            plt.subplot(2, 5, i+1)
            plt.imshow(generated_images[i].reshape(28, 28), cmap='gray')
        plt.show()
```

#### Step 4: Evaluate the GAN

The discriminator accuracy was used to measure how well the discriminator distinguishes between real and fake images. The quality of the generated images was visually inspected after every 1,000 epochs to track the GAN's progress.

### Expected outcome

**Discriminator accuracy**: In the early stages, the discriminator accuracy should be high, as the discriminator easily distinguishes real from fake images. As the generator improves, this accuracy level may decrease as the generated images become more realistic.

**Visual quality**: The generated images should start off as noise but gradually resemble MNIST digits as training progresses. By the 10,000th epoch, the images should be visually recognizable as handwritten digits.

## Summary of results

After completing the activity, you should have:

- **Autoencoder**: Achieved a low MSE score for the reconstruction of MNIST images, demonstrating its ability to compress and reconstruct data.
- **GAN**: Generated realistic images over time while observing the evolution of the discriminator accuracy and the improvement in image quality.

Both models play a crucial role in generative AI, with autoencoders excelling in reconstruction tasks and GANs shining in generating new, realistic data.

## Conclusion

This walkthrough provided you with practical insights into the evaluation of autoencoders and GANs within the realm of generative AI. You learned how to implement, train, and assess the performance of these models using quantitative metrics and qualitative observations. By understanding the strengths and weaknesses of each approach, you are better equipped to apply these techniques in real-world applications, paving the way for advancements in AI and ML.

# Practice activity: Analyzing a dataset and implementing a neural network for deep learning analysis

## Introduction
This activity will guide you through the process of using deep learning to solve a real-world problem using a dataset. By the end of the activity, you should be able to apply neural networks to various types of data for analysis and prediction.

By the end of this activity, you'll be able to:

- Analyze a given dataset to understand its structure and features.
- Implement a deep learning neural network for analysis and prediction.
- Evaluate the results of the neural network model to understand its performance.

## Step-by-step instructions

### Step 1: Analyzing the dataset

#### Objective
You'll begin by analyzing the dataset to understand its structure, the features it contains, and the target variable you want to predict.

#### Dataset
For this activity, we'll use the Pima Indians diabetes dataset (or another dataset provided by your instructor). This dataset contains several medical predictor variables and a target variable indicating whether a patient has diabetes.

#### Steps
1. **Load the dataset**  
    You can load the dataset using pandas. The dataset consists of medical attributes such as glucose level, insulin, body mass index (BMI), age, and the target variable (1 for diabetes, 0 for no diabetes).

    ```python
    import pandas as pd

    # Load the dataset
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    data = pd.read_csv(url, names=columns)

    # Display the first few rows of the dataset
    print(data.head())
    ```

2. **Perform basic data analysis**  
    Explore the dataset by checking for missing values, noting statistical summaries, and understanding the distribution of the features.

    ```python
    # Check for missing values
    print(data.isnull().sum())

    # Display basic statistics
    print(data.describe())

    # Check the distribution of the target variable
    print(data['Outcome'].value_counts())
    ```

### Step 2: Preprocessing the data

#### Objective
Before implementing the neural network, the data must be preprocessed. This includes normalizing the features and splitting the dataset into training and test sets.

#### Steps
1. **Feature scaling**  
    Neural networks perform better when the features are scaled to a similar range. We'll scale the feature values to a range between 0 and 1 using MinMaxScaler.

    ```python
    from sklearn.preprocessing import MinMaxScaler

    # Separate features and target variable
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    # Scale the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    ```

2. **Splitting the dataset**  
    Split the dataset into training and testing sets. Typically, 80 percent of the data is used for training and 20 percent for testing.

    ```python
    from sklearn.model_selection import train_test_split

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    ```

### Step 3: Implementing a deep learning neural network

#### Objective
You will implement a deep learning neural network using TensorFlow's Keras API to predict whether a patient has diabetes.

#### Steps
1. **Build the neural network**  
    We'll create a simple feedforward neural network with two hidden layers.

    ```python
    import tensorflow as tf
    from tensorflow.keras import layers, models

    # Build the neural network model
    model = models.Sequential([
         layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # First hidden layer
         layers.Dense(32, activation='relu'),  # Second hidden layer
         layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    ```

2. **Compile the model**  
    Compile the model using binary cross-entropy loss (suitable for binary classification) and the adaptive moment estimation (Adam) optimizer.

    ```python
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    ```

3. **Train the model**  
    Train the model using the training dataset, and validate its performance using the test dataset. We'll train for 50 epochs.

    ```python
    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    ```

### Step 4: Evaluating the results

#### Objective
Evaluate the model's performance using accuracy and visualize the training process with loss and accuracy plots.

#### Steps
1. **Evaluate the model on test data**  
    After training, evaluate the model's accuracy on the test dataset.

    ```python
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {test_accuracy}')
    ```

2. **Plot accuracy and loss**  
    Use Matplotlib to plot the model's training and validation accuracy and loss over time to observe its performance.

    ```python
    import matplotlib.pyplot as plt

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    ```

## Deliverables
By the end of this activity, you should create:

- Your code for data analysis, preprocessing, building, and training the neural network.
- A report (300–400 words) summarizing the results, including:
  - The model's test accuracy.
  - Insights gained from the accuracy and loss plots.
  - Any challenges or observations during the training process.

## Conclusion
In this activity, you learned how to analyze a dataset, preprocess the data, implement a deep learning neural network, and evaluate the model's performance. This process is crucial for applying deep learning to solve real-world problems across various industries, from healthcare to finance.
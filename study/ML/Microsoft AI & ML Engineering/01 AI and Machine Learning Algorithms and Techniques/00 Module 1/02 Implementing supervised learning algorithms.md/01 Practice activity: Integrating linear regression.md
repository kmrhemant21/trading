# Practice Activity: Integrating Linear Regression

## Introduction

In this activity, you'll implement and integrate a linear regression model using Python and the scikit-learn library. Linear regression is a fundamental algorithm for predicting a continuous target variable based on input features.

By the end of this activity, you will be able to:

- Set up a linear regression model
- Train the model with data
- Evaluate the model's performance

---

## 1. Setting Up Your Environment

Before you begin, ensure you have the necessary libraries installed. If not, install them using:

```bash
pip install numpy pandas scikit-learn matplotlib
```

---

## 2. Importing Required Libraries

Start by importing the libraries needed for this lab:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
```

- **NumPy** and **pandas** help handle numerical and tabular data.
- **scikit-learn's** `LinearRegression` is used to build the model.
- **Matplotlib** is used to visualize results.

---

## 3. Loading and Preparing the Data

For this example, you'll predict house prices based on square footage. You can use your own dataset or create a synthetic one:

```python
# Sample dataset (house prices based on square footage)
data = {
    'SquareFootage': [1500, 1800, 2400, 3000, 3500, 4000, 4500],
    'Price': [200000, 250000, 300000, 350000, 400000, 500000, 600000]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the first few rows
print(df.head())
```

---

## 4. Splitting the Data into Training and Testing Sets

Split the dataset into training and testing sets to evaluate model performance:

```python
# Features (X) and Target (y)
X = df[['SquareFootage']]
y = df['Price']

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display shapes
print(f"Training data: {X_train.shape}, {y_train.shape}")
print(f"Testing data: {X_test.shape}, {y_test.shape}")
```

---

## 5. Training the Linear Regression Model

Create and train the linear regression model:

```python
# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Display learned coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")
```

- The intercept and coefficient define the linear equation `y = mx + b`.

---

## 6. Making Predictions

Use the trained model to make predictions on the test data:

```python
# Predict on the testing set
y_pred = model.predict(X_test)

# Display predictions
print("Predicted Prices:", y_pred)
print("Actual Prices:", y_test.values)
```

---

## 7. Evaluating the Model

Evaluate the model using Mean Squared Error (MSE) and R-squared (R²):

```python
# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Calculate R-squared
r2 = r2_score(y_test, y_pred)

# Display metrics
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
```

- **MSE**: Average squared difference between actual and predicted values (lower is better).
- **R²**: Indicates how well the model fits the data (1 = perfect fit, 0 = no fit).

---

## 8. Visualizing the Results

Visualize the regression line against the data points:

```python
# Plot data points
plt.scatter(X_test, y_test, color='blue', label='Actual Data')

# Plot regression line
plt.plot(X_test, y_pred, color='red', label='Regression Line')

# Labels and title
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.title('House Prices vs. Square Footage')
plt.legend()

# Show plot
plt.show()
```

The plot displays actual data points and the fitted regression line, showing how well the model fits.

---

## Conclusion

In this lab, you learned how to:

- Set up a linear regression model using scikit-learn
- Train the model on a dataset
- Evaluate its performance using MSE and R²
- Visualize the results

Linear regression is a simple yet powerful tool for predicting continuous values. You can now apply linear regression to your own datasets and projects. Feel free to experiment with different datasets or tweak model parameters to further explore linear regression!

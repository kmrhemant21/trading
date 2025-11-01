# Practice activity: Implementing preprocessing techniques

## Introduction
In this hands-on activity, you will preprocess a dataset using several common data preprocessing techniques. This exercise will help to reinforce your understanding of data cleaning, transformation, and preparation, providing practical experience with tools and techniques used in real-world machine learning projects.

By the end of this activity, you will be able to:

- Apply preprocessing techniques to effectively prepare data for machine learning models.

## Step-by-step process for data preprocessing
This reading will guide you through the following steps:

1. Step 1: Load and inspect the dataset.
2. Step 2: Handle missing values.
3. Step 3: Standardize numerical features.
4. Step 4: Encode categorical variables.
5. Step 5: Detect and handle outliers.
6. Step 6: Address skewed data.
7. Step 7: Split the dataset.

Open a new Jupyter Notebook in Azure Machine Learning to run this code. As always, make sure you've selected the Python 3.8 - AzureML kernel.  

### Step 1: Load and inspect the dataset
Define the dataset and import necessary libraries:

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats

# Generate synthetic dataset for testing
np.random.seed(42)
n_samples = 100
data = {
    'income': np.random.normal(50000, 15000, n_samples),
    'credit_score': np.random.normal(650, 50, n_samples),
    'job_title': np.random.choice(['Engineer', 'Teacher', 'Doctor', 'Artist'], n_samples),
    'target': np.random.choice([0, 1], n_samples)
}

# Introduce missing values and outliers for testing
data['income'][np.random.randint(0, n_samples, 5)] = np.nan
data['credit_score'][np.random.randint(0, n_samples, 3)] = np.nan
data['income'][np.random.randint(0, n_samples, 2)] = 150000  # Outliers
df = pd.DataFrame(data)
```

Inspect the dataset to identify missing values, outliers, and inconsistencies. 

### Step 2: Handle missing values
Use imputation techniques to handle missing data:

```python
# Handle missing values by filling with median
for column in df.select_dtypes(include=['float64', 'int64']).columns:
    df[column].fillna(df[column].median(), inplace=True)
```

Remove duplicate entries to ensure data consistency:

```python
# Remove duplicates
df.drop_duplicates(inplace=True)
```

### Step 3: Standardize numerical features 
Scale numerical features to ensure uniformity across features:

```python
scaler = StandardScaler()
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_features] = scaler.fit_transform(df[numeric_features])
```

Standardizing helps to prevent any single feature from dominating the model.

### Step 4: Encode categorical variables 
Convert categorical features into numerical format using one-hot encoding:

```python
# One-hot encode categorical features
df = pd.get_dummies(df, drop_first=True)
```

This transformation ensures compatibility with machine learning algorithms.

### Step 5: Detect and handle outliers 
Identify and remove outliers using Z-score analysis:

```python
# Detect and remove outliers using Z-score
z_scores = np.abs(stats.zscore(df.select_dtypes(include=['float64', 'int64'])))
df = df[(z_scores < 3).all(axis=1)]
```

Removing outliers maintains the integrity of the dataset.

### Step 6: Address skewed data
Apply logarithmic transformation to reduce skewness in skewed features:

```python
# Apply log transformation to skewed data
df['income_log'] = np.log1p(df['income'])
```

This technique makes the data more suitable for modeling. You can ignore the error message here about invalid values. 

### Step 7: Split the dataset
Separate the data into training and testing sets:

```python
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Splitting ensures an unbiased evaluation of the model's performance.

## Real-world scenario
By applying the preprocessing techniques outlined in this guide, you can:

- Impute missing incomes and remove duplicate records to create a clean dataset.
- Standardize numerical features such as income and credit scores to bring them to a uniform scale.
- Encode job titles into numerical values, allowing algorithms to interpret them effectively.
- Detect and remove outliers in income and credit histories, improving model reliability.
- Transform skewed data using logarithmic techniques for better feature distribution.
- Split the data into training and testing sets, ensuring fair model validation.

These steps result in a robust, high-quality dataset that enhances the accuracy and reliability of predictive modeling.

## Conclusion
In this activity, you learned how to:

- Apply data cleaning techniques, such as handling missing values and correcting errors.
- Use feature scaling techniques to ensure data consistency.
- Encode categorical variables and perform feature selection.
- Detect and handle outliers appropriately.
- Transform data for improved model suitability.
- Prepare the dataset for training and testing.

By following these preprocessing steps, you can transform messy, real-world data into a reliable foundation for machine learning models, ensuring better predictions and improved project outcomes.

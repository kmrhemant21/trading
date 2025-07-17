# Practice Activity: Apply the Preprocessing Tool to a Dummy Dataset for ML Application

## Introduction

Now that you've set up your data preprocessing tool, it's time to put it to use. In this reading, we'll walk you through how to apply your tool to preprocess a set of dummy data, preparing it for ML applications. This practical example will demonstrate how to clean, handle missing values, manage outliers, scale, and encode your data, ensuring it’s ready for training an ML model.

By the end of this activity, you will be able to: 

- Load and create a dummy dataset for preprocessing.
- Apply data cleaning techniques to handle missing values and outliers.
- Scale numeric features and encode categorical variables for machine learning.
- Save the cleaned and preprocessed data for future use.

---

## 1. Loading the Dummy Data

First, let's create a set of dummy data that simulates a typical dataset you might encounter in an ML project. For this example, you will generate a DataFrame with numeric and categorical data, some missing values, and a few outliers.

### Step-by-Step Guide:

**Step 1: Generate and Load the Dummy Data**

```python
import pandas as pd
import numpy as np

# Create a dummy dataset
np.random.seed(0)
dummy_data = {
    'Feature1': np.random.normal(100, 10, 100).tolist() + [np.nan, 200],  # Normally distributed with an outlier
    'Feature2': np.random.randint(0, 100, 102).tolist(),  # Random integers
    'Category': ['A', 'B', 'C', 'D'] * 25 + [np.nan, 'A'],  # Categorical with some missing values
    'Target': np.random.choice([0, 1], 102).tolist()  # Binary target variable
}

# Convert the dictionary to a pandas DataFrame
df_dummy = pd.DataFrame(dummy_data)

# Display the first few rows of the dummy dataset
print(df_dummy.head())
```

**Explanation:**  
This code generates a dummy dataset with 100 rows and 4 columns: two numeric features, one categorical feature, and a binary target variable. The dataset includes some missing values and an outlier to simulate real-world data challenges.

---

## 2. Applying the Preprocessing Tool 

Next, use the preprocessing tool you set up in the previous lesson to clean and preprocess this dummy data, making it ready for ML.

### Step 2: Load the Preprocessing Tool 

Ensure your preprocessing functions are loaded into your environment. These functions include handling missing values, removing outliers, scaling data, and encoding categorical variables.

```python
def load_data(df):
    return df

def handle_missing_values(df):
    return df.fillna(df.mean())  # For numeric data, fill missing values with the mean

def remove_outliers(df):
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
    return df[(z_scores < 3).all(axis=1)]  # Remove rows with any outliers

def scale_data(df):
    scaler = StandardScaler()
    df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    return df

def encode_categorical(df, categorical_columns):
    return pd.get_dummies(df, columns=categorical_columns)

def save_data(df, output_filepath):
    df.to_csv(output_filepath, index=False)
```

**Explanation:**  
These functions encapsulate the core preprocessing tasks, making them reusable across different datasets. They will be applied to our dummy data.

---

### Step 3: Preprocess the Dummy Data

Apply the preprocessing tool to the dummy data:

```python
# Load the data
df_preprocessed = load_data(df_dummy)

# Handle missing values
df_preprocessed = handle_missing_values(df_preprocessed)

# Remove outliers
df_preprocessed = remove_outliers(df_preprocessed)

# Scale the data
df_preprocessed = scale_data(df_preprocessed)

# Encode categorical variables
df_preprocessed = encode_categorical(df_preprocessed, ['Category'])

# Display the preprocessed data
print(df_preprocessed.head())
```

**Explanation:**  
This code applies the preprocessing steps to the dummy data. It handles missing values by filling them with the mean, removes outliers using the Z-score method, scales the numeric data, and encodes the categorical variables using one-hot encoding.

---

## 3. Saving the Preprocessed Data

Finally, save the preprocessed data to a new comma-separated values (CSV) file for use in ML tasks. 

### Step 4: Save the Preprocessed Data

```python
# Save the cleaned and preprocessed DataFrame to a CSV file
save_data(df_preprocessed, 'preprocessed_dummy_data.csv')

print('Preprocessing complete. Preprocessed data saved as preprocessed_dummy_data.csv')
```

**Explanation:**  
Saving the preprocessed data to a new file ensures that it’s ready for use in training ML models. This step makes it easy to use the cleaned and processed data in future analysis or modeling efforts.

---

## 4. Verifying the Preprocessing Steps 

After preprocessing, it’s important to verify that the data has been processed correctly:

**Check for Missing Values:**  

```python
print(df_preprocessed.isnull().sum())
```

**Explanation:**  
This checks that all missing values have been handled properly.

**Verify Outlier Removal:**  

```python
print(df_preprocessed.describe())
```

**Explanation:**  
This summarizes the dataset and confirms that any extreme values (outliers) have been removed.

**Inspect Scaled Data:**  

```python
print(df_preprocessed.head())
```

**Explanation:**  
This ensures that the numeric features have been scaled properly, making them ready for ML algorithms.

**Check Categorical Encoding:**  

```python
print(df_preprocessed.columns)
```

**Explanation:**  
This confirms that the categorical variables have been encoded into numerical values correctly.

---

## Conclusion

By completing this activity, you have successfully used your data preprocessing tool to clean and prepare a set of dummy data for ML. This process included handling missing values, managing outliers, scaling numeric features, and encoding categorical variables. The preprocessed data is now ready for use in training ML models.

As you continue to work with real-world datasets, this preprocessing tool will be invaluable in ensuring that your data is clean, consistent, and properly formatted, ultimately leading to better-performing models. Continue to refine and adapt this tool to suit the specific needs of your projects, and you'll be ready to handle the challenges of data preprocessing in any ML workflow.
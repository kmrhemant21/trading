# Practice activity: Setup a local data cleaning and preprocessing tool

## Introduction
Data cleaning and preprocessing are essential steps in preparing your data for analysis and machine learning. Setting up a local environment for these tasks allows you to automate and streamline your workflow. In this reading, we’ll guide you through the process of setting up a local data cleaning and preprocessing tool using Python. This setup will involve installing necessary libraries, creating a reusable script, and automating common data preprocessing tasks.

By the end of this hands-on activity, you will be able to: 

- Set up a local environment for data cleaning and preprocessing.
- Load and clean datasets by handling missing values and outliers.
- Normalize and encode data for machine learning applications.
- Automate data preprocessing tasks using reusable Python functions.

---

## Part 1. Prerequisites
Before you begin, make sure you have the following installed on your local machine:

- **Python 3.x**: the programming language we’ll use for scripting
- **pip**: Python’s package installer, which you’ll use to install the necessary libraries
- **A code editor**: such as VS Code, PyCharm, or a text editor such as Sublime Text

### Key Python libraries for data cleaning and preprocessing:
- `pandas`: for data manipulation and analysis
- `NumPy`: for numerical operations
- `Scikit-learn`: for data preprocessing and machine learning tasks
- `Missingno` (optional): for visualizing missing data

Install these libraries using pip:

```bash
pip install pandas numpy scikit-learn missingno
```

---

## Part 2. Create the data cleaning and preprocessing script
Once you have the necessary libraries installed, you can start creating your data cleaning and preprocessing tool.

### Step-by-step guide

#### Step 1: Import the required libraries
Begin by importing the libraries you’ll use in your script:

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import missingno as msno  # Optional: for visualizing missing data
```

**Explanation**: These libraries provide the functions and methods you’ll need to clean and preprocess your data, such as handling missing values, scaling data, and visualizing missing data.

---

#### Step 2: Load the dataset
Load the dataset you want to clean and preprocess:

```python
# Load your dataset into a pandas DataFrame
df = pd.read_csv('your_dataset.csv')  # Replace 'your_dataset.csv' with your actual file path

# Display the first few rows of the dataset
print(df.head())
```

**Explanation**: The `pd.read_csv()` function loads the dataset into a pandas DataFrame, which is a powerful data structure for manipulating and analyzing data in Python.

---

#### Step 3: Handle missing values
One of the first steps in data cleaning is handling missing values. You can choose to remove, fill, or visualize missing data:

```python
# Visualize missing data (optional)
msno.matrix(df)
msno.heatmap(df)

# Drop rows with missing values
df_cleaned = df.dropna()

# Or, fill missing values with the mean
df_filled = df.fillna(df.mean())
```

**Explanation**: The `msno` library provides visualization tools to understand where missing data is in your dataset. The `dropna()` and `fillna()` methods allow you to handle missing values by either removing them or filling them with a substitute value.

---

#### Step 4: Handle outliers
Outliers can be managed by either removing them or transforming them:

```python
# Identify outliers using Z-score
from scipy import stats

z_scores = np.abs(stats.zscore(df_cleaned))
df_no_outliers = df_cleaned[(z_scores < 3).all(axis=1)]

# Or cap outliers at a threshold
upper_limit = df_cleaned['column_name'].quantile(0.95)
df_cleaned['column_name'] = np.where(df_cleaned['column_name'] > upper_limit, upper_limit, df_cleaned['column_name'])
```

**Explanation**: The Z-score method helps identify outliers by calculating how many standard deviations a data point is from the mean. You can remove these outliers or cap them to reduce their impact on your analysis.

---

#### Step 5: Scale and normalize data
Normalize or scale your data to ensure that all features contribute equally to the model:

```python
# Min-Max Scaling
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_cleaned), columns=df_cleaned.columns)

# Z-score Standardization
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df_cleaned), columns=df_cleaned.columns)
```

**Explanation**: Scaling ensures that all numerical features in your dataset are on the same scale, which is important for many machine learning algorithms. Min-Max Scaling scales data to a [0, 1] range, while Z-score Standardization scales data to have a mean of 0 and a standard deviation of 1.

---

#### Step 6: Encode categorical variables
Convert categorical variables into a numerical format that machine learning algorithms can process:

```python
# One-hot encoding for categorical variables
df_encoded = pd.get_dummies(df_scaled, columns=['categorical_column_name'])
```

**Explanation**: One-hot encoding converts categorical variables into a format that can be provided to ML algorithms to do a better job in prediction.

---

#### Step 7: Save the cleaned and preprocessed data
Once you’ve cleaned and preprocessed your data, save it to a new file:

```python
# Save the cleaned and preprocessed DataFrame to a new CSV file  
df_encoded.to_csv('cleaned_preprocessed_data.csv', index=False)

print('Data cleaning and preprocessing complete. File saved as cleaned_preprocessed_data.csv')
```

**Explanation**: The cleaned and preprocessed data is saved to a new CSV file, making it ready for use in analysis or model training.

---

## Part 3. Automate the workflow
To streamline your data preprocessing workflow, consider wrapping these steps into functions or a reusable script. Here’s a basic structure:

```python
def load_data(filepath):
    return pd.read_csv(filepath)

def handle_missing_values(df):
    return df.fillna(df.mean())

def remove_outliers(df):
    z_scores = np.abs(stats.zscore(df))
    return df[(z_scores < 3).all(axis=1)]

def scale_data(df):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

def encode_categorical(df, categorical_columns):
    return pd.get_dummies(df, columns=categorical_columns)

def save_data(df, output_filepath):
    df.to_csv(output_filepath, index=False)

# Example usage:
df = load_data('your_dataset.csv')
df = handle_missing_values(df)
df = remove_outliers(df)
df = scale_data(df)
df = encode_categorical(df, ['categorical_column_name'])
save_data(df, 'cleaned_preprocessed_data.csv')
```

**Explanation**: These functions encapsulate each step of the data cleaning and preprocessing workflow, making it easier to apply the same process to different datasets.

---

## Conclusion
By setting up a local data cleaning and preprocessing tool, you can automate much of the work involved in preparing your data for analysis. This setup ensures that your data is clean, consistent, and properly formatted, which is essential for building accurate and reliable machine learning models. 

As you refine your script and add more functionality, you’ll become more efficient in handling a wide variety of data challenges, paving the way for successful AI/ML projects.

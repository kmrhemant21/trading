# Practice activity: Implementing dimensionality reduction techniques

## Introduction

In this activity, you will implement dimensionality reduction techniques using Python. You will apply both principal component analysis (PCA) and t-distributed stochastic neighbor embedding (t-SNE) to a dataset. The goal is to reduce the number of features and visualize the dataset in a lower-dimensional space while retaining as much useful information as possible.

By the end of this activity, you'll be able to:

- **Implement dimensionality reduction techniques**: Apply PCA and t-SNE to reduce the dimensionality of a dataset using Python.
- **Visualize reduced data**: Create scatter plots to visualize the results of PCA and t-SNE, helping you understand how these techniques retain important information and reveal patterns in a lower-dimensional space.
- **Interpret the outcomes of PCA and t-SNE**: Explain how PCA captures variance through linear transformation and how t-SNE preserves local structures to highlight clusters, gaining insights into the dataset's underlying patterns.

## Step-by-step guide:

### Step 1: Setting up the environment

Before starting the activity, ensure that you have the necessary libraries installed. You will need pandas, Scikit-learn, and Matplotlib.

Install the required libraries using the following command:

```bash
pip install pandas scikit-learn matplotlib
```

### Step 2: Importing required libraries

Import the necessary libraries for data manipulation, dimensionality reduction, and visualization:

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
```

- pandas for data handling
- Scikit-learn for applying PCA and t-SNE algorithms
- Matplotlib for visualizing the results

### Step 3: Loading the dataset

You will use a fictional dataset with customer data. The dataset includes the features Annual Income, Spending Score, and Age. Load the dataset using the following code:

```python
# Create a sample dataset with customer annual income, spending score, and age
data = {
    'AnnualIncome': [
        15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5, 
        20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5, 24, 24.5, 
        25, 25.5, 26, 26.5, 27, 27.5, 28, 28.5, 29, 29.5, 
        30, 30.5, 31, 31.5, 32, 32.5, 33, 33.5, 34, 34.5, 
        35,   # Normal points
        80, 85, 90  # Outliers
    ],
    'SpendingScore': [
        39, 42, 45, 48, 51, 54, 57, 60, 63, 66,
        69, 72, 75, 78, 81, 84, 87, 90, 93, 96,
        6, 9, 12, 15, 18, 21, 24, 27, 30, 33,
        5, 8, 11, 14, 17, 20, 23, 26, 29, 32,
        56,   # Normal points
        2, 3, 100  # Outliers
    ],
    'Age': [
        20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5, 24, 24.5, 
        25, 25.5, 26, 26.5, 27, 27.5, 28, 28.5, 29, 29.5, 
        30, 30.5, 31, 31.5, 32, 32.5, 33, 33.5, 34, 34.5, 
        35, 35.5, 36, 36.5, 37, 37.5, 38, 38.5, 39, 39.5, 
        40,   # Normal points
        15, 60, 70  # Outliers
    ]
}

df = pd.DataFrame(data)

# Display the first few rows of the dataset
print(df.head())
```

### Step 4: Preprocessing the data

Before applying dimensionality reduction techniques, it is important to preprocess the data by scaling the features. Dimensionality reduction algorithms are sensitive to the scale of the features, so we'll use StandardScaler to normalize the data.

```python
# Normalize the data
scaler = StandardScaler()
scaled = scaler.fit_transform(df)

# Convert back to DataFrame for easy handling
df_scaled = pd.DataFrame(scaled, columns=['AnnualIncome', 'SpendingScore','Age'])

print(df_scaled.head())
```

Now the dataset is scaled, ensuring that each feature has a mean of 0 and a standard deviation of 1, which is essential for dimensionality reduction techniques such as PCA and t-SNE.

### Step 5: Implementing PCA

Now you will apply PCA to reduce the dimensionality of the dataset from three to two principal components. PCA will capture most of the variance in the data using fewer dimensions.

```python
# Apply PCA to reduce dimensions from 3 to 2
pca = PCA(n_components=2)
df_pca = pca.fit_transform(scaled)

# Convert back to DataFrame for easy handling
df_pca = pd.DataFrame(df_pca, columns=['PCA1', 'PCA2'])
print(df_pca.head())
```

`n_components = 2`: This reduces the dataset to two principal components that capture the majority of the variance in the original data.

### Step 6: Visualizing PCA results

Once you have implemented PCA, visualize the two principal components using a scatter plot. This will help you understand how PCA reduces dimensionality while retaining important information.

```python
# Plot the PCA components
plt.scatter(df_pca['PCA1'], df_pca['PCA2'])
plt.title('PCA - Dimensionality Reduction')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()
```

The scatter plot will show how PCA reduces the dataset to two dimensions while capturing the most important patterns from the original data.

### Step 7: Implementing t-SNE

Next, apply t-SNE to the scaled dataset. t-SNE is a nonlinear dimensionality reduction technique that is useful for visualizing high-dimensional data and preserving local structures.

```python
# Apply t-SNE to reduce dimensions to 2
tsne = TSNE(n_components=2, perplexity=3, random_state=42)
df_tsne = tsne.fit_transform(scaled)

# Convert the t-SNE result back to a DataFrame
df_tsne = pd.DataFrame(df_tsne, columns=['t-SNE1', 't-SNE2'])
print(df_tsne.head())
```

`n_components = 2`: This reduces the data to two components, preserving the local relationships between data points.

### Step 8: Visualizing t-SNE results

Now visualize the results of the t-SNE transformation using a scatter plot. This plot will help you understand how t-SNE reveals the structure and clusters in the data by preserving local similarities.

```python
# Plot the t-SNE components
plt.scatter(df_tsne['t-SNE1'], df_tsne['t-SNE2'])
plt.title('t-SNE - Dimensionality Reduction')
plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')
plt.show()
```

The t-SNE scatter plot will highlight clusters and patterns in the dataset that might not be apparent in higher dimensions.

## Conclusion

In this activity, you successfully:

- Preprocessed the dataset using StandardScaler to normalize the features.
- Applied PCA to reduce the dataset from three to two dimensions.
- Visualized the two principal components of PCA to understand the reduction.
- Applied t-SNE for nonlinear dimensionality reduction to reveal local structures.
- Visualized the t-SNE results to identify clusters in the data.

Both PCA and t-SNE are powerful tools for reducing the complexity of high-dimensional datasets and revealing important patterns that can be hard to detect in the original data.

By completing this activity, you now have hands-on experience applying dimensionality reduction techniques and visualizing their results. Experiment with these techniques on different datasets to explore how dimensionality reduction can improve data analysis and visualization.
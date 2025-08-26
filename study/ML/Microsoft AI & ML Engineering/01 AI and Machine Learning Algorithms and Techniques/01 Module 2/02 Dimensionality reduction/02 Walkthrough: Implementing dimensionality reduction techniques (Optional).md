# Walkthrough: Implementing dimensionality reduction techniques (Optional)

## Introduction

In this walkthrough, we will review the correct implementation of dimensionality reduction techniques, including principal component analysis (PCA) and t-distributed stochastic neighbor embedding (t-SNE). These techniques are used to reduce the number of features in a dataset while retaining as much of the relevant information as possible. You applied these techniques to a dataset containing customer data on annual income, spending score, and age.

By the end of this walkthrough, you'll be able to:

- Apply and interpret PCA to reduce a dataset's dimensionality and understand how it captures variance.
- Implement t-SNE for nonlinear dimensionality reduction, preserving local data structures and visualizing clusters.
- Differentiate between PCA and t-SNE, understanding their unique use cases and the types of data structures they are best suited for analyzing.

## Step-by-step guide:

### Step 1: Loading and preprocessing the dataset

In this activity, we used a dataset with customer data, including AnnualIncome, SpendingScore, and Age. Here's how we loaded and preprocessed the dataset:

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Sample dataset: Customer annual income, spending score, and age
data = = {
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

# Normalize the data
scaler = StandardScaler()
scaled = scaler.fit_transform(df)

# Convert back to DataFrame for easy handling
df_scaled = pd.DataFrame(scaled, columns=['AnnualIncome', 'SpendingScore','Age'])
```

StandardScaler was used to normalize the data. This ensures that all features have a mean of 0 and a standard deviation of 1, which is important for dimensionality reduction techniques such as PCA and t-SNE to work properly.

### Step 2: Applying PCA

You applied PCA to reduce the dimensionality of the dataset from three to two components. PCA is a linear technique that aims to maximize the variance captured by each component.

```python
from sklearn.decomposition import PCA

# Apply PCA to reduce dimensions from 3 to 2
pca = PCA(n_components=2)
df_pca = pca.fit_transform(scaled)

# Convert the PCA result back to a DataFrame
df_pca = pd.DataFrame(df_pca, columns=['PCA1', 'PCA2'])
print(df_pca.head())
```

n_components = 2: this reduces the original three features to two principal components.

PCA works by finding directions in the data that capture the maximum variance. In this case, the two principal components capture most of the variation in the original data while reducing the number of features.

### Step 3: Visualizing PCA results

You visualized the two principal components of the dataset using a scatter plot. This plot helps you see how PCA has compressed the dataset into two dimensions while retaining the most important information.

```python
import matplotlib.pyplot as plt

# Plot the PCA components
plt.scatter(df_pca['PCA1'], df_pca['PCA2'])
plt.title('PCA - Dimensionality Reduction')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()
```

Plot interpretation: The PCA scatter plot shows how the dataset has been reduced to two principal components while still maintaining much of the structure in the original data. This is useful for simplifying high-dimensional data for further analysis or visualization.

### Step 4: Applying t-SNE

Next, you applied t-SNE to reduce the dimensionality of the dataset. Unlike PCA, which focuses on capturing global variance, t-SNE is a nonlinear technique that preserves local relationships between data points, making it effective for visualizing clusters.

```python
from sklearn.manifold import TSNE

# Apply t-SNE to reduce dimensions to 2
tsne = TSNE(n_components=2, perplexity=3, random_state=42)
df_tsne = tsne.fit_transform(scaled)

# Convert the t-SNE result back to a DataFrame
df_tsne = pd.DataFrame(df_tsne, columns=['t-SNE1', 't-SNE2'])
print(df_tsne.head())
```

t-SNE maps the data points into a lower-dimensional space while preserving local structures, making it ideal for visualizing high-dimensional data and uncovering clusters.

### Step 5: Visualizing t-SNE results

You visualized the t-SNE results using a scatter plot to examine the structure of the data after dimensionality reduction.

```python
# Plot the t-SNE components
plt.scatter(df_tsne['t-SNE1'], df_tsne['t-SNE2'])
plt.title('t-SNE - Dimensionality Reduction')
plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')
plt.show()
```

Plot interpretation: The t-SNE scatter plot highlights clusters and patterns in the dataset that may not be immediately visible in higher dimensions. t-SNE is particularly useful for identifying clusters in data, such as groups of customers with similar behaviors.

### Step 6: Interpreting the results

#### PCA results

PCA reduced the dimensionality of the dataset while capturing the most variance from the original data. By projecting the data onto the first two principal components, you simplified the dataset to make it easier to visualize and analyze.

PCA is particularly useful when the goal is to reduce dimensionality while maintaining the overall variance structure of the data.

#### t-SNE results

t-SNE preserved local similarities in the data, making it effective for discovering hidden patterns or clusters.

Unlike PCA, which is a linear technique, t-SNE works well for nonlinear data structures and is often used for exploratory data analysis, especially when identifying clusters in high-dimensional data.

#### Choosing between PCA and t-SNE

- PCA is preferred when the goal is dimensionality reduction for further modeling or computational efficiency, as it reduces dimensions while preserving variance.
- t-SNE is ideal for visualization and understanding the local structure of high-dimensional data, making it great for identifying clusters or patterns.

## Conclusion

In this activity, you successfully:

- Preprocessed the dataset by normalizing the features using StandardScaler.
- Applied PCA to reduce the dataset from three to two dimensions and visualized the results.
- Applied t-SNE to reduce the dataset to two dimensions, revealing local structures and clusters.
- Interpreted the results of both PCA and t-SNE to understand how each technique reduces dimensionality while preserving important information.

Both PCA and t-SNE are powerful dimensionality reduction techniques with different use cases. PCA is ideal for linear dimensionality reduction, while t-SNE is great for visualizing complex, nonlinear structures.

By following this solution, you have successfully implemented dimensionality reduction techniques and gained hands-on experience in reducing and visualizing high-dimensional data.
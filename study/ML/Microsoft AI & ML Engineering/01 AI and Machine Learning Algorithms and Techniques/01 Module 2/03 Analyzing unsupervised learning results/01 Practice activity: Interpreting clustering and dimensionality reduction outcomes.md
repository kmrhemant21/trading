# Practice activity: Interpreting clustering and dimensionality reduction outcomes

## Introduction

In this activity, you will apply clustering and dimensionality reduction techniques to a dataset and interpret the outcomes. You will use both k-means clustering and principal component analysis (PCA) or t-distributed stochastic neighbor embedding (t-SNE) to analyze a dataset and uncover hidden patterns or clusters. The goal is to understand how these techniques work together to simplify data and reveal meaningful groupings.

By the end of this activity, you'll be able to:

- Apply k-means clustering: learn how this technique can help to identify distinct clusters in data.
- Perform dimensionality reduction: use PCA or t-SNE to reduce the dataset's dimensions, making it easier to visualize and analyze high-dimensional data while retaining significant patterns.
- Visualize and interpret results: create scatter plots to visualize clusters and dimensionality reduction outcomes, and interpret how these methods reveal hidden patterns or simplify complex datasets.

## Step-by-step guide:

### Step 1: Setting up the environment

Before starting the activity, ensure that you have the necessary libraries installed. You will need pandas, Scikit-learn, and Matplotlib. Install the required libraries using the following command:

```
pip install pandas scikit-learn matplotlib
```

### Step 2: Importing required libraries

Import the necessary libraries for data manipulation, clustering, dimensionality reduction, and visualization:

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
```

### Step 3: Loading the dataset

You will use a sample dataset containing customer information, including Annual Income, Spending Score, and Age. Load the dataset using the following code:

```python
# Create a sample dataset with customer annual income, spending score, and age
data = {'AnnualIncome': [15, 16, 17, 18, 19, 20, 22, 25, 30, 35],
    'SpendingScore': [39, 81, 6, 77, 40, 76, 94, 5, 82, 56],
    'Age': [20, 22, 25, 24, 35, 40, 30, 21, 50, 31]}

df = pd.DataFrame(data)

# Display the first few rows of the dataset
print(df.head())
```

### Step 4: Preprocessing the data

Before applying clustering and dimensionality reduction, it is essential to preprocess the data by scaling the features. Use StandardScaler to normalize the data:

```python
from sklearn.preprocessing import StandardScaler

# Normalize the dataset
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Convert the scaled data back into a DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=['AnnualIncome', 'SpendingScore', 'Age'])
print(df_scaled.head())
```

### Step 5: Applying k-means clustering

Now, apply k-means clustering to group the customers into clusters based on their spending behavior and income. Use k = 3 clusters for this task:

```python
from sklearn.cluster import KMeans

# Apply K-Means with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
df_scaled['KMeans_Cluster'] = kmeans.fit_predict(df_scaled)

# Display the cluster assignments
print(df_scaled.head())
```

The k-means algorithm will assign each customer to one of the three clusters.

The KMeans_Cluster column will indicate the cluster assignment for each customer.

### Step 6: Visualizing k-means clusters

Visualize the clusters using a scatter plot. Color the data points based on their cluster assignment:

```python
# Visualize the K-Means clusters
plt.scatter(df_scaled['AnnualIncome'], df_scaled['SpendingScore'], c=df_scaled['KMeans_Cluster'], cmap='viridis')
plt.title('K-Means Clustering of Customers')
plt.xlabel('Annual Income (in thousands)')
plt.ylabel('Spending Score (1-100)')
plt.show()
```

This plot will show how k-means has grouped customers into clusters based on their income and spending score.

### Step 7: Applying dimensionality reduction (PCA or t-SNE)

Next, reduce the dimensionality of the dataset to two components for visualization. You can use PCA or t-SNE.

For PCA:

```python
# Apply PCA to reduce dimensions to 2
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Convert the PCA result back to a DataFrame
df_pca = pd.DataFrame(df_pca, columns=['PCA1', 'PCA2'])
print(df_pca.head())
```

For t-SNE:

```python
from sklearn.manifold import TSNE

# Set perplexity to a value smaller than the number of samples
tsne = TSNE(n_components=2, perplexity=5, random_state=42)

df_tsne = tsne.fit_transform(df_scaled)

# Convert the t-SNE result back to a DataFrame
df_tsne = pd.DataFrame(df_tsne, columns=['t-SNE1', 't-SNE2'])
print(df_tsne.head())
```

### Step 8: Visualizing dimensionality reduction results

Visualize the two-dimensional representation of the dataset created by PCA or t-SNE.

For PCA:

```python
# Visualize the PCA components
plt.scatter(df_pca['PCA1'], df_pca['PCA2'], c=df_scaled['KMeans_Cluster'], cmap='viridis')
plt.title('PCA - Dimensionality Reduction with K-Means Clusters')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()
```

For t-SNE:

```python
# Visualize the t-SNE components
plt.scatter(df_tsne['t-SNE1'], df_tsne['t-SNE2'], c=df_scaled['KMeans_Cluster'], cmap='viridis')
plt.title('t-SNE - Dimensionality Reduction with K-Means Clusters')
plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')
plt.show()
```

The scatter plot will show the clusters in a lower-dimensional space. Color the points based on their k-means cluster assignment to interpret how the clusters are distributed after dimensionality reduction.

### Step 9: Interpreting the results

In this step, you will interpret the clustering and dimensionality reduction outcomes. Answer the following questions:

- How well did k-means group similar customers? Were the clusters distinct or overlapping?
- How did PCA or t-SNE help simplify the data? Were any hidden patterns or clusters revealed that were not obvious in the original data?
- If you used PCA, how much variance was captured by the two principal components? Did they effectively summarize the data?
- If you used t-SNE, how well did the method preserve local structures? Did you see any meaningful clusters or patterns?

Document your interpretations and insights in a short paragraph.

## Conclusion

By completing this activity, you have successfully:

- Preprocessed a dataset using StandardScaler to normalize the features.
- Applied k-means clustering to group similar data points.
- Reduced the dimensionality of the dataset using PCA or t-SNE.
- Visualized and interpreted the clustering and dimensionality reduction outcomes.

Both clustering and dimensionality reduction are powerful techniques for uncovering hidden patterns in data. While clustering reveals groupings of similar data points, dimensionality reduction simplifies the data, making it easier to visualize and analyze. 

This activity provided hands-on experience with clustering and dimensionality reduction techniques, allowing you to explore how these methods work together to make sense of high-dimensional data.
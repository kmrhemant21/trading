# Practice activity: Implementing unsupervised learning methods

## Introduction

In this activity, you will apply various unsupervised learning methods—including k-means clustering, density-based spatial clustering of applications with noise (DBSCAN), and principal component analysis (PCA)—to a real dataset. 

The goal is to explore how these techniques can uncover hidden patterns, groupings, or structures without the need for labeled data.

By the end of this activity, you'll be able to:

- **Apply k-means and DBSCAN clustering**: Implement k-means to partition data into a predefined number of clusters, and use DBSCAN to identify clusters and outliers based on density.

- **Perform dimensionality reduction with PCA**: Apply PCA to reduce the dimensionality of the dataset, simplifying it while retaining most of the variance for visualization and analysis.

- **Visualize and interpret unsupervised learning results**: Create scatterplots to visualize clusters formed by k-means and DBSCAN, and analyze how PCA transforms the data for easier interpretation of underlying patterns.

## Step-by-step guide:

### Step 1: Setting up the environment

Ensure that you have the necessary libraries installed before starting the activity. You will need pandas, Scikit-learn, and Matplotlib for this exercise. You can install the required packages using the following command:

```python
pip install pandas scikit-learn matplotlib
```

### Step 2: Importing required libraries

First, import the necessary libraries for data manipulation, clustering, dimensionality reduction, and visualization:

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
```

### Step 3: Loading the dataset

We will use a sample dataset that contains customer data, including Annual Income, Spending Score, and Age. Load the dataset into a pandas DataFrame:

```python
# Sample dataset: Customer annual income, spending score, and age
data = {'AnnualIncome': [15, 16, 17, 18, 19, 20, 22, 25, 30, 35],
    'SpendingScore': [39, 81, 6, 77, 40, 76, 94, 5, 82, 56],
    'Age': [20, 22, 25, 24, 35, 40, 30, 21, 50, 31]}

df = pd.DataFrame(data)

# Display the first few rows of the dataset
print(df.head())
```

### Step 4: Preprocessing the data

Before applying unsupervised learning methods, you need to normalize the data. Use StandardScaler to scale the features so that they all have the same scale, which is important for algorithms such as k-means and PCA:

```python
# Normalize the dataset using StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Convert the scaled data back into a DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=['AnnualIncome', 'SpendingScore', 'Age'])
print(df_scaled.head())
```

### Step 5: Task 1: Apply k-means clustering

For this task, you will apply k-means clustering to group the customers into k = 3 clusters based on their income, spending score, and age. k-means partitions the data into k clusters by minimizing the distance between data points and the cluster centroids.

```python
from sklearn.cluster import KMeans

# Apply K-Means clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
df_scaled['KMeans_Cluster'] = kmeans.fit_predict(df_scaled)

# Display the cluster assignments
print(df_scaled.head())
```

KMeans_Cluster will contain the cluster assignment for each data point. Each customer will be assigned to one of the three clusters based on their features.

### Step 6: Task 2: Apply DBSCAN clustering

Next, apply DBSCAN, a density-based clustering algorithm that identifies clusters based on the density of data points. This method is useful for detecting irregularly shaped clusters and outliers. You will experiment with eps (the maximum distance between two points to be considered neighbors) and min_samples (the minimum number of points required to form a dense cluster).

```python
from sklearn.cluster import DBSCAN

# Apply DBSCAN with predefined parameters
dbscan = DBSCAN(eps=0.5, min_samples=2)
df_scaled['DBSCAN_Cluster'] = dbscan.fit_predict(df_scaled)

# Display the cluster assignments and noise (-1)
print(df_scaled.head())
```

DBSCAN_Cluster will contain the cluster assignments, and any points labeled –1 represent outliers or noise that DBSCAN identified as not belonging to any cluster.

### Step 7: Task 3: Apply PCA

For dimensionality reduction, apply PCA to reduce the number of features from three to two, while retaining as much variance in the data as possible. PCA helps simplify the dataset and allows for easier visualization of clusters.

```python
from sklearn.decomposition import PCA

# Apply PCA to reduce dimensions from 3 to 2
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Convert the PCA result back to a DataFrame for easy handling
df_pca = pd.DataFrame(df_pca, columns=['PCA1', 'PCA2'])
print(df_pca.head())
```

### Step 8: Visualizing the results

#### Visualize k-means clusters:

To visualize the clusters formed by k-means, create a scatterplot of Annual Income and Spending Score, with each point colored according to its cluster assignment.

```python
# Plot K-Means clusters
plt.scatter(df_scaled['AnnualIncome'], df_scaled['SpendingScore'], c=df_scaled['KMeans_Cluster'], cmap='viridis')
plt.title('K-Means Clustering of Customers')
plt.xlabel('Annual Income (in thousands)')
plt.ylabel('Spending Score (1-100)')
plt.show()
```

#### Visualize DBSCAN clusters:

Similarly, visualize the clusters and outliers detected by DBSCAN.

```python
# Plot DBSCAN clusters
plt.scatter(df_scaled['AnnualIncome'], df_scaled['SpendingScore'], c=df_scaled['DBSCAN_Cluster'], cmap='rainbow')
plt.title('DBSCAN Clustering of Customers')
plt.xlabel('Annual Income (in thousands)')
plt.ylabel('Spending Score (1-100)')
plt.show()
```

#### Visualize PCA results:

Finally, create a scatterplot to visualize the two principal components obtained from PCA.

```python
# Plot PCA components
plt.scatter(df_pca['PCA1'], df_pca['PCA2'], c=df_scaled['KMeans_Cluster'], cmap='viridis')
plt.title('PCA - Dimensionality Reduction with K-Means Clusters')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()
```

### Step 9: Interpreting the results

Now that you've visualized the results of each method, answer the following questions to interpret your findings:

1. How well did k-means and DBSCAN cluster the data? Were there any significant differences in how they grouped customers?

2. Did DBSCAN identify any outliers? How does this impact the clustering results?

3. How did PCA help simplify the dataset? Did the principal components retain most of the variance?

4. Compare the cluster structures between the original feature space and the two-dimensional space produced by PCA. Do the clusters look similar?

## Conclusion

By completing this activity, you have gained hands-on experience implementing key unsupervised learning methods—k-means clustering, DBSCAN, and PCA. You explored how clustering algorithms group data points and how dimensionality reduction can simplify and visualize complex datasets.

These techniques are critical for tasks such as customer segmentation, anomaly detection, and high-dimensional data simplification for further analysis.
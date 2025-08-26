# Walkthrough: Clustering and visualization (Optional)

## Introduction

In this walkthrough, we will review the correct implementation of the clustering algorithms (k-means and density-based spatial clustering of algorithms with noise (DBSCAN)) and the visualization of their results from the activity. You applied both algorithms to a dataset containing AnnualIncome and SpendingScore features to understand how they group data points based on different techniques.

By the end of this walkthrough, you'll be able to:

- Normalize and preprocess data to ensure equal contribution of features to clustering algorithms.
- Apply k-means and DBSCAN clustering algorithms to a dataset, understanding how each algorithm groups data points based on different principles.
- Visualize and interpret clustering results using scatterplots, recognizing the differences between partition-based clustering (k-means) and density-based clustering (DBSCAN) along with their handling of outliers.

## Step-by-step guide:

### Step 1: Loading and preparing the dataset

In the activity, we started by loading a dataset with customer data containing AnnualIncome and SpendingScore. Here's how you loaded it into a pandas DataFrame:

```python
import pandas as pd

# Sample dataset: Customer annual income and spending score
data = {'AnnualIncome': [
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
    ]}


df = pd.DataFrame(data)
print(df.head())
```

This dataset contains 15 customers and their respective AnnualIncome (in thousands) and SpendingScore (out of 100).

### Step 2: Preprocessing the data

To ensure that the clustering algorithms treat each feature equally, you normalized the dataset using StandardScaler. Scaling transforms the data so that each feature has a mean of 0 and a standard deviation of 1.

```python
from sklearn.preprocessing import StandardScaler

# Normalize the dataset
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Convert the scaled data back into a DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=['AnnualIncome', 'SpendingScore'])
print(df_scaled.head())
```

Now, the AnnualIncome and SpendingScore features are normalized, ensuring they contribute equally to the clustering process.

### Step 3: Implementing k-means clustering

For k-means clustering, you set the number of clusters (k) to 3 and applied the algorithm to the scaled dataset:

```python
from sklearn.cluster import KMeans

# Initialize and fit K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df_scaled)

# Add K-Means cluster labels to the original DataFrame
df['KMeans_Cluster'] = kmeans.labels_
print(df.head())
```

n_clusters = 3: you set the algorithm to create three clusters.

After fitting the k-means algorithm, each data point was assigned to one of the three clusters, and the cluster labels were added to the DataFrame.

### Step 4: Visualizing k-means clusters

You visualized the k-means clustering results with a scatterplot. Each data point was colored based on its cluster assignment:

```python
import matplotlib.pyplot as plt

# Plot K-Means clusters
plt.scatter(df['AnnualIncome'], df['SpendingScore'], c=df['KMeans_Cluster'], cmap='viridis')
plt.title('K-Means Clustering of Customers')
plt.xlabel('Annual Income (in thousands)')
plt.ylabel('Spending Score (1-100)')
plt.show()
```

Plot interpretation: The plot showed how k-means grouped the data points into three clusters. The algorithm partitions the dataset based on the similarity of data points. Customers with similar Annual Income and Spending Score were grouped into the same cluster.

### Step 5: Implementing DBSCAN clustering

Next, you implemented DBSCAN, a density-based clustering algorithm, which does not require specifying the number of clusters but instead uses eps and min_samples to define clusters. You started with eps = 0.5 and min_samples = 3:

```python
from sklearn.cluster import DBSCAN

# Initialize and fit DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=3)
dbscan.fit(df_scaled)

# Add DBSCAN cluster labels to the original dataframe
df['DBSCAN_Cluster'] = dbscan.labels_
print(df.head())
```

eps: This defines the maximum distance between two points to be considered as neighbors.

min_samples: This represents the minimum number of data points required to form a dense region (cluster).

Points labeled –1 represent noise (outliers) that DBSCAN detected as not belonging to any cluster.

### Step 6: Visualizing DBSCAN clusters and outliers

After running DBSCAN, you visualized the clusters and outliers (if any) using a scatterplot:

```python
# Plot DBSCAN clusters
plt.scatter(df['AnnualIncome'], df['SpendingScore'], c=df['DBSCAN_Cluster'], cmap='rainbow')
plt.title('DBSCAN Clustering of Customers')
plt.xlabel('Annual Income (in thousands)')
plt.ylabel('Spending Score (1-100)')
plt.show()
```

Plot interpretation: In the plot, different colors represented different clusters, and noise points (outliers) were shown in a separate color. DBSCAN formed clusters based on the density of data points rather than partitioning the data into a predefined number of clusters such as k-means.

### Step 7: Interpreting the results

After visualizing both k-means and DBSCAN results, you can interpret how each algorithm behaves:

- **k-means**: This algorithm partitioned the dataset into three equal clusters based on the proximity of data points. This algorithm works well when the clusters are roughly spherical and equal in size.

- **DBSCAN**: This algorithm identified clusters based on the density of points and labeled outliers as noise (if any). DBSCAN is more flexible for detecting clusters of arbitrary shapes and works well when outliers are present.

#### Parameter tuning

- For k-means, you can experiment with different values of k to find the best number of clusters for your dataset.
- For DBSCAN, adjusting eps and min_samples can help improve clustering by including more points or reducing noise.

## Conclusion

In this activity, you successfully:

- Preprocessed the dataset by normalizing the features using StandardScaler.
- Implemented both k-means and DBSCAN clustering algorithms.
- Visualized the clusters and outliers using scatterplots.
- Interpreted the differences between partition-based (k-means) and density-based (DBSCAN) clustering methods.

This exercise demonstrates how different clustering techniques can reveal patterns in data and highlights the strengths of each algorithm. You now have a deeper understanding of clustering techniques and visualization, and how to apply them to real-world datasets. 

Continue exploring by adjusting parameters such as k in k-means and eps/min_samples in DBSCAN to see how they impact the clustering results.
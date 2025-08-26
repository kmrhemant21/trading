# Walkthrough: Implementing DBSCAN clustering (Optional)

## Introduction

In this walkthrough, we will review the correct implementation of the density-based spatial clustering of applications with noise (DBSCAN) algorithm and analyze the clustering results from the activity. DBSCAN is a versatile unsupervised learning algorithm that clusters data based on density and is particularly useful for detecting outliers or clusters of arbitrary shapes.

By the end of this walkthrough, you'll be able to:

- Load and preprocess data to prepare it for DBSCAN clustering, ensuring all features contribute equally to the process.
- Implement the DBSCAN algorithm to detect clusters of varying densities and identify outliers in a dataset, using key parameters.
- Visualize and interpret DBSCAN clustering results with scatterplots, and adjust DBSCAN parameters to fine-tune clustering outcomes and improve the identification of clusters and noise points.

## Step-by-step guide:

### Step 1: Loading and preparing the dataset

We used a sample dataset containing two features: AnnualIncome and SpendingScore. The first step was to load the dataset into a pandas DataFrame.

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

This dataset contains two features:

- **AnnualIncome**: the annual income of the customer in thousands.
- **SpendingScore**: a score between 1 and 100 representing customer spending behavior.

### Step 2: Preprocessing the data

Before applying DBSCAN, we scaled the data using StandardScaler to ensure that all features contributed equally to the clustering process.

```python
from sklearn.preprocessing import StandardScaler

# Normalize the dataset using StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Convert the scaled data back into a DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=['AnnualIncome', 'SpendingScore'])
print(df_scaled.head())
```

StandardScaler was used to scale both features to have a mean of 0 and a standard deviation of 1, making them comparable in the clustering process.

### Step 3: Implementing DBSCAN clustering

We implemented the DBSCAN algorithm, specifying the parameters eps (the maximum distance between two points to be considered neighbors) and min_samples (the minimum number of points required to form a dense region). We used eps = 0.5 and min_samples = 3 as our initial values.

```python
from sklearn.cluster import DBSCAN

# Initialize DBSCAN with eps and min_samples
dbscan = DBSCAN(eps=0.5, min_samples=3)

# Fit the model to the scaled data
dbscan.fit(df_scaled)

# Assign cluster labels to the data points
df['Cluster'] = dbscan.labels_

# Display the first few rows with cluster labels
print(df.head())
```

- Cluster column contains the cluster labels assigned to each data point.
- Points labeled –1 represent noise (outliers), which DBSCAN detects as points that do not belong to any cluster.

### Step 4: Visualizing the clusters

After applying DBSCAN, we visualized the clusters using a scatterplot. Each data point is colored according to its assigned cluster, with outliers (noise points) displayed in a separate color.

```python
import matplotlib.pyplot as plt

# Plot the clusters
plt.scatter(df['AnnualIncome'], df['SpendingScore'], c=df['Cluster'], cmap='rainbow')
plt.title('DBSCAN Clustering of Customers')
plt.xlabel('Annual Income (in thousands)')
plt.ylabel('Spending Score (1-100)')
plt.show()
```

Different colors represent different clusters, and noise points (if any) are represented by a separate color (often black or gray).

In this visualization, you can observe how DBSCAN has grouped the data points into clusters based on their density.

### Step 5: Interpreting the results

#### Cluster labels

- Data points that were assigned a cluster label of –1 are outliers or noise points.
- The remaining data points are grouped into clusters based on density. For example, data points with similar income and spending scores are likely to be clustered together.

#### Parameter tuning

- If too many points are classified as noise or if the clusters are too sparse, adjusting the eps and min_samples parameters can improve the results. For example, increasing eps allows more points to be included in a cluster, while decreasing min_samples makes it easier for points to form clusters.

### Step 6: Tuning the DBSCAN parameters

To better understand how DBSCAN behaves with different parameters, you could experiment with changing eps and min_samples. Here is an example in which we increase eps to 0.7 to reduce the number of outliers:

```python
# Increase eps to 0.7 and refit DBSCAN
dbscan = DBSCAN(eps=0.7, min_samples=3)
dbscan.fit(df_scaled)
df['Cluster'] = dbscan.labels_

# Plot the updated clusters
plt.scatter(df['AnnualIncome'], df['SpendingScore'], c=df['Cluster'], cmap='rainbow')
plt.title('DBSCAN Clustering with eps=0.7')
plt.xlabel('Annual Income (in thousands)')
plt.ylabel('Spending Score (1-100)')
plt.show()
```

By increasing eps, more data points are included in the clusters, and fewer points are classified as noise.

Adjusting the eps and min_samples parameters allows you to fine-tune DBSCAN's behavior for different datasets.

### Step 7: Visualizing the final results

Once you've tuned the parameters, you can visualize the final clustering results. Here is an example of what your plot might look like after adjusting eps and min_samples:

- **Cluster 0**: Customers with relatively low income and low spending scores
- **Cluster 1**: Customers with moderate-to-high income and moderate spending scores
- **Cluster 2**: Customers with high income and high spending scores
- **Noise points**: Outliers or customers who don't fit into any cluster based on their income or spending behavior

## Conclusion

You successfully:

- Implemented the DBSCAN clustering algorithm to group customers based on their annual income and spending score.
- Preprocessed the data by normalizing it using StandardScaler.
- Visualized the clusters and identified outliers using a scatterplot.
- Tuned the eps and min_samples parameters to adjust the clustering behavior and improve results.

DBSCAN is a powerful algorithm for detecting clusters of arbitrary shapes and identifying noise points (outliers). It does not require you to specify the number of clusters in advance, making it highly useful for exploratory data analysis. Continue experimenting with different datasets and parameter settings to deepen your understanding of DBSCAN clustering.

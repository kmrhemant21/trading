# Walkthrough: Implementing k-means clustering (Optional)

## Introduction

In this walkthrough, we will review the correct implementation of the k-means clustering algorithm and interpret the results from the activity. This fundamental clustering algorithm groups data points into clusters based on feature similarity. 

This guide will walk through each step of the lab, from loading the data to visualizing the clusters and identifying the optimal number of clusters using the elbow method.

By the end of this walkthrough, you'll be able to:

- Correctly implement k-means clustering: Understand the step-by-step process of applying k-means to a dataset, including initializing, fitting the model, and assigning cluster labels.

- Preprocess and visualize clusters: Use techniques such as data scaling and scatterplots to visualize and interpret the resulting clusters effectively.

- Determine the optimal number of clusters: Apply the elbow method to identify the optimal number of clusters, enhancing your ability to make data-driven decisions in clustering analysis.

## Step-by-step guide:

### Step 1. Loading and preparing the dataset

In the lab, we used a sample dataset with AnnualIncome and SpendingScore features for customer segmentation. The first step was to load the dataset into a pandas DataFrame.

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

This dataset consists of:

- AnnualIncome: annual income of the customer in thousands.

- SpendingScore: a score representing the customer’s spending behavior.

### Step 2. Preprocessing the data

Since k-means is sensitive to the scale of the features, we used StandardScaler to normalize the data. Scaling ensures that all features contribute equally to the clustering process.

```python
from sklearn.preprocessing import StandardScaler

# Normalize the dataset
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Convert the scaled data back into a DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=['AnnualIncome', 'SpendingScore'])
print(df_scaled.head())
```

In this step, both AnnualIncome and SpendingScore are standardized, meaning that they now have a mean of 0 and a standard deviation of 1.

### Step 3. Implementing k-means clustering

We initialized the k-means algorithm with a predefined number of clusters (k = 3) and applied it to the normalized dataset. The algorithm assigned each data point to a cluster based on its proximity to the cluster centroid.

```python
from sklearn.cluster import KMeans

# Initialize the KMeans algorithm with k clusters
k = 3  # Starting with 3 clusters
kmeans = KMeans(n_clusters=k, random_state=42)

# Fit the model and predict cluster labels
kmeans.fit(df_scaled)
df['Cluster'] = kmeans.labels_

# Display the first few rows with cluster assignments
print(df.head())
```

The Cluster column in the DataFrame now contains the cluster label assigned to each data point. Data points in the same cluster have similar AnnualIncome and SpendingScore characteristics.

### Step 4. Visualizing the clusters

To better understand the results, we plotted the clusters using a scatterplot, with different colors representing different clusters. The x-axis represents AnnualIncome, and the y-axis represents SpendingScore.

```python
import matplotlib.pyplot as plt

# Plot the clusters
plt.scatter(df['AnnualIncome'], df['SpendingScore'], c=df['Cluster'], cmap='viridis')
plt.title('K-Means Clustering of Customers')
plt.xlabel('Annual Income (in thousands)')
plt.ylabel('Spending Score (1-100)')
plt.show()
```

This plot visually shows how the customers were grouped into clusters. You can observe that customers with similar income and spending patterns are grouped together.

### Step 5. Finding the optimal number of clusters (elbow method)

To find the optimal number of clusters, we applied the elbow method, which helps identify the point where adding more clusters no longer significantly improves the clustering performance. This is done by calculating the within-cluster sum of squares (WCSS) for different values of k.

```python
# Calculate the WCSS for different values of k
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

# Plot the WCSS to visualize the Elbow
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
```

In the elbow plot, the x-axis represents the number of clusters (k), and the y-axis represents the WCSS. The "elbow" point is the value of k where the reduction in WCSS starts to slow down. In this case, you may observe an elbow around k = 3 or k = 4, indicating that these values are optimal for this dataset.

### Step 6. Interpreting the results

After running the k-means algorithm and visualizing the clusters, you will get:

- Cluster 0: customers with relatively low income but moderate-to-high spending scores.

- Cluster 1: customers with low income and low spending scores.

- Cluster 2: customers with high income and high spending scores.

This segmentation can be used for targeted marketing, where you might prioritize high-income, high-spending customers for premium offers.

## Conclusion

In this activity, you successfully:

- Implemented the k-means clustering algorithm on a customer dataset.

- Preprocessed the data by scaling it with StandardScaler.

- Visualized the clusters using a scatterplot.

- Applied the elbow method to determine the optimal number of clusters.

This process demonstrated how clustering can help uncover patterns in data, such as grouping customers based on income and spending behavior. Continue experimenting with different datasets and values of k to deepen your understanding of clustering.
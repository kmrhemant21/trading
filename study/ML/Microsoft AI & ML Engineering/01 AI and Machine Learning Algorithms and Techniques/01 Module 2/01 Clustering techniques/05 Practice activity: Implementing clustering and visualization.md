# Practice activity: Implementing clustering and visualization

## Introduction

In this activity, you will implement clustering algorithms and visualize the results using Python. You will apply k-means and density-based spatial clustering of algorithms with noise (DBSCAN) clustering algorithms to a dataset and then create visualizations to better understand the clustering outcomes. By the end of this activity, you will be able to cluster and interpret data visually using k-means and DBSCAN clustering algorithms.

By the end of this reading, you'll be able to:

- Preprocess and normalize data to ensure that features contribute equally to clustering algorithms.
- Apply and compare k-means and DBSCAN clustering algorithms to a dataset, understanding the differences in how they group data points and detect outliers.
- Visualize clustering results with scatterplots to interpret how k-means forms fixed clusters and DBSCAN identifies clusters of varying shapes and outliers.

## Step-by-step guide:

### Step 1: Setting up the environment

Before starting, ensure that you have the necessary libraries installed. You will need pandas, Scikit-learn, and Matplotlib.

Install the required libraries using the following command:

```
pip install pandas scikit-learn matplotlib
```

### Step 2: Importing required libraries

Import the necessary libraries for data manipulation, clustering, and visualization:

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
```

- pandas for data manipulation
- Scikit-learn for clustering algorithms
- Matplotlib for plotting and visualizing clusters
- StandardScaler to normalize the dataset

### Step 3: Loading the dataset

You will use a fictional dataset that contains customer data based on AnnualIncome and SpendingScore. Load the dataset using the following code:

```python
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

# Display the first few rows of the dataset
print(df.head())
```

This dataset includes:

- **AnnualIncome**: The customer's annual income (in thousands).
- **SpendingScore**: A score between 1 and 100 representing customer spending behavior.

### Step 4: Preprocessing the data

Before clustering, it is essential to preprocess the data by normalizing the features. Clustering algorithms are sensitive to the scale of the features, so we'll use StandardScaler to scale the data:

```python
# Normalize the dataset using StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Convert the scaled data back into a DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=['AnnualIncome', 'SpendingScore'])
print(df_scaled.head())
```

StandardScaler scales the features so that both AnnualIncome and SpendingScore have a mean of 0 and a standard deviation of 1. This ensures that each feature contributes equally to the clustering process.

### Step 5: Implementing k-means clustering

Now, you will apply the k-means clustering algorithm to the dataset. This algorithm requires specifying the number of clusters (k), which you will set to 3 in this case.

```python
# Initialize and fit K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df_scaled)

# Add cluster labels to the original dataframe
df['KMeans_Cluster'] = kmeans.labels_

# Display the first few rows with cluster labels
print(df.head())
```

- n_clusters: This parameter defines the number of clusters (in this case, k = 3).
- kmeans.labels_: This provides the cluster label assigned to each data point.

### Step 6: Visualizing k-means clusters

Once you have implemented k-means, visualize the results with a scatterplot where each point is colored according to its assigned cluster:

```python
# Plot K-Means clusters
plt.scatter(df['AnnualIncome'], df['SpendingScore'], c=df['KMeans_Cluster'], cmap='viridis')
plt.title('K-Means Clustering of Customers')
plt.xlabel('Annual Income (in thousands)')
plt.ylabel('Spending Score (1-100)')
plt.show()
```

The plot will display how k-means has grouped the customers into three clusters based on their income and spending scores.

### Step 7: Implementing DBSCAN clustering

Next, implement the DBSCAN clustering algorithm. DBSCAN groups data points based on their density and can detect outliers. The algorithm requires setting two key parameters:

- **eps**: The maximum distance between two points to be considered neighbors.
- **min_samples**: The minimum number of points needed to form a dense region (cluster).

```python
# Initialize and fit DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=3)
dbscan.fit(df_scaled)

# Add DBSCAN cluster labels to the original dataframe
df['DBSCAN_Cluster'] = dbscan.labels_

# Display the first few rows with cluster labels
print(df.head())
```

Points labeled –1 represent noise (outliers) detected by DBSCAN.

You can adjust eps and min_samples to change the sensitivity of the clustering.

### Step 8: Visualizing DBSCAN clusters and outliers

Now that DBSCAN has been applied, visualize the clusters and any outliers in the dataset:

```python
# Plot DBSCAN clusters
plt.scatter(df['AnnualIncome'], df['SpendingScore'], c=df['DBSCAN_Cluster'], cmap='rainbow')
plt.title('DBSCAN Clustering of Customers')
plt.xlabel('Annual Income (in thousands)')
plt.ylabel('Spending Score (1-100)')
plt.show()
```

Outliers (noise points) will appear in a separate color from the clusters.

The plot will help you understand how DBSCAN forms clusters based on density and identifies points that do not belong to any cluster.

## Conclusion

In this activity, you successfully:

- Loaded and preprocessed a dataset using StandardScaler.
- Applied two different clustering algorithms: k-means and DBSCAN.
- Visualized the results of both algorithms to understand the differences between clustering techniques and their outputs.

k-means clusters the data into a specified number of equal-sized groups, while DBSCAN detects clusters of varying shapes and identifies outliers. Experiment with different values of k, eps, and min_samples to see how these parameters affect the clustering results.

By completing this activity, you've gained hands-on experience with algorithm clustering and visualization techniques. Continue practicing to explore how clustering can reveal insights in various types of datasets.

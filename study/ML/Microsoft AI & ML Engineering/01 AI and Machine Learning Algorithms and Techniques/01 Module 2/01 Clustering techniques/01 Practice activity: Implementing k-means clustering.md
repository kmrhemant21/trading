# Practice activity: Implementing k-means clustering

## Introduction
In this activity, you will implement the k-means clustering algorithm using Python. By the end of this activity, you will have hands-on experience in applying k-means to a dataset, visualizing clusters, and interpreting the results. This powerful clustering algorithm partitions data into k distinct clusters based on feature similarity.

By the end of this activity, you'll be able to:

*   **Implement k-means clustering**: apply the k-means algorithm to partition data into clusters based on feature similarity.
*   **Preprocess and visualize data**: normalize the dataset using `StandardScaler`, and create visualizations to understand the resulting clusters.
*   **Determine the optimal number of clusters**: use the elbow method to identify the optimal number of clusters for the k-means algorithm.

## Step-by-step guide:

### Step 1. Setting up the environment
Before you begin, ensure that you have the necessary libraries installed. You will need pandas, Scikit-learn, and Matplotlib.

Install these libraries using the following command (if you haven’t already):
```bash
pip install pandas scikit-learn matplotlib
```

### Step 2. Importing required libraries
Start by importing the libraries needed for data handling, clustering, and visualization.

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
```
*   **pandas** for data manipulation
*   **Matplotlib** for plotting and visualization
*   **k-means** from Scikit-Learn to apply the k-means clustering algorithm
*   **StandardScaler** for normalizing the dataset before clustering

### Step 3. Loading the dataset
You will use a sample dataset for this activity. For this example, we will use a fictional customer dataset where each row represents a customer, and the features are `AnnualIncome` and `SpendingScore`.

Load the dataset:

```python
# Create a sample dataset with customer annual income and spending score
data = {'AnnualIncome': [
        15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5, 
        20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5, 24, 24.5, 
        25, 25.5, 26, 26.5, 27, 27.5, 28, 28.5, 29, 29.5, 
        30, 30.5, 31, 31.5, 32, 32.5, 33, 33.5, 34, 34.5, 
        35,   # Normal points
        80, 85, 90  # Outliers
    ],
    'SpendingScore': [
        39, 41, 35, 45, 30, 50, 25, 55, 20, 60,
        15, 65, 10, 70, 5, 75, 4, 80, 3, 85,
        2, 90, 1, 95, 2, 98, 3, 100, 4, 97,
        5, 96, 6, 94, 7, 92, 8, 91, 9, 89,
        10,  # Normal points
        15, 20, 10 # Outliers
    ]}
df = pd.DataFrame(data)
print(df.head())
```

This dataset contains two features:

*   **AnnualIncome**: Annual income of the customer in thousands.
*   **SpendingScore**: A score assigned based on customer behavior, from 1 (low) to 100 (high).

### Step 4. Preprocessing the data
Clustering algorithms are sensitive to the scale of the features, so it’s essential to normalize the data. We’ll use `StandardScaler` to scale the features.

```python
# Normalize the dataset using StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Convert the scaled data back into a DataFrame for easier handling
df_scaled = pd.DataFrame(df_scaled, columns=['AnnualIncome', 'SpendingScore'])
print(df_scaled.head())
```

### Step 5. Implementing k-means clustering
Now it’s time to implement the k-means clustering algorithm. First, we’ll initialize the algorithm with a predefined number of clusters (k), fit it to the data, and then assign each data point to a cluster.

```python
# Initialize the KMeans algorithm with k clusters
k = 3  # You can start with 3 clusters
kmeans = KMeans(n_clusters=k, random_state=42)

# Fit the model and assign cluster labels
kmeans.fit(df_scaled)
df['Cluster'] = kmeans.labels_

# Display the first few rows with cluster assignments
print(df.head())
```

*   **k**: The number of clusters. You can start with an arbitrary value of k = 3 and later evaluate the performance to choose the best number of clusters.
*   **kmeans.labels_**: The labels assigned to each data point, indicating the cluster they belong to.

### Step 6. Visualizing the clusters
Now that we’ve clustered the data, let’s visualize the clusters using a scatterplot. We’ll color the data points based on their assigned cluster.

```python
# Plot the clusters
plt.scatter(df['AnnualIncome'], df['SpendingScore'], c=df['Cluster'], cmap='viridis')
plt.title('K-Means Clustering of Customers')
plt.xlabel('Annual Income (in thousands)')
plt.ylabel('Spending Score (1-100)')
plt.show()
```
This scatterplot shows how the data points are grouped into clusters based on their annual income and spending score.

### Step 7. Finding the optimal number of clusters (optional)
One way to determine the optimal number of clusters (k) is by using the elbow method, which involves plotting the within-cluster sum of squares (WCSS) against the number of clusters.

Here’s how to apply the elbow method:

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
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.show()
```
The elbow method helps determine the optimal k by identifying the point where the WCSS stops decreasing significantly, forming an "elbow" in the graph.

### Step 8. Interpreting the results
After running k-means and visualizing the clusters:

*   Check whether the clusters make sense based on the features. For example, customers with similar income and spending behaviors should belong to the same cluster.
*   Use the cluster assignments to make business decisions, such as targeting specific customer segments.

## Conclusion
In this activity, you successfully implemented k-means clustering to group customers based on their annual income and spending score. You also learned how to scale data, fit a clustering algorithm, and visualize the results. This is a versatile algorithm with applications in customer segmentation, image compression, and more.

Feel free to experiment with different values of k and datasets to better understand how k-means works in practice.
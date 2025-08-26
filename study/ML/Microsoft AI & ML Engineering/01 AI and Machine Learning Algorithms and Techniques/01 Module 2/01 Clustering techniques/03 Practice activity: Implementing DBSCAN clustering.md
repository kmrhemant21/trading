# Practice activity: Implementing DBSCAN clustering

## Introduction
In this activity, you will implement density-based spatial clustering of applications with noise (DBSCAN) using Python. DBSCAN is an unsupervised learning algorithm that forms clusters based on the density of data points, which makes it especially useful for detecting outliers and finding clusters of arbitrary shapes. By the end of this activity, you will have experience applying DBSCAN to a dataset, tuning its parameters, and identifying noise points (outliers).

By the end of this activity, you'll be able to:
- Implement DBSCAN clustering using Python to detect clusters of arbitrary shapes and identify noise points in a dataset.
- Tune DBSCAN parameters to optimize clustering results and understand the impact of these parameters on cluster formation and outlier detection.
- Visualize and interpret DBSCAN clusters to gain insights into the data, including recognizing how changes in parameters affect cluster boundaries and the identification of outliers.

## Step-by-step guide:
### Step 1: Setting up the environment
Before you begin, ensure that you have the necessary libraries installed. You will need pandas, Scikit-learn, and Matplotlib.

Install the required libraries using the following command:

```
pip install pandas scikit-learn matplotlib
```

### Step 2: Importing required libraries
Start by importing the libraries needed for data manipulation, clustering, and visualization.

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
```

pandas for handling and manipulating the dataset

DBSCAN from Scikit-learn to perform DBSCAN clustering

StandardScaler for normalizing the dataset before clustering

### Step 3: Loading the dataset
You will use a sample dataset. For this example, we will use a fictional dataset that contains customer information based on their AnnualIncome and SpendingScore.

Load the dataset as follows:

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

- **AnnualIncome**: Annual income of the customer (in thousands).
- **SpendingScore**: A score representing the customer's spending behavior on a scale of 1 to 100.

### Step 4: Preprocessing the data
Clustering algorithms such as DBSCAN are sensitive to the scale of the features, so it's essential to normalize the data. We'll use StandardScaler to scale the features.

```python
# Normalize the dataset using StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Convert the scaled data back into a DataFrame for easier handling
df_scaled = pd.DataFrame(df_scaled, columns=['AnnualIncome', 'SpendingScore'])
print(df_scaled.head())
```

Now that the data is scaled, the AnnualIncome and SpendingScore features have a mean of 0 and a standard deviation of 1, ensuring that all features contribute equally to the clustering process.

### Step 5: Implementing DBSCAN clustering
Now, let's apply the DBSCAN clustering algorithm. DBSCAN requires two important parameters:

- **eps**: The maximum distance between two points to be considered neighbors.
- **min_samples**: The minimum number of points required to form a dense region (cluster).

```python
from sklearn.cluster import DBSCAN

# Initialize DBSCAN with the parameters
dbscan = DBSCAN(eps=0.5, min_samples=3)

# Fit the model to the scaled data
dbscan.fit(df_scaled)

# Assign cluster labels to the data points
df['Cluster'] = dbscan.labels_

# Display the first few rows with cluster labels
print(df.head())
```

eps is set to 0.5, which is the radius within which points are considered neighbors.

min_samples is set to 3, meaning that a core point must have at least three neighbors within the eps radius to form a cluster.

The Cluster column shows the cluster label assigned to each data point. Points labeled –1 represent noise (outliers) that do not belong to any cluster.

### Step 6: Visualizing the clusters
After running DBSCAN, you can visualize the results by plotting the clusters. Data points will be colored based on their assigned cluster, and outliers (noise points) will be displayed as a separate color.

```python
# Plot the clusters
plt.scatter(df['AnnualIncome'], df['SpendingScore'], c=df['Cluster'], cmap='rainbow')
plt.title('DBSCAN Clustering of Customers')
plt.xlabel('Annual Income (in thousands)')
plt.ylabel('Spending Score (1-100)')
plt.show()
```

In this scatterplot, different colors represent different clusters.

Points labeled as noise (outliers) are usually colored separately (often black or gray) and do not belong to any cluster.

### Step 7: Tuning the DBSCAN parameters
DBSCAN's performance depends heavily on the values of eps and min_samples. If eps is too small, many points may be classified as outliers, whereas if it's too large, clusters may merge. Similarly, adjusting min_samples affects the density of the clusters.

Try experimenting with different values of eps and min_samples to see how they impact the clustering results. For example, to reduce the number of outliers, you can try increasing eps:

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

Continue adjusting the parameters, and observe how the clusters and outliers change.

## Conclusion
In this activity, you implemented the DBSCAN clustering algorithm successfully to group customers based on their AnnualIncome and SpendingScore. You also:

- Preprocessed the data using StandardScaler to normalize the features.
- Applied DBSCAN clustering with adjustable parameters such as eps and min_samples.
- Visualized the clusters and identified noise (outliers) using a scatterplot.

DBSCAN is particularly effective when you want to find clusters of arbitrary shapes or need to detect outliers in a dataset. It does not require the number of clusters to be specified in advance, making it versatile for various real-world applications.

You've gained practical experience in DBSCAN clustering, including data preprocessing, cluster visualization, and parameter tuning. Feel free to experiment with different datasets and values for eps and min_samples to gain a deeper understanding of DBSCAN clustering.

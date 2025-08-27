# Walkthrough: Interpreting clustering and dimensionality reduction outcomes (Optional)

## Introduction

This walkthrough will review the proper implementation of k-means clustering and dimensionality reduction techniques—principal component analysis (PCA) and t-distributed stochastic neighbor embedding (t-SNE)—to a dataset. In this process, you'll learn how to group similar data points into clusters and simplify complex datasets for visualization and interpretation. These techniques were applied to customer data, including features such as annual income, spending score, and age, to uncover meaningful patterns and groupings.

By the end of this walkthrough, you'll be able to:

- Apply the k-means clustering algorithm to a dataset to group similar data points, and understand how to assign and interpret cluster labels based on feature similarity.
- Utilize dimensionality reduction techniques to reduce a dataset's dimensions, making it easier to visualize and interpret complex data.
- Understand the differences between PCA and t-SNE.
- Visualize and analyze clustering results using scatter plots to interpret the effectiveness of clustering and dimensionality reduction techniques in uncovering patterns in the dataset.

## Step-by-step guide:

### Step 1: Loading and preprocessing the dataset

The dataset included features such as annual income, spending score, and age. You began by loading and preprocessing the dataset using StandardScaler to normalize the features, ensuring all features had the same scale:

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Create a sample dataset with customer annual income, spending score, and age
data = {'AnnualIncome': [15, 16, 17, 18, 19, 20, 22, 25, 30, 35],
        'SpendingScore': [39, 81, 6, 77, 40, 76, 94, 5, 82, 56],
        'Age': [20, 22, 25, 24, 35, 40, 30, 21, 50, 31]}

df = pd.DataFrame(data)

# Normalize the dataset
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Convert the scaled data back into a DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=['AnnualIncome', 'SpendingScore', 'Age'])
print(df_scaled.head())
```

StandardScaler ensures that each feature contributes equally to the clustering and dimensionality reduction processes by centering and scaling the data.

### Step 2: Applying k-means clustering

Next, you applied k-means clustering with k = 3 clusters to group the customers based on their annual income, spending score, and age.

```python
from sklearn.cluster import KMeans

# Apply K-Means with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
df_scaled['KMeans_Cluster'] = kmeans.fit_predict(df_scaled)

# Display the cluster assignments
print(df_scaled.head())
```

KMeans_Cluster contains the cluster assignments for each data point. Each customer is assigned to one of the three clusters based on the similarity of their features.

### Step 3: Visualizing k-means clusters

To visualize the clusters, you created a scatter plot with annual income and spending score, using color to represent the cluster assignments:

```python
import matplotlib.pyplot as plt

# Visualize the K-Means clusters
plt.scatter(df_scaled['AnnualIncome'], df_scaled['SpendingScore'], c=df_scaled['KMeans_Cluster'], cmap='viridis')
plt.title('K-Means Clustering of Customers')
plt.xlabel('Annual Income (in thousands)')
plt.ylabel('Spending Score (1-100)')
plt.show()
```

The scatter plot shows how k-means clustered the customers based on their income and spending behavior. In this case, k-means formed three distinct clusters, grouping similar customers together.

### Step 4: Applying dimensionality reduction (PCA or t-SNE)

To reduce the dimensionality of the dataset for visualization, you applied either PCA or t-SNE to reduce the three features to two components. You could then visualize the data in two dimensions.

For PCA:

```python
from sklearn.decomposition import PCA

# Apply PCA to reduce dimensions from 3 to 2
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Convert the PCA result back to a DataFrame
df_pca = pd.DataFrame(df_pca, columns=['PCA1', 'PCA2'])
print(df_pca.head())
```

For t-SNE:

```python
from sklearn.manifold import TSNE

# Apply t-SNE to reduce dimensions to 2
tsne = TSNE(n_components=2, random_state=42)
df_tsne = tsne.fit_transform(df_scaled)

# Convert the t-SNE result back to a DataFrame
df_tsne = pd.DataFrame(df_tsne, columns=['t-SNE1', 't-SNE2'])
print(df_tsne.head())
```

PCA captures the global variance and reduces dimensionality linearly.

t-SNE preserves local structures in the data and is more effective for visualizing clusters in nonlinear datasets.

### Step 5: Visualizing dimensionality reduction results

To visualize the clusters in two dimensions after dimensionality reduction, you created scatter plots showing the clusters identified by k-means, colored according to the cluster assignments.

For PCA:

```python
# Visualize the PCA components with K-Means clusters
plt.scatter(df_pca['PCA1'], df_pca['PCA2'], c=df_scaled['KMeans_Cluster'], cmap='viridis')
plt.title('PCA - Dimensionality Reduction with K-Means Clusters')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()
```

For t-SNE:

```python
# Visualize the t-SNE components with K-Means clusters
plt.scatter(df_tsne['t-SNE1'], df_tsne['t-SNE2'], c=df_scaled['KMeans_Cluster'], cmap='viridis')
plt.title('t-SNE - Dimensionality Reduction with K-Means Clusters')
plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')
plt.show()
```

The scatter plot (PCA or t-SNE) allows you to interpret how the clusters are distributed after dimensionality reduction. t-SNE preserves local relationships so is likely to show tighter clusters compared to PCA.

### Step 6: Interpreting the results

After performing the clustering and dimensionality reduction, you interpreted the results based on the visualizations and the cluster assignments. Here are some key points to consider:

**K-means clustering:**

- How well did k-means group similar customers? Did the clusters make sense based on the features?
- Were the clusters distinct or overlapping? If there was significant overlap, increasing or decreasing k—the number of clusters—could improve the result.

**Dimensionality reduction:**

- PCA: How much variance was captured by the two principal components? If a large percentage of variance was retained—e.g., more than 80 percent—PCA provided a good summary of the dataset.
- t-SNE: Did t-SNE reveal any hidden patterns or clusters that were not obvious in the original dataset? t-SNE is often better at visualizing clusters when the dataset has nonlinear relationships.

**Conclusion:**

- PCA simplifies the dataset while retaining global variance, making it useful for visualizing data with linear relationships.
- t-SNE excels at preserving local structures and is particularly effective for visualizing nonlinear clusters.
- Both techniques worked well together with k-means to group the customers and reveal meaningful patterns in the dataset.

## Conclusion

By completing this activity, you successfully:

- Preprocessed a dataset using StandardScaler to normalize the features.
- Applied k-means clustering to group customers based on their behavior.
- Reduced the dimensionality of the dataset using PCA or t-SNE.
- Visualized and interpreted the clustering and dimensionality reduction outcomes.

Together, clustering and dimensionality reduction techniques are powerful tools for analyzing and visualizing high-dimensional data. They help reveal hidden patterns, simplify complex datasets, and provide insights into the underlying structure of the data. K-means clustering, combined with PCA or t-SNE, offers a comprehensive approach to understanding and interpreting data effectively.

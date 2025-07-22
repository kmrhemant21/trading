Let's understand **Principal Component Analysis (PCA)** thoroughly, step-by-step, from scratch. I'll introduce and clearly explain every new concept along the way.

---

## 🟢 **What is PCA?**

PCA (**Principal Component Analysis**) is a **dimensionality reduction** technique used to simplify datasets by transforming them into fewer variables (principal components) while retaining as much useful information as possible.

It helps in:

* **Visualizing high-dimensional data**
* **Reducing noise**
* **Improving computational efficiency**

---

## 🟢 **Intuition behind PCA**

Imagine you have data in 2D, spread out in an ellipse. The data has a clear direction of greatest variance (the longest axis of the ellipse).

PCA finds these directions of greatest variance and transforms your data to align along them. The direction with highest variance becomes the **first principal component (PC1)**, the second-highest becomes **PC2**, and so on.

---

## 🟢 **Step-by-step PCA explanation**

Suppose we have an original dataset:

| Feature 1 (X₁) | Feature 2 (X₂) |
| -------------- | -------------- |
| 2              | 3              |
| 3              | 4              |
| 4              | 5              |
| 5              | 7              |

Let's perform PCA step by step:

---

## 🔵 **Step 1: Standardize the data**

This ensures that each feature contributes equally.

Formula for standardization:

$$
X_{\text{standardized}} = \frac{X - \mu}{\sigma}
$$

* **μ** (mu) = Mean of the feature
* **σ** (sigma) = Standard deviation of the feature

For simplicity, let’s just center the data (subtract mean) without scaling, as an example:

| X₁ | X₂ | X₁ - mean(X₁) | X₂ - mean(X₂) |
| -- | -- | ------------- | ------------- |
| 2  | 3  | -1.5          | -2.25         |
| 3  | 4  | -0.5          | -1.25         |
| 4  | 5  | 0.5           | -0.25         |
| 5  | 7  | 1.5           | 1.75          |

Mean(X₁) = 3.5; Mean(X₂) = 5.25

---

## 🔵 **Step 2: Calculate the covariance matrix**

**Covariance** measures how two variables vary together:

* **Positive covariance:** Variables increase/decrease together.
* **Negative covariance:** Variables move oppositely.
* **Zero covariance:** No predictable relationship.

**Covariance Matrix formula** for 2 features (X₁, X₂):

$$
Cov = 
\begin{bmatrix}
Cov(X_1, X_1) & Cov(X_1, X_2) \\
Cov(X_2, X_1) & Cov(X_2, X_2)
\end{bmatrix}
$$

**Covariance** between two variables, X and Y, is calculated as:

$$
Cov(X,Y) = \frac{\sum{(X_i - \mu_X)(Y_i - \mu_Y)}}{n - 1}
$$

Using the example data above:

$$
Cov = 
\begin{bmatrix}
\frac{-1.5^2 + (-0.5)^2 + 0.5^2 + 1.5^2}{3} & \frac{(-1.5)(-2.25) + (-0.5)(-1.25) + (0.5)(-0.25) + (1.5)(1.75)}{3} \\[6pt]
(same) & \frac{-2.25^2 + (-1.25)^2 + (-0.25)^2 + 1.75^2}{3}
\end{bmatrix}
$$

Compute this to get the covariance matrix.

---

## 🔵 **Step 3: Compute eigenvectors and eigenvalues**

### New concepts: Eigenvectors and Eigenvalues 🆕

**Eigenvectors** point to the directions of maximum variance (principal directions).

**Eigenvalues** represent the magnitude of variance along the corresponding eigenvectors.

They satisfy this equation:

$$
Cov \times v = \lambda \times v
$$

* $v$ = eigenvector
* $\lambda$ = eigenvalue

This is solved using matrix algebra (linear algebra):

* Compute determinant: $det(Cov - \lambda I) = 0$ to find eigenvalues $\lambda$.
* Use eigenvalues to solve for eigenvectors.

> Typically, this step is done by software libraries (e.g., NumPy).

---

## 🔵 **Step 4: Choose Principal Components**

Sort eigenvectors based on eigenvalues from largest to smallest. The largest eigenvalue corresponds to the most important principal component.

If we have 2 original features:

* **PC1**: Eigenvector with largest eigenvalue.
* **PC2**: Second-largest eigenvalue eigenvector.

If dimensionality reduction is desired, pick only top principal components.

---

## 🔵 **Step 5: Project original data**

Use eigenvectors to transform data onto new PCA axes:

$$
\text{Transformed Data} = X_{\text{centered}} \times \text{Eigenvectors}
$$

Resulting in data represented by principal components:

| PC1     | PC2     |
| ------- | ------- |
| value₁₁ | value₁₂ |
| value₂₁ | value₂₂ |
| ...     | ...     |

---

## 🟢 **PCA - Visual recap**

**Original Data (Features)** ➡️ **Standardization** ➡️ **Covariance Matrix** ➡️ **Eigenvectors & Eigenvalues** ➡️ **Pick Top Components** ➡️ **Transform Data** ➡️ **Reduced Dataset**

---

## 🟢 **Practical Example (Quick NumPy implementation)**

```python
import numpy as np

# Original data
X = np.array([[2, 3],
              [3, 4],
              [4, 5],
              [5, 7]])

# Step 1: Center the data
X_centered = X - np.mean(X, axis=0)

# Step 2: Covariance matrix
cov_mat = np.cov(X_centered, rowvar=False)

# Step 3: Eigenvectors & eigenvalues
eigen_values, eigen_vectors = np.linalg.eig(cov_mat)

# Step 4: Sort eigenvectors by eigenvalues
sorted_indices = np.argsort(eigen_values)[::-1]
eigen_vectors_sorted = eigen_vectors[:, sorted_indices]

# Step 5: Transform data
X_pca = X_centered @ eigen_vectors_sorted

print("Transformed Data (PCs):")
print(X_pca)
```

---

## 🟢 **Key PCA Terms (summary)**

* **Variance:** Measure of spread or variability.
* **Covariance:** Indicates how two variables vary together.
* **Eigenvectors:** Directions of principal components.
* **Eigenvalues:** Amount of variance explained by principal components.
* **Dimensionality Reduction:** Reducing number of variables.

---

## 🟢 **Applications of PCA**

* Reducing complexity of large datasets
* Visualization (2D or 3D plots of complex datasets)
* Preprocessing before ML algorithms
* Noise reduction in data

---

🎯 **Summary of PCA from Scratch:**

PCA finds directions (**principal components**) where data varies most, reduces the number of features, simplifies analysis, and maintains important information.

You learned about:

* Standardization and covariance
* Eigenvectors and eigenvalues
* Projecting data into a new dimension
* Practical PCA implementation (NumPy)

Now you’ve got a comprehensive understanding of PCA!

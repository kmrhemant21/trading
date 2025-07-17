# Manage Missing Values, Outliers, Normalize, and Transform Data

## Introduction

In machine learning, the quality of your data directly impacts the effectiveness of your models. This makes data preprocessing a critical step in any AI/ML pipeline. In this lesson, we’ll cover four key aspects of data preprocessing: handling missing values, managing outliers, normalization, and transformation. This lesson will help you understand and apply these techniques to ensure that your data is clean, consistent, and ready for analysis.

By the end of this reading, you will be able to: 

- Handle missing values using various strategies.
- Identify and manage outliers effectively.
- Normalize and standardize data for improved model performance.
- Transform data to meet the assumptions of statistical models and enhance machine learning algorithms.

---

## Handle Missing Values

Missing values are a common issue in datasets and can arise for various reasons, such as data entry errors or unavailability of certain information. If not addressed, missing values can lead to biased results or reduce the accuracy of your model.

### Strategies for handling missing values:

1. **Remove missing data**  
    **Description**: If a small number of rows or columns have missing values, you might consider removing them from the dataset.  
    **When to use**: This approach is suitable when the missing data is minimal and its removal won’t significantly impact the dataset.  

    **Code example**:
    ```python
    # Drop rows with missing values
    df_cleaned = df.dropna()

    # Drop columns with missing values
    df_cleaned = df.dropna(axis=1)
    ```

2. **Impute missing data**  
    **Description**: Imputation involves filling in missing values with a substitute value, such as the mean, median, or mode of the column.  
    **When to use**: This is useful when missing data is more prevalent, but you don’t want to lose information by removing rows or columns.  

    **Code example**:
    ```python
    # Fill missing values with the mean of the column
    df['column_name'].fillna(df['column_name'].mean(), inplace=True)

    # Fill missing values with the median of the column
    df['column_name'].fillna(df['column_name'].median(), inplace=True)
    ```

3. **Forward or backward fill**  
    **Description**: Forward fill propagates the last valid observation forward, while backward fill does the opposite.  
    **When to use**: This is particularly useful in time series data where trends or sequences are important.  

    **Code example**:
    ```python
    # Forward fill
    df.fillna(method='ffill', inplace=True)

    # Backward fill
    df.fillna(method='bfill', inplace=True)
    ```

---

## Manage Outliers

Outliers are data points that differ significantly from other observations. They can distort statistical analyses and negatively impact the performance of machine learning models.

Here’s a deeper dive into both concepts, with formulas, worked examples, and when to use each.

---

## 1. Z-Score (Standard Score)

### 1.1 Definition

A **z-score** tells you how many standard deviations a value $x$ lies from the mean.  Formally, for a population:

$$
z = \frac{x - \mu}{\sigma},
$$

and for a sample,

$$
z = \frac{x - \bar{x}}{s},
$$

where

* $\mu$ (or $\bar{x}$) is the (population or sample) mean,
* $\sigma$ (or $s$) is the (population or sample) standard deviation.

### 1.2 Interpretation

* **$z=0$** means $x$ equals the mean.
* **$z>0$** means $x$ is above the mean; **$z<0$** means below.
* $|z|$ quantifies “how unusual” $x$ is.  Common rule-of-thumb: $|z|>3$ flags a potential outlier.

### 1.3 Uses

* **Standardization:**  Convert disparate scales to a common Normal(0,1) framework so you can compare values from different distributions.
* **Probability lookup:**  Under Normality, $P(X \le x)=P(Z\le z)$ via standard-normal tables (or software).
* **Z-tests:**  Many statistical tests (e.g. one-sample z-test) use z-scores to form test statistics.

### 1.4 Worked Example

Data: test scores $[65,\,70,\,75,\,80,\,85]$.

1. Mean $\bar{x}=75$.
2. Sample SD

   $$
     s = \sqrt{\frac{\sum (x_i - 75)^2}{5-1}}
       = \sqrt{\frac{(10^2 +5^2+0^2+5^2+10^2)}{4}}
       = \sqrt{\frac{250}{4}} \approx 7.91.
   $$
3. For $x=85$:

   $$
     z = \frac{85 - 75}{7.91} \approx +1.26,
   $$

   so 85 is about 1.26 SDs above the mean.

---

## 2. IQR (Interquartile Range)

### 2.1 Definition

The **interquartile range** is

$$
\mathrm{IQR} = Q_3 - Q_1,
$$

where

* $Q_1$ = the 25th percentile (the “lower quartile”),
* $Q_3$ = the 75th percentile (the “upper quartile”).

It measures the spread of the **middle 50%** of your data.

### 2.2 Computing Quartiles

1. **Sort** the data.
2. Find the **median** (the 50th percentile, $Q_2$).
3. Split the ordered data at the median into “lower half” and “upper half.”
4. The median of the lower half is $Q_1$; of the upper half is $Q_3$.

### 2.3 Outlier “Fences”

Values outside

$$
\bigl[\,Q_1 - 1.5\,\mathrm{IQR},\;Q_3 + 1.5\,\mathrm{IQR}\bigr]
$$

are often flagged as outliers.

### 2.4 Worked Example

Data: $[3,\,7,\,8,\,12,\,13,\,14,\,18,\,21,\,23]$

1. Sorted already.
2. Median ($Q_2$) = 12.
3. Lower half $[3,7,8,\,\color{blue}{12}]$ → ignore the 12 itself for odd sets, so lower half = \[3,7,8] → $Q_1=7$.
4. Upper half $[\color{blue}{12},13,14,18,21,23]$ → upper half = \[13,14,18,21,23] → $Q_3=18$.
5. IQR = $18 - 7 = 11$.
6. Fences = $[7 - 1.5×11,\;18 + 1.5×11] = [-9.5,\;34.5]$.
   All points lie within, so no outliers by this rule.

---

## 3. Z-Score vs. IQR

| Feature           | Z-Score                                   | IQR                                              |                   |                                                         |
| ----------------- | ----------------------------------------- | ------------------------------------------------ | ----------------- | ------------------------------------------------------- |
| **Measures**      | Distance from mean in SD units            | Spread of middle 50% of data                     |                   |                                                         |
| **Sensitive to…** | Every data point (mean & SD use all data) | Only middle half (robust to extremes)            |                   |                                                         |
| **Use cases**     | Normalization; tails & probabilities      | Describing variability; robust outlier detection |                   |                                                         |
| **Outlier rule**  | (                                         | z                                                | >3) (approximate) | Outside $Q_1-1.5\,\mathrm{IQR},\,Q_3+1.5\,\mathrm{IQR}$ |

---

### When to Choose Which

* **Use z-scores** when you assume (or know) data are approximately Normal, want to compute probabilities or compare values across different units.
* **Use IQR** when you need a robust measure of spread (resistant to extreme values) or a nonparametric outlier‐detection rule.

Feel free to ask if you’d like more on computing these in software (e.g. Excel, R, Python), or on their use in specific analyses!


### Strategies for managing outliers:

1. **Identify outliers**  
    **Description**: The first step is to identify outliers, which can be done using statistical methods such as Z-score or the Interquartile Range (IQR).  

    **Code example**:
    ```python
    # Using Z-score to identify outliers
    from scipy import stats
    import numpy as np

    z_scores = np.abs(stats.zscore(df['column_name']))
    outliers = df[z_scores > 3]

    # Using IQR to identify outliers
    Q1 = df['column_name'].quantile(0.25)
    Q3 = df['column_name'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df['column_name'] < (Q1 - 1.5 * IQR)) | (df['column_name'] > (Q3 + 1.5 * IQR))]
    ```

2. **Handle outliers**  
    a. **Remove outliers**  
    **Description**: Outliers can be removed from the dataset if they are believed to be errors or not representative of the population.  

    **Code example**:
    ```python
    # Remove outliers identified by Z-score
    df_cleaned = df[(z_scores <= 3)]

    # Remove outliers identified by IQR
    df_cleaned = df[~((df['column_name'] < (Q1 - 1.5 * IQR)) | (df['column_name'] > (Q3 + 1.5 * IQR)))]
    ```

    b. **Cap or transform outliers**  
    **Description**: Instead of removing outliers, you might cap them to a certain threshold or transform them using logarithmic or other functions to reduce their impact.  

    **Code example**:
    ```python
    # Cap outliers to a threshold
    df['column_name'] = np.where(df['column_name'] > upper_threshold, upper_threshold, df['column_name'])

    # Log transform to reduce the impact of outliers
    df['column_name_log'] = np.log(df['column_name'] + 1)
    ```

---

## Normalization

Normalization (or scaling) is the process of adjusting the values of numeric columns in a dataset to a common scale, typically between zero and one. This is especially important for machine learning algorithms that rely on the magnitude of features such as gradient descent-based algorithms.

### Methods of normalization:

1. **Min-Max scaling**  
    **Description**: Scales all numeric values in a column to a range between zero and one.  

    **Code example**:
    ```python
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    df['scaled_column'] = scaler.fit_transform(df[['column_name']])
    ```

2. **Z-score standardization**  
    **Description**: Scales the data so that it has a mean of zero and a standard deviation of one. This method is useful when you want to compare features with different units or scales.  

    **Code example**:
    ```python
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    df['standardized_column'] = scaler.fit_transform(df[['column_name']])
    ```

---

## Data Transformation

Data transformation involves converting data from one format or structure to another. This is often necessary to meet the assumptions of statistical models or to improve the performance of machine learning algorithms.

### Common data transformations:

1. **Logarithmic transformation**  
    **Description**: Log transformation is used to stabilize variance, by making the data appear more like normal distribution and reducing the impact of outliers.  

    **Code example**:
    ```python
    df['log_column'] = np.log(df['column_name'] + 1)  # Adding 1 to avoid log(0)
    ```

2. **Box-Cox transformation**  
    **Description**: This transformation is used to stabilize variance and make the data more normally distributed.  

    **Code example**:
    ```python
    from scipy import stats

    df['boxcox_column'], _ = stats.boxcox(df['column_name'] + 1)  # Adding 1 to avoid log(0)
    ```

3. **Binning**  
    **Description**: Binning, or discretization, involves converting continuous variables into discrete categories.  

    **Code example**:
    ```python
    # Create bins for a continuous variable
    df['binned_column'] = pd.cut(df['column_name'], bins=[0, 10, 20, 30], labels=['Low', 'Medium', 'High'])
    ```

4. **Encoding categorical variables**  
    **Description**: Transforming categorical data into numerical format, which is necessary for many machine learning algorithms.  

    **Code example**:
    ```python
    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=['category_column'])
    ```

---

## Conclusion

Handling missing values, managing outliers, normalization, and transformation are essential steps in preparing your data for machine learning. Properly applying these techniques ensures that your dataset is clean, consistent, and in the right format for analysis, leading to more accurate and reliable models. 

As you work with different datasets, practice these techniques to become proficient in data preprocessing, which is a critical skill in the data science workflow.

By mastering these preprocessing techniques, you’ll be better equipped to tackle a wide range of data challenges, ensuring that your models are built on a solid foundation of high-quality data.

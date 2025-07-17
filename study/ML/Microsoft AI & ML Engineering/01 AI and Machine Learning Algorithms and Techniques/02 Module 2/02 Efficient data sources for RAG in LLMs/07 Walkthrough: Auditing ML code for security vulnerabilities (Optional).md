# Walkthrough: Auditing ML Code for Security Vulnerabilities (Optional)

## Introduction

You were asked to review a block of intentionally flawed ML code and audit it for potential security vulnerabilities. Doing this involved auditing the code for potential security vulnerabilities that could pose risks to data integrity, confidentiality, and the overall security of the ML system.

In this reading, we will walk through the lab assignment step by step, discussing the vulnerabilities present in the code, their potential impacts, and the proper solutions to secure the ML system.

By the end of this walkthrough, you will be able to: 

- Identify common security risks in ML code and mitigate the risks effectively. 

---

## Review the Provided ML Code Block

### Original Code Block:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset (Flaw: No data validation or sanitization)
data = pd.read_csv('user_data.csv')

# Split the dataset into features and target (Flaw: No input validation)
X = data.iloc[:, :-1]
```

**Explanation:** This code block contains several security vulnerabilities that could compromise the integrity and confidentiality of the ML system. We’ll go through each identified flaw and discuss how to address them.

---

## Identify Security Vulnerabilities

### Vulnerability 1: Lack of Data Validation and Sanitization

- **Issue:** The dataset is loaded directly from a file without any validation or sanitization. This opens the door to potential injection attacks or the use of corrupted data.
- **Impact:** Malicious data could compromise the model’s accuracy or introduce security risks by executing unwanted code.
- **Solution:** Implement data validation and sanitization routines to ensure that the input data is clean and free of malicious content.

```python
# Validate and sanitize input data
def validate_data(df):
    if df.isnull().values.any():
        raise ValueError("Dataset contains null values. Please clean the data before processing.")
    # Additional validation checks can be added here
    return df

# Load and validate dataset
data = validate_data(pd.read_csv('user_data.csv'))
```

---

### Vulnerability 2: No Input Validation

- **Issue:** The code assumes that the input data will always be in the correct format. However, if the data is corrupted or contains unexpected types, this could lead to errors or vulnerabilities.
- **Impact:** Invalid input data could cause the model training process to fail or produce inaccurate results.
- **Solution:** Include input validation checks to ensure that the data meets expected formats and values before processing.

```python
# Split the dataset into features and target with validation
X = validate_data(data.iloc[:, :-1])
y = validate_data(data.iloc[:, -1])
```

---

### Vulnerability 3: Fixed Random State

- **Issue:** The use of a fixed random state (e.g., `random_state=42`) can make the model training process predictable, potentially exposing it to attacks that exploit this predictability.
- **Impact:** Attackers can use predictable random states to infer model behaviors or replicate the training process maliciously.
- **Solution:** Use a securely generated random state, such as one derived from a secure source such as `os.urandom()`.

```python
import os

# Split the data into training and testing sets with a securely managed random state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=os.urandom(16))
```

---

### Vulnerability 4: Unencrypted Model Saving

- **Issue:** The trained model is saved to disk without encryption, which means that anyone with access to the storage location can access and potentially tamper with the model.
- **Impact:** Unencrypted models are vulnerable to unauthorized access, theft, and tampering.
- **Solution:** Encrypt the model before saving it to disk to protect its confidentiality and integrity.

```python
import cryptography.fernet

# Encrypt model before saving
key = cryptography.fernet.Fernet.generate_key()
cipher = cryptography.fernet.Fernet(key)

# Save the encrypted model to disk
filename = 'finalized_model.sav'
encrypted_model = cipher.encrypt(pickle.dumps(model))
with open(filename, 'wb') as f:
    f.write(encrypted_model)
```

---

### Vulnerability 5: No Integrity Checks on Loaded Model

- **Issue:** The model is loaded from disk without any integrity checks, meaning that it could be tampered with or corrupted without detection.
- **Impact:** A compromised model could produce incorrect predictions or expose vulnerabilities in downstream applications.
- **Solution:** Implement integrity checks to verify that the model has not been altered between saving and loading.

```python
import hashlib

# Load the encrypted model from disk and verify its integrity
with open(filename, 'rb') as f:
    encrypted_model = f.read()
    decrypted_model = cipher.decrypt(encrypted_model)

loaded_model = pickle.loads(decrypted_model)

# Compute hash of the loaded model
model_hash = hashlib.sha256(decrypted_model).hexdigest()
```

---

## Proper Solution Overview

By addressing these vulnerabilities, the improved ML code ensures that the data is validated and sanitized, the model is trained and stored securely, and integrity checks are performed during model loading. Here’s the finalized code with all the security improvements applied:

```python
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import hashlib
import cryptography.fernet

# Validate and sanitize input data
def validate_data(df):
    if df.isnull().values.any():
        raise ValueError("Dataset contains null values. Please clean the data before processing.")
    # Additional validation checks can be added here
    return df

# Load and validate dataset
data = validate_data(pd.read_csv('user_data.csv'))

# Split the dataset into features and target with validation
X = validate_data(data.iloc[:, :-1])
y = validate_data(data.iloc[:, -1])

# Split the data into training and testing sets with a securely managed random state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=os.urandom(16))

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Encrypt model before saving
key = cryptography.fernet.Fernet.generate_key()
cipher = cryptography.fernet.Fernet(key)
filename = 'finalized_model.sav'
encrypted_model = cipher.encrypt(pickle.dumps(model))
with open(filename, 'wb') as f:
    f.write(encrypted_model)

# Load the encrypted model from disk and verify its integrity
with open(filename, 'rb') as f:
    encrypted_model = f.read()
    decrypted_model = cipher.decrypt(encrypted_model)
loaded_model = pickle.loads(decrypted_model)
model_hash = hashlib.sha256(decrypted_model).hexdigest()
```

---

## Conclusion

This activity highlights the importance of implementing security best practices in every stage of the ML life cycle, from data collection and preprocessing to model training and deployment. By following the steps outlined in this walkthrough, you should now have a better understanding of how to identify and mitigate security vulnerabilities in ML code.

Securing ML systems is not just about preventing unauthorized access; it’s also about ensuring the integrity and reliability of the models you build. By incorporating these practices into your development workflow, you can protect your models from a wide range of potential threats and ensure that they remain trustworthy and effective in real-world applications.

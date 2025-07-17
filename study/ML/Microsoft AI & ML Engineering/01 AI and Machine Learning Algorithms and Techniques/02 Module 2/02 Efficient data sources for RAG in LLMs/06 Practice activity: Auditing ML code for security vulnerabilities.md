# Practice Activity: Auditing ML Code for Security Vulnerabilities

## Introduction

In this activity, you will be tasked with reviewing a block of intentionally flawed ML code. Your objective is to audit the code for potential security vulnerabilities that could pose risks to data integrity, confidentiality, and the overall security of the ML system. This activity will take you approximately 60 minutes to complete.

By the end of this activity, you will be able to: 

- Identify and mitigate security risks in AI/ML development.
- Explain the implications of security vulnerabilities in ML code.
- Propose solutions to mitigate the identified risks.

---

## Instructions

### Step-by-Step Guide

#### Step 1: Review the Provided ML Code Block

You will be given a block of ML code that contains several intentional security flaws. Your first task is to carefully review the code and identify any potential vulnerabilities. These may include issues related to data handling, access controls, model deployment, and other security aspects.

**Code block example (with intentional flaws):**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset (Flaw: No data validation or sanitization)
data = pd.read_csv('user_data.csv')

# Split the dataset into features and target (Flaw: No input validation)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training and testing sets (Flaw: Fixed random state)
```

---

#### Step 2: Identify Security Vulnerabilities

As you review the code, look for potential security vulnerabilities such as:

- **Data validation and sanitization:** Is the input data validated or sanitized to prevent malicious input?
- **Input validation:** Are there any checks on the input data to ensure it meets expected formats or values?
- **Random state and seed management:** Is the random state used securely to prevent predictability in model training?
- **Model security:** Are there security measures in place to protect the model from tampering or unauthorized access?
- **Encryption:** Is sensitive data, such as the trained model, encrypted when stored or transmitted?
- **Integrity checks:** Are there mechanisms to verify the integrity of the model when loading it from storage?

---

#### Step 3: Document Your Findings

For each vulnerability you identify, document the following:

- **Vulnerability description:** Clearly describe the issue you’ve identified.
- **Potential impact:** Explain the potential security risks if this vulnerability were exploited.
- **Mitigation strategy:** Propose a solution or best practice to mitigate the identified vulnerability.

**Example documentation:**

- **Vulnerability:** Lack of data validation and sanitization when loading the dataset.
- **Impact:** Malicious users could inject harmful code or corrupted data, leading to data breaches or compromised model integrity.
- **Mitigation:** Implement input validation and sanitization routines before processing the data. For example, ensure that the CSV file only contains expected data types and values.

---

#### Step 4: Propose Code Improvements

Based on your findings, propose improvements to the provided code block to enhance its security. Rewrite sections of the code where necessary, incorporating best practices such as:

- Adding data validation and sanitization.
- Using secure random state management.
- Encrypting models before saving them to disk.
- Implementing integrity checks when loading models.

**Example code improvement:**

```python
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import hashlib

# Validate and sanitize input data
def validate_data(df):
    # Example validation: Check for null values, correct data types, etc.
    if df.isnull().values.any():
```

---

#### Step 5: Compile Your Findings

Compile your findings, including the identified vulnerabilities, their potential impacts, proposed mitigation strategies, and the improved code. 

---

## Conclusion

This activity is designed to enhance your ability to identify and address security vulnerabilities in ML code. By completing this exercise, you will gain practical experience in securing AI/ML systems, an essential skill in today’s increasingly data-driven world. Remember, the goal is not just to write functional code, but to ensure that it is secure and robust against potential threats.

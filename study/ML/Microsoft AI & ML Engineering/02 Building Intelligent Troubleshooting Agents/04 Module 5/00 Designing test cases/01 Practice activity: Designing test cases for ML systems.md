# Practice activity: Designing test cases for ML systems

## Introduction
In this activity, you will design and implement test cases to ensure that your machine learning (ML) model performs effectively across typical, edge, and error scenarios. The goal is to create a robust suite of test cases that validate the model's behavior in a variety of conditions, ensuring its reliability, accuracy, and resilience.

By the end of this activity, you will:

- Design test cases for typical scenarios based on representative input data.
- Develop test cases for edge scenarios, focusing on extreme, unusual, or unexpected inputs.
- Automate test execution to validate the model after updates or when new data is introduced.

## Step-by-step process to design test cases for ML systems
Create a new Jupyter notebook. Make sure you have the appropriate Python 3.8 Azure ML kernel selected.

The remaining of this reading will guide you through the following steps:

1. Step 1: Set up the environment
2. Step 2: Define test cases for typical scenarios
3. Step 3: Design test cases for edge scenarios
4. Step 4: Create error-handling test cases
5. Step 5: Automate test case execution
6. Step 6: Evaluate the effectiveness of your test cases

### Step 1: Set up the environment
Begin by setting up your Python environment. Make sure you have a testing framework like pytest or unittest installed to support automated testing.

#### Setup commands
```python
!pip install pytest
!Pip install ipytest
```

#### Explanation
These libraries will help you automate test case execution and ensure that the test cases can be run efficiently every time your model is updated.

Let's create a model.

```python
import ipytest
import pytest
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Load the dataset
iris = load_iris()
X, y = iris.data, iris.target

# Set up the model
model = DecisionTreeClassifier()
model.fit(X, y)
```

Here we've used the iris dataset and a decision tree classifier to create a decision tree classifier model.

### Step 2: Define test cases for typical scenarios
Start by defining test cases for typical scenarios. These scenarios represent the normal input data your model will frequently encounter during regular operation.

#### Instructions
1. Identify representative input data from your dataset.
2. Define the expected output for each test case based on your model's behavior.
3. Create a set of test cases that cover the most common use cases.

#### Code example
```python
def test_typical_case():
    input_data = np.array([[4.5, 2.3, 1.3, 0.3]])  # Example input for a flower classification model
    expected_output = 0  # Expected output for typical case (Setosa class index)
    result = model.predict(input_data)[0]
    assert result == expected_output, f"Expected {expected_output}, but got {result}"
```

#### Explanation
This test case checks whether the model correctly classifies a typical input, ensuring it handles common situations well.

### Step 3: Design test cases for edge scenarios
Next, create test cases that challenge the model with edge scenarios. These tests use data that falls outside the normal range and include extreme or unusual values.

#### Instructions
1. Identify edge cases with unusually high or low values.
2. Test how the model responds to these edge cases and ensure robustness.
3. Define the expected output or error handling for each edge case.

#### Code example
```python
def test_edge_case_extreme_values():
    input_data = np.array([[1000, 1000, 1000, 1000]])  # Extreme values for flower classification
    try:
        model.predict(input_data)
    except ValueError:
        assert True  # The model should raise a ValueError for extreme inputs
    else:
        assert False, "Expected ValueError for extreme values, but no error was raised"
```

#### Explanation
Edge case testing ensures that the model doesn't produce incorrect results or crash when encountering extreme or unusual inputs.

### Step 4: Create error-handling test cases
Now create test cases to verify the model's ability to handle invalid inputs. Error-handling tests check whether the model behaves correctly when it receives flawed or incomplete data.

#### Instructions
1. Create test cases with missing, null, or malformed data.
2. Ensure that the model handles these cases appropriately, either by returning an error message or handling the input gracefully.

#### Code example
```python
def test_error_handling_missing_values():
    input_data = np.array([[None, None, None, None]])  # Missing values in input
    try:
        model.predict(input_data)
    except ValueError:
        assert True  # The model should raise a ValueError for missing inputs
    else:
        assert False, "Expected ValueError for missing values, but no error was raised"
```

#### Explanation
This test case checks how the model handles missing data, ensuring it doesn't fail or produce invalid results when inputs are incomplete.

### Step 5: Automate test case execution
After defining your test cases, automate the testing process using pytest or unittest. Automation ensures that tests are run every time the model is updated, preventing errors from slipping through.

#### Instructions
1. Write automated tests for each of your test cases.
2. Use a framework such as pytest to automatically execute all tests.
3. Set up continuous testing to validate the model after updates or new data are introduced.

#### Code example (with pytest)
```python
# Run tests using ipytest
ipytest.run('-v')
```

#### Explanation
Automating test execution ensures that all test cases are checked after each update, helping to catch potential issues before deployment.

### Step 6: Evaluate the effectiveness of your test cases
Finally, evaluate your test cases to ensure they comprehensively cover typical, edge, and error scenarios. Refine them based on the results to improve coverage and accuracy.

#### Instructions
1. Review the test results to ensure the model passes all cases.
2. Check whether the model behaves as expected in edge and error scenarios.
3. Improve the test cases if any failures or unexpected behaviors are identified.

#### Explanation
Evaluating the effectiveness of your test cases helps ensure that they thoroughly test the model's behavior and identify any areas where the model might fail.

## Conclusion
In this activity, you designed and implemented test cases for an ML model, covering typical, edge, and error scenarios. By automating the test execution process, you ensure that your model remains robust, reliable, and well suited for real-world applications, even as it is updated or exposed to new data.

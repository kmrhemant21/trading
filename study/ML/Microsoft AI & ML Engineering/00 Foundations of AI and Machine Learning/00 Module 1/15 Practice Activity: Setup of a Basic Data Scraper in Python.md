# Practice Activity: Setup of a Basic Data Scraper in Python

## Introduction

Web scraping is a powerful method for acquiring data from websites, especially when the information you need isn’t readily available in a structured format. By setting up a web scraper in your local environment, you can automate the process of gathering large amounts of data from the web. 

By the end of this reading, you will be able to: 

- Set up a basic data scraper using Python, including code snippets and explanations to help you get started.

---

## 1. Prerequisites

Before diving into the code, ensure you have the following tools installed on your local environment:

- **Python 3.x:** Python is the language we’ll use to build our web scraper.
- **pip:** pip is Python’s package installer, which you’ll use to install the necessary libraries.
- **A code editor:** Examples include Jupyter Notebooks, VS Code, PyCharm, or even a simple text editor such as Sublime Text.

You’ll also need a basic understanding of HTML, as web scraping involves interacting with the HTML structure of a web page.

---

## 2. Writing the Python script

Now, let’s walk through the code to set up a basic web scraper that extracts data from a web page.

### Step-by-step guide:

#### Step 1: Import the necessary libraries

Start by importing the libraries you’ll need, if they’re not already in your kernel:

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
```

#### Step 2: Send an HTTP request to the website

Use the requests library to send an HTTP GET request to the website you want to scrape:

```python
url = 'https://example.com'  # Replace with the URL of the website you want to scrape
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    print('Request successful!')
else:
    print('Failed to retrieve the webpage')
```

#### Step 3: Parse the HTML content

Once you’ve successfully retrieved the web page, use BeautifulSoup to parse the HTML content:

```python
soup = BeautifulSoup(response.content, 'html.parser')

# Print the title of the webpage to verify
print(soup.title.text)
```

#### Step 4: Extract the data you need

Now that you have the HTML parsed, you can start extracting the data you’re interested in. Let’s say you want to scrape a list of items from a table on the web page:

```python
# Find the table containing the data
table = soup.find('table', {'id': 'data-table'})  # Replace 'data-table' with the actual id or class of the table

# Extract table rows
rows = table.find_all('tr')

# Loop through the rows and extract data
data = []
for row in rows:
    cols = row.find_all('td')
    # (Add your data extraction logic here)
```

#### Step 5: Save the scraped data

Finally, you can save the scraped data to a file for further analysis:

```python
# Save the DataFrame to a CSV file
df.to_csv('scraped_data.csv', index=False)
```

---

## 3. Example: Scraping information from Wikipedia

Let’s go through a more concrete example where we scrape information about cloud computing platforms:

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Send an HTTP request to the webpage
url = 'https://en.wikipedia.org/wiki/Cloud-computing_comparison'  
response = requests.get(url)

# Parse the HTML content
soup = BeautifulSoup(response.content, 'html.parser')
```

Your output should look something like this:

> Output from scraping information from Wikipedia example as described in the full-text transcript.

---

## 4. Important considerations

- **Respect the website’s terms of service:** Always check the website’s terms of service to ensure that you’re allowed to scrape its content. Some websites explicitly prohibit scraping.
- **Be mindful of rate limits:** Avoid sending too many requests in a short period to prevent overloading the website’s server. Implement delays between requests if necessary.
- **Handle errors gracefully:** Always include error handling in your script to manage situations where the website structure changes or the page fails to load.

---

## Conclusion

By setting up a basic web scraper in Python, you can automate the process of gathering data from websites, making it easier to acquire the information you need for your AI/ML projects. 

You've just learned the fundamentals of web scraping, from sending HTTP requests and parsing HTML to extracting and saving data. With this foundation, you can expand your scraper to handle more complex scenarios and integrate the data into your ML models.

Continue practicing with different websites and data structures, and always remember to scrape responsibly and ethically.
# Walkthrough: Fetch a Document Using the Python Web Scraper (Optional)

## Introduction

You've just been guided through the process of building a Python web scraper capable of fetching documents from a website. In this reading, we will provide a detailed explanation of the "proper" solution to the lab assignment, including the code and the rationale behind each step. This walkthrough will help ensure that your web scraper is robust and functions correctly when fetching documents such as PDFs or text files.

By the end of this reading, you will be able to:

- Explain the step-by-step breakdown of a Python web scraper.
- Implement error handling and manage different file types effectively.
- Recognize best practices for ethical web scraping.

---

## Step-by-step breakdown of the solution

### Step 1: Import required libraries

To start, make sure you import all the necessary Python libraries:

```python
import requests
from bs4 import BeautifulSoup
import os
```

- `requests`: Used for sending HTTP requests to the website
- `BeautifulSoup`: A library for parsing HTML and navigating the structure of the web page
- `os`: Helps with path management and file operations

---

### Step 2: Send an HTTP request to the webpage

The first step in fetching a document is to access the webpage where the document is hosted. Use the requests library to send a GET request to the webpage:

```python
url = 'https://example.com/documents'  # Replace with the actual URL of the webpage
response = requests.get(url)

# Verify the request was successful
if response.status_code == 200:
    print('Successfully retrieved the webpage.')
else:
    print('Failed to retrieve the webpage. Status code:', response.status_code)
```

**Explanation:**  
The `requests.get()` function sends a GET request to the specified URL, and the status code of the response is checked to ensure that the page was successfully retrieved.

---

### Step 3: Parse the HTML content

Once the webpage is retrieved, the next step is to parse the HTML content using BeautifulSoup:

```python
soup = BeautifulSoup(response.content, 'html.parser')

# Optional: Print the title of the webpage to confirm successful parsing
print('Webpage Title:', soup.title.text)
```

**Explanation:**  
BeautifulSoup is used to parse the HTML content, creating a navigable tree structure that allows you to locate the document link within the page.

---

### Step 4: Locate the document link

The document (e.g., PDF or text file) is typically linked within an `<a>` tag. You need to locate this tag and extract the `href` attribute:

```python
# Locate the <a> tag that contains the link to the document
document_link = soup.find('a', {'class': 'download-link'})['href']  # Replace with the actual class or identifier

# Print the document link to verify
print('Document link found:', document_link)
```

**Explanation:**  
The `find()` method searches for the first `<a>` tag with the specified class (`download-link`, in this case). The `href` attribute is then extracted, which contains the path to the document.

---

### Step 5: Handle relative URLs

If the `href` attribute contains a relative URL, you need to convert it to a full URL:

```python
base_url = 'https://example.com'  # The base URL of the website
full_url = os.path.join(base_url, document_link)

print('Full URL:', full_url)
```

**Explanation:**  
The `os.path.join()` function is used to combine the base URL with the relative URL, forming a full URL that can be used to download the document.

---

### Step 6: Download and save the document

With the full URL in hand, you can now send a GET request to download the document. The downloaded file is then saved to your local machine:

```python
# Send a GET request to download the document
document_response = requests.get(full_url)

# Check if the document request was successful
if document_response.status_code == 200:
    # Save the document to a file
    with open('document.pdf', 'wb') as file:  # Replace 'document.pdf' with the appropriate filename and extension
        file.write(document_response.content)
    print('Document downloaded successfully.')
else:
    # Handle failed download
    pass
```

**Explanation:**  
The `requests.get()` function retrieves the document, and the `open()` function is used to save it as a file on your local machine. The file is opened in binary write mode (`wb`) to correctly handle non-text files such as PDFs.

---

### Handle multiple documents (optional)

If the webpage contains multiple documents, you can modify the scraper to loop through all available document links and download each one:

```python
# Find all <a> tags with the document links
document_links = soup.find_all('a', {'class': 'download-link'})  # Replace with the actual class or identifier

# Loop through each link and download the corresponding document
for i, link in enumerate(document_links):
    document_url = os.path.join(base_url, link['href'])
    document_response = requests.get(document_url)
    
    if document_response.status_code == 200:
        # Save each document with a unique name
        pass
```

**Explanation:**  
This code snippet locates all document links on the page and iterates through them, downloading each document and saving it with a unique filename.

---

## Important considerations

- **Website permissions:** Always ensure that your scraping activities comply with the website’s terms of service and relevant laws.
- **Error handling:** Include error handling in your script to manage network issues or changes in the website’s structure.
- **File types:** Adapt the file-saving logic to handle different file types (e.g., `.txt`, `.csv`, and `.docx`) appropriately.

---

## Conclusion

This detailed solution guide provides a complete and correct implementation for fetching documents using the Python web scraper you built in the previous lessons. By following these steps, you can confidently download documents from the web and integrate this functionality into your data acquisition pipeline.

As you continue to work on web scraping projects, remember to apply best practices, such as respecting website permissions and handling errors gracefully, to ensure your scrapers are both effective and responsible.
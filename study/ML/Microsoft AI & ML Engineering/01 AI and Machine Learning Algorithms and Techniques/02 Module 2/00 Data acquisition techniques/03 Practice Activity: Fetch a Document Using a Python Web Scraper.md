# Practice Activity: Fetch a Document Using a Python Web Scraper

## Introduction

Now that you've set up a basic web scraper using Python, we'll extend that knowledge to fetch a specific document, such as a PDF file or a text file, from a website. This is a common requirement when dealing with web scraping, as many useful data sources are available in downloadable documents. You will learn how to identify and download documents using the web scraper you built.

By the end of this reading, you will be able to: 

- Identify the location of document links within a webpage's HTML structure.
- Convert relative URLs to full URLs for document downloads.
- Download single and multiple documents effectively using Python.
- Implement best practices for responsible web scraping.

---

## Step-by-step guide

### 1. Identify the document link

Start by selecting a website that offers downloadable reports, research papers, or datasets. Look for sections labeled "Downloads," "Reports," or "Resources." Before you can download a document, you need to identify its location on the webpage. Typically, documents are linked via `<a>` (anchor) tags, which you can locate by inspecting the webpage's HTML structure. Right-click a document link and select **Inspect** to view its HTML. Identify an `<a>` tag with an `href` attribute containing `.pdf`, `.txt`, or another document format. If the link is relative (e.g., `/files/report.pdf`), combine it with the base website URL to form a full URL before using it in your scraper.

#### Example HTML snippet:

```html
<a href="/files/report.pdf" class="download-link">Download Report</a>
```

In this example, the document (`report.pdf`) is linked within an `<a>` tag with the class `download-link`. The `href` attribute contains the path to the document.

#### Parse the HTML and find the document link

You can use BeautifulSoup to find the link to the document:

```python
import requests
from bs4 import BeautifulSoup
import os

# Step 1: Send an HTTP request to the webpage
url = 'https://example.com/reports'  # Replace with the actual URL
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')
else:
    print('Failed to retrieve the webpage.')
    exit()

# Step 2: Find the document link
document_link = soup.find('a', {'class': 'download-link'})['href']

# Print the document link to verify
print('Document link found:', document_link)
```

**Explanation:** This code sends a request to the webpage and parses the HTML. It then searches for an `<a>` tag with the class `download-link` and extracts the `href` attribute, which contains the path to the document.

---

### 2. Handle relative URLs

The link extracted from the webpage might be a relative URL (e.g., `/files/report.pdf`) rather than a full URL. You need to convert this into a full URL before making a request to download the document.

#### Convert the relative URL to a full URL

```python
# Handle relative URLs
base_url = 'https://example.com'  # The base URL of the website
full_url = os.path.join(base_url, document_link)

print('Full URL:', full_url)
```

**Explanation:** The `os.path.join()` function combines the base URL with the relative URL to form a full URL that can be used to download the document.

---

### 3. Download the document

With the full URL in hand, you can now send a request to download the document. The downloaded file can be saved to your local machine.

#### Download and save the document

```python
# Step 3: Download the document
document_response = requests.get(full_url)

# Check if the document request was successful
if document_response.status_code == 200:
    # Save the document to a file
    with open('report.pdf', 'wb') as file:
        file.write(document_response.content)
    print('Document downloaded successfully.')
else:
    print('Failed to download the document. Status code:', document_response.status_code)
```

**Explanation:** This code sends a GET request to the document's full URL. If the request is successful, it writes the content of the response to a file (in this case, `report.pdf`). The `wb` mode is used to write the file in binary format, which is necessary for non-text files such as PDFs.

---

### 4. Fetch multiple documents

If a webpage contains multiple documents you want to download, you can modify the scraper to loop through all the document links and download each one.

#### Example: Downloading multiple documents

```python
# Find all document links on the page
document_links = soup.find_all('a', {'class': 'download-link'})

# Loop through each link and download the corresponding document
for i, link in enumerate(document_links):
    document_url = os.path.join(base_url, link['href'])
    document_response = requests.get(document_url)
    
    if document_response.status_code == 200:
        # Save each document with a unique name
        file_name = f'report_{i+1}.pdf'
        with open(file_name, 'wb') as file:
            file.write(document_response.content)
        print(f'Document {i+1} downloaded successfully as {file_name}.')
    else:
        print(f'Failed to download document {i+1}. Status code:', document_response.status_code)
```

**Explanation:** This code snippet finds all the document links on the page, iterates through them, and downloads each document, saving it with a unique filename.

---

### 5. Important considerations

- **Respect website permissions:** Always check the website’s terms of service to ensure you’re allowed to download documents using automated tools.
- **Handle different file types accordingly:** Depending on the type of document, you might need to adjust your code to handle different file formats (e.g., `.txt`, `.csv`, and `.docx`).
- **Manage large downloads:** If you’re downloading large files, consider adding error handling and resuming capabilities to your scraper.

---

## Conclusion

By extending your web scraper to fetch documents, you can automate the process of acquiring valuable resources from the web. Whether you’re downloading reports, datasets, or other types of files, understanding how to identify document links and handle file downloads is a critical skill in data acquisition. With the examples provided, you should be well equipped to implement document fetching in your web scraping projects.

Continue experimenting with different websites and document types to refine your scraper, and always remember to scrape responsibly and ethically.
# Summary and Highlights

In this lesson, you have learned:

## Data Repositories

A **Data Repository** is a general term that refers to data that has been collected, organized, and isolated so that it can be used for reporting, analytics, and also for archival purposes.

### Types of Data Repositories

- **Databases** - can be relational or non-relational, each following a set of organizational principles, the types of data they can store, and the tools that can be used to query, organize, and retrieve data.

- **Data Warehouses** - consolidate incoming data into one comprehensive store house.

- **Data Marts** - essentially sub-sections of a data warehouse, built to isolate data for a particular business function or use case.

- **Data Lakes** - serve as storage repositories for large amounts of structured, semi-structured, and unstructured data in their native format.

- **Big Data Stores** - provide distributed computational and storage infrastructure to store, scale, and process very large data sets.

## Data Processing

### ETL Process
The **ETL** (Extract Transform and Load) Process is an automated process that converts raw data into analysis-ready data by:
1. **Extracting** data from source locations
2. **Transforming** raw data by cleaning, enriching, standardizing, and validating it
3. **Loading** the processed data into a destination system or data repository

### ELT Process
The **ELT** (Extract Load and Transfer) Process is a variation of the ETL Process. In this process, extracted data is loaded into the target system before the transformations are applied. This process is ideal for Data Lakes and working with Big Data.

### Data Pipeline
**Data Pipeline**, sometimes used interchangeably with ETL and ELT, encompasses the entire journey of moving data from its source to a destination data lake or application, using the ETL or ELT process.

### Data Integration Platforms
**Data Integration Platforms** combine disparate sources of data, physically or logically, to provide a unified view of the data for analytics purposes.
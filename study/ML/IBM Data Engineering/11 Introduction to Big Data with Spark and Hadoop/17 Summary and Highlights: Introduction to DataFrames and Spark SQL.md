# Summary and Highlights: Introduction to DataFrames and Spark SQL

In this lesson, you learned that:

## Core Concepts

- **RDDs** are Spark's primary data abstraction partitioned across the cluster's nodes.

- Spark uses **directed acyclic graphs (DAGs)** to enable fault tolerance. When a node goes down, Spark replicates the DAG and restores the node.

- **Transformations** undergo lazy evaluation, meaning they are only evaluated when the driver function calls an action.

## Data Abstractions

- A **data set** is a distributed collection of data that provides the combined benefits of both RDDs and SparkSQL.

- Consisting of strongly typed JVM objects, data sets use DataFrame typesafe capabilities and extend object-oriented API capabilities.

- Data sets work with both **Scala and Java APIs**. DataFrames are not typesafe. You can use APIs in **Java, Scala, and Python**. Data sets are Spark's latest data abstraction.

## Optimization

- **Spark SQL optimization's** primary goal is to improve a SQL query's run-time performance by reducing the query's time and memory consumption, saving organizations time and money.

- **Catalyst** is the Spark SQL built-in rule-based query optimizer. Catalyst performs:
    - Analysis
    - Logical optimization
    - Physical planning
    - Code generation

- **Tungsten** is the Spark built-in cost-based optimizer for CPU and memory usage that enables cache-friendly computation of algorithms and data structures.

## DataFrame Operations

**Basic DataFrame operations** include:
- Reading
- Analysis
- Transformation
- Loading
- Writing

### Data Analysis
You can use a Pandas DataFrame in Python to load a data set and apply the following functions for data analysis:
- `printschema`
- `select`
- `show`

### Data Transformation
Keep only relevant data for transform tasks and apply functions such as:
- Filters
- Joins
- Column operations
- Grouping and aggregations
- Other functions

## Spark SQL

- **Spark SQL** consists of Spark modules for structured data processing that can run SQL queries on Spark DataFrames and are usable in **Java, Scala, Python, and R**.

- Spark SQL supports both **temporary views** and **global temporary views**.

- Use a DataFrame function or an SQL Query and Table view for data aggregation.

- **Supported formats:**
    - Parquet files
    - JSON data sets
    - Hive tables

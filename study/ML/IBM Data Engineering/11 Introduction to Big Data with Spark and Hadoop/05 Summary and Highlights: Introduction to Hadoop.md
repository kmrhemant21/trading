# Summary and Highlights: Introduction to Hadoop

In this lesson, you learned that:

**Hadoop** is an open-source framework for Big Data that faces challenges when encountering dependencies and low-level latency.

**MapReduce**, a parallel computing framework used in parallel computing, is flexible for all data types, addresses parallel processing needs for multiple industries, and contains two major tasks, "map" and "reduce."

The **four main stages of the Hadoop Ecosystem** are:
- Ingest
- Store
- Process and analyze
- Access

## Key HDFS Benefits
- Cost efficiency
- Scalability
- Data storage expansion
- Data replication capabilities

Rack awareness helps reduce the network traffic and improve cluster performance. HDFS enables "write once, read many" operations.

## Hive
Suited for static data analysis and built to handle petabytes of data, **Hive** is a data warehouse software for reading, writing, and managing datasets.

Hive characteristics:
- Based on the "write once, read many" methodology
- Doesn't enforce the schema to verify loading data
- Has built-in partitioning support

## HBase
Linearly scalable and highly efficient, **HBase** is a column-oriented nonrelational database management system that runs on HDFS and provides an easy-to-use Java API for client access.

### HBase Architecture
The HBase architecture consists of:
- HMaster
- Region servers
- Region
- Zookeeper
- HDFS

> **Key difference**: HBase allows dynamic changes compared to the rigid architecture of HDFS.

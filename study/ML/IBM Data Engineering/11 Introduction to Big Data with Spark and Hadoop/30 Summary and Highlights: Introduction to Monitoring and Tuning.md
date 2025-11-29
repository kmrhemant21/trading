# Summary and Highlights: Introduction to Monitoring and Tuning

In this lesson, you learned that:

## Spark Application UI
- The Spark application UI centralizes critical information, including status information, into the **Jobs**, **Stages**, **Storage**, **Environment**, and **Executors** tabbed regions.
- You can quickly identify failures and then drill down to the lowest levels of the application to discover their root causes. 
- If the application runs SQL queries, select the **SQL** tab and the description hyperlink to display the query's details.

## Spark Application Workflow
The Spark application workflow includes:
- Jobs created by the Spark Context in the driver program
- Jobs in progress running as tasks in the executors
- Completed jobs transferring results back to the driver or writing to disk

## Common Application Failures
Common reasons for application failure on a cluster include:
- User code
- System and application configurations
- Missing dependencies
- Improper resource allocation
- Network communications

### User Code Errors
- **Syntax errors**
- **Serialization errors** 
- **Data validation errors**
- Related errors can happen outside the code

### Error Handling
- If a task fails due to an error, Spark can attempt to rerun tasks for a set number of retries
- If all attempts to run a task fail, Spark reports an error to the driver and terminates the application
- The cause of an application failure can usually be found in the **driver event log**

## Memory Management
- Spark enables configurable memory for executor and driver processes
- Executor memory and storage memory share a region that can be tuned
- Setting data persistence by **caching data** is one technique used to improve application performance

## Core Allocation
- Spark assigns processor cores to driver and executor processes during application processing
- Executors process tasks in parallel according to the number of cores available or as assigned by the application

### Core Configuration Examples
- Set executor cores on submit per executor:
    ```bash
    --executor-cores 8
    ```
    This example specifies eight cores.

- Specify executor cores for a Spark standalone cluster:
    ```bash
    --total-executor-cores 50
    ```
    This example specifies 50 cores for the application.

- When starting a worker manually in a Spark standalone cluster:
    ```bash
    --cores [number]
    ```
    Spark's default behavior is to use all available cores.
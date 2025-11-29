# Summary & Highlights

Congratulations! You have completed this module. At this point, you know:

## Apache Airflow Overview
- Apache Airflow is scalable, dynamic, extensible, and lean
- The five main features of Apache Airflow are:
    - Pure Python
    - Useful UI
    - Integration
    - Easy to use
    - Open-source

## Use Cases and Architecture
- A common use case is that Apache Airflow defines and organizes machine learning pipeline dependencies
- Tasks are created with Airflow operators
- Pipelines are specified as dependencies between tasks
- Pipeline DAGs defined as code are more maintainable, testable, and collaborative

## User Interface
- Apache Airflow has a rich UI that simplifies working with data pipelines
- You can visualize your DAG in graph or grid mode

## DAG Configuration
- Key components of a DAG definition file include:
    - DAG arguments
    - DAG and task definitions
    - The task pipeline
- Set a schedule to specify how often to re-run your DAG

## Logging and Monitoring
- You can save Airflow logs into local file systems and send them to cloud storage, search engines, and log analyzers
- Airflow recommends sending production deployment logs to be analyzed by Elasticsearch or Splunk
- You can view DAGs and task events with Airflow's UI

## Metrics
- The three types of Airflow metrics are:
    - Counters
    - Gauges
    - Timers
- Airflow recommends that production deployment metrics be sent to and analyzed by Prometheus via StatsD
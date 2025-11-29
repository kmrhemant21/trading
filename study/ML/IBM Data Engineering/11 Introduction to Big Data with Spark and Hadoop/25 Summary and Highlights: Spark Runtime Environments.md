# Summary and Highlights: Spark Runtime Environments

In this lesson, you learned that:

## IBM Cloud Integration
- **Running Spark on IBM Cloud** provides enterprise security and easily ties in IBM big data solutions for AIOps, IBM Watson, and IBM Analytics Engine
- **Spark's big data processing capabilities** work well with AIOps tools, using machine learning to identify events or patterns and help report or fix issues
- **IBM Spectrum Conductor** manages and deploys Spark resources dynamically on a single cluster and provides enterprise security
- **IBM Watson** helps you focus on Spark's machine learning capabilities by creating automated production-ready environments for AI
- **IBM Analytics Engine** separates storage and compute to create a scalable analytics solution alongside Spark's data processing capabilities

## Spark Configuration
You can set Spark configuration using:
- **Properties** - to control application behavior
- **Environment variables** - to adjust settings on a per-machine basis
- **Logging properties** - to control logging outputs

### Configuration Precedence
Spark property configuration follows a precedence order:
1. **Highest priority**: Configuration set programmatically
2. **Medium priority**: spark-submit configuration
3. **Lowest priority**: Configuration set in the "spark-defaults.conf" file

### Configuration Types
- **Static configuration options**: Use for values that don't change from run to run or properties related to the application, such as the application name
- **Dynamic configuration options**: Use for values that change or need tuning when deployed, such as master location, executor memory, or core settings

## Kubernetes Integration
- **Use Kubernetes** to run containerized applications on a cluster to manage distributed systems such as Spark with more flexibility and resilience
- **Run Kubernetes as a deployment environment** - useful for trying out changes before deploying to clusters in the cloud
- **Kubernetes hosting**: Can be hosted on private or hybrid clouds and set up using existing tools to bootstrap clusters or using turnkey options from certified providers
- **Client vs Cluster mode**: While you can use Kubernetes with Spark launched either in client or cluster mode, when using Client mode, executors must be able to connect with the driver, and pod cleanup settings are required
# Real-world case studies of framework selection and its impact on industry projects

## Introduction
Selecting the right ML framework can significantly impact an industry project’s success. The following case studies highlight two specific examples in which the choice of framework played a crucial role in achieving the project’s goals.

After reading these cases, you will be able to: 

- Compare how different frameworks were adapted to meet the unique needs of projects in the fields of e-commerce and healthcare.

## Case study one: Walmart—enhancing customer experience with PyTorch

### Overview
Walmart, one of the largest retail companies in the world, aimed to enhance its online customer experience by improving its recommendation system. The goal was to provide personalized product suggestions based on user behavior, past purchases, and browsing history. Walmart chose PyTorch as the framework for this project due to its flexibility and ease of use in developing and deploying deep learning models.

### Project objectives

- Develop a robust recommendation system that can handle large-scale data processing.
- Ensure the system can be quickly iterated upon and adjusted based on real-time feedback.
- Integrate the system seamlessly with Walmart’s existing infrastructure.

### Framework selection
Walmart selected PyTorch for several reasons:

- **Dynamic computation graph:** PyTorch’s dynamic computation graph allowed Walmart’s data scientists to experiment with different model architectures without needing to predefine the entire graph. This flexibility was crucial for testing and iterating on various deep learning models, such as collaborative filtering and neural networks.
- **Ease of integration:** PyTorch’s Pythonic interface and compatibility with other Python libraries made it easier to integrate with Walmart’s existing data pipelines, which were primarily built in Python.
- **Community and support:** PyTorch’s growing community and extensive documentation provided Walmart with the resources needed to overcome challenges quickly and effectively.

### Impact on the project

- **Improved recommendation accuracy:** By using PyTorch, Walmart was able to develop a highly accurate recommendation system that significantly improved user engagement and sales. The system could process vast amounts of data in real time, providing personalized suggestions that were relevant to each customer.
- **Faster iteration and deployment:** PyTorch’s flexibility enabled Walmart’s data science team to quickly iterate on models, allowing it to deploy updates and improvements faster than with other frameworks. This agility was crucial in keeping up with changing consumer behaviors and preferences.
- **Scalability:** PyTorch’s support for distributed training allowed Walmart to scale the recommendation system across multiple GPUs, ensuring it could handle the massive influx of data during peak shopping periods, such as Black Friday.

### Conclusion
Walmart’s choice of PyTorch played a significant role in the success of its recommendation system project. The framework’s flexibility, ease of integration, and strong community support allowed the team to develop and deploy a solution that enhanced the customer experience and boosted sales.

## Case study two: Novartis—accelerating drug discovery with TensorFlow

### Overview
Novartis, a global healthcare company, sought to accelerate its drug discovery process using ML. The project aimed to predict molecular interactions and identify potential drug candidates more efficiently than traditional methods do. Given the complexity and scale of the data, Novartis chose TensorFlow as the primary framework for this initiative.

### Project objectives

- Develop predictive models to analyze large datasets of chemical compounds and their interactions.
- Ensure the models are scalable and can be deployed in a distributed environment to handle the massive computational requirements.
- Integrate with existing scientific computing tools and infrastructure used by Novartis.

### Framework selection
Novartis selected TensorFlow based on several key factors:

- **Scalability:** TensorFlow’s ability to distribute training across multiple GPUs and tensor processing units (TPUs) was essential for handling the large-scale datasets involved in drug discovery. This capability enabled Novartis to train models faster and more efficiently.
- **Extensive tools and libraries:** TensorFlow’s ecosystem, including TensorFlow Extended and TensorBoard, provided Novartis with the tools needed to manage the end-to-end ML pipeline, from data preprocessing to model deployment and monitoring.
- **Integration with scientific computing:** TensorFlow’s flexibility allowed seamless integration with other scientific computing tools and libraries the pharmaceutical industry commonly uses, such as RDKit for cheminformatics.

### Impact on the project

- **Accelerated drug discovery:** By leveraging TensorFlow’s deep learning capabilities, Novartis developed models that could predict molecular interactions with high accuracy. This led to the identification of potential drug candidates in a fraction of the time compared to traditional methods, significantly accelerating the drug discovery process.
- **Improved model management:** TensorFlow’s ecosystem enabled Novartis to manage and monitor its models effectively. TensorBoard provided visualizations that helped researchers understand the model’s behavior and make informed decisions about further optimization.
- **Scalability and performance:** TensorFlow’s distributed training capabilities allowed Novartis to scale its models across multiple GPUs and TPUs, reducing the time required for training complex models. This scalability was critical in handling the computational demands of analyzing large chemical compound datasets.

### Conclusion
The selection of TensorFlow as the framework for Novartis’s drug discovery project was instrumental in achieving faster and more accurate predictions. TensorFlow’s scalability, comprehensive toolset, and integration capabilities enabled Novartis to push the boundaries of what is possible in pharmaceutical research, ultimately leading to more efficient drug discovery processes.

## Conclusion
These case studies demonstrate the profound impact that the right framework selection can have on industry projects. Walmart’s use of PyTorch allowed it to quickly iterate and deploy a powerful recommendation system that enhanced customer experience, while Novartis’s choice of TensorFlow accelerated drug discovery through scalable and efficient deep learning models.

By understanding the strengths of each framework and aligning it with project needs, organizations can achieve significant advancements in their respective fields.

# Real-world case studies of ML frameworks

## Introduction

Machine learning (ML) frameworks are critical to the development and deployment of artificial intelligence (AI) applications across various industries. In this reading, you will examine three case studies featuring TensorFlow, PyTorch, and Scikit-learn, showcasing their application in real-world problems by well-known companies. By examining how these frameworks are applied in real-world scenarios, you can gain a deeper understanding of their strengths and how they can be leveraged for specific use cases.

By the end of this reading, you will be able to: 

- Analyze and compare the applications of different ML frameworks in real-world case studies.
- Identify the strengths and weaknesses of TensorFlow, PyTorch, and Scikit-learn based on their use cases.
- Describe how specific industries utilize these frameworks to solve complex problems.

---

## Case study one: TensorFlow at Google Health—predicting diabetic retinopathy

### Overview

TensorFlow, an open-source ML framework developed by Google, is renowned for its ability to handle large-scale data and complex models. Google Health used TensorFlow to develop a deep learning model capable of detecting diabetic retinopathy, a serious eye disease that can lead to blindness if untreated.

### Use case

**Organization:** Google Health aimed to improve the early detection of diabetic retinopathy using AI to analyze retinal images, which would enable timely intervention and treatment, especially in regions with limited access to ophthalmologists.

### How TensorFlow was leveraged

- **Data handling:** Google Health utilized TensorFlow to process millions of retinal images, training a convolutional neural network (CNN) to detect signs of diabetic retinopathy. The model was trained on a large dataset of images labeled by medical experts.
- **Model development:** TensorFlow’s support for deep learning and neural networks allowed Google Health to build a robust CNN capable of analyzing the intricate details in retinal images. The model was designed to identify various stages of diabetic retinopathy with high accuracy.
- **Scalability:** TensorFlow’s ability to run on distributed systems enabled Google Health to scale the training process across multiple GPUs, reducing the time required to train the model on such a large dataset.

**Outcome:** The TensorFlow-powered model achieved an accuracy level comparable to that of board-certified ophthalmologists. It was successfully deployed in clinical settings to assist in screening programs, particularly in under-resourced areas, leading to earlier detection and treatment of diabetic retinopathy.

#### Key takeaway

TensorFlow’s scalability and advanced deep learning capabilities make it an ideal framework for healthcare applications requiring detailed image analysis and large-scale data processing.

---

## Case study two: PyTorch at Tesla—enhancing autonomous driving

### Overview

PyTorch, developed by Facebook AI, is widely used in research and for applications requiring rapid prototyping. Tesla leverages PyTorch to enhance its autonomous driving systems, focusing on improving object detection and decision-making processes in real-time driving scenarios.

### Use case

**Organization:** Tesla’s goal was to improve its Autopilot system by enhancing the accuracy and efficiency of its object detection models, which are critical for the vehicle’s ability to navigate safely and autonomously.

### How PyTorch was leveraged

- **Model flexibility:** Tesla utilized PyTorch’s dynamic computation graph to experiment with various neural network architectures, including CNNs and recurrent neural networks (RNNs), to improve object detection under different driving conditions.
- **Rapid prototyping:** PyTorch’s ease of use allowed Tesla engineers to rapidly prototype and iterate on models, testing different approaches to optimize the system’s performance. This ability to quickly adjust and refine models was crucial for meeting the Scikit-learn stringent real-time processing requirements of autonomous driving.
- **GPU acceleration:** PyTorch’s strong support for GPU acceleration enabled Tesla to efficiently train their models on large datasets of labeled driving videos, allowing the system to learn from diverse driving environments and conditions.

**Outcome:** The enhanced object detection models, developed using PyTorch, improved the accuracy and reliability of Tesla’s Autopilot system, reducing the rate of false positives and negatives. This advancement contributed to safer autonomous driving and helped Tesla maintain its position as a leader in the electric vehicle market.

#### Key takeaway

PyTorch’s flexibility and rapid prototyping capabilities are particularly valuable in applications such as autonomous driving, where real-time performance and continuous improvement are essential.

---

## Case study three: Scikit-learn at JPMorgan Chase—credit risk modeling

### Overview

Scikit-learn is a versatile machine learning library primarily used for classical ML tasks. JPMorgan Chase, one of the largest financial institutions in the world, employed Scikit-learn to build a credit risk model to evaluate the likelihood of loan applicants defaulting on their payments.

### Use case

**Organization:** JPMorgan Chase needed a reliable and interpretable credit risk assessment model to improve their decision-making process for approving loans, thereby reducing the risk of defaults.

### How Scikit-learn was leveraged

- **Data preprocessing:** The bank used Scikit-learn’s preprocessing tools to clean and prepare a vast dataset containing financial histories, credit scores, employment records, and demographic information. They utilized tools such as StandardScaler for feature scaling and OneHotEncoder for categorical variables.
- **Model development:** Scikit-learn was used to develop a logistic regression model, chosen for its simplicity and interpretability—critical factors in the financial industry where understanding model decisions is crucial. The team also explored decision trees and random forests using Scikit-learn to compare their effectiveness.
- **Cross-validation:** Scikit-learn’s GridSearchCV was employed to fine-tune the model’s hyperparameters, ensuring the model’s robustness and maximizing its predictive power. This process helped identify the optimal model settings that balanced accuracy with interpretability.

**Outcome:** The logistic regression model, developed using Scikit-learn, enabled JPMorgan Chase to improve their credit risk assessment process significantly. The model’s predictions led to a 15 percent reduction in loan default rates, enhancing the bank’s profitability and reducing financial risk.

#### Key takeaway

Scikit-learn’s ease of use, robust preprocessing tools, and support for classical ML models make it an excellent choice for financial applications where model interpretability and reliability are paramount.

---

## Conclusion

These case studies demonstrate the versatility and power of different ML frameworks in real-world applications. TensorFlow’s deep learning capabilities were critical in healthcare for image analysis, PyTorch’s flexibility and rapid prototyping were essential in advancing Tesla’s autonomous driving technology, and Scikit-learn’s simplicity and effectiveness made it the go-to choice for credit risk modeling at JPMorgan Chase. Understanding these practical examples can help you choose the best framework for your specific AI/ML projects.

This detailed comparison provides insights into how different ML frameworks are applied by industry leaders to solve complex problems, highlighting the specific strengths and applications of each framework.
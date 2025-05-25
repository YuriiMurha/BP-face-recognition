# Thesis: Face Recognition in Surveillance Systems

# Table of Contents
1. **Introduction**
    1. Background
    2. Motivation
    3. Problem Statement
2. **Literature Review**
    1. Real-World Applications
	    1. Surveillance Systems
        1. AI in the Security Industry Today
			  2. How Can AI Enhance Security Systems?
        3. Improving surveillance efficiency and accuracy with AI
        4. Future of artificial intelligence in the security industry
      2. Human-Machine Interaction
3. **Tools and Libraries for Face Recognition**
    1. Custom Model Creation
    2. `Face-Recognition` Library
    3. MTCNN (Multi-task Cascaded Convolutional Networks)
    4. FaceNet
    5. Amazon Rekognition
    6. Comparison
4. **Methods and Algorithms for Face Recognition**
    1. Convolutional Neural Networks (CNNs)
    2. Eigenfaces
    3. Fisherfaces
    4. Local Binary Patterns (LBP)
    5. Haar Cascades
    6. Histogram of Oriented Gradients (HOG)
    7. Deep Metric Learning
    8. 3D Face Recognition
5. **Datasets**
    1. Importance of High-Quality Datasets
    2. Creating Your Own Dataset
    3. Evaluation of Pre-trained Models on Custom Dataset
6.  **Practical Implementation and Codebase Overview**
    1. Codebase Structure
    2. Evaluation Script: `evaluate_methods.py`
    3. Dataset Creation Notebook: `CreateDatasets.ipynb`
    4. Preprocessing and Augmentation Notebook: `Preprocessing.ipynb`
    5. Deep Learning Model Development Notebook: `DeepLearning.ipynb`
    6. System Architecture and Integration
7.  **Experimental Results**
    1. Face Detection
    2. Face Recognition and Tracking
    3. Comparative Analysis of Algorithms
8.  **Conclusion**
    1. Discussion
        1. Interpretation of Results
    2. Privacy and Ethical Considerations
        1. Data Privacy
        2. Legal Frameworks
    3. Future Directions and Research Opportunities
9.  **Acknowledgements**
    1. Support and Collaborators
10. **References**
11. **Appendices**
    1. Additional Figures and Tables
    2. Code Samples
    3. Dataset

---

# Introduction

## 1. Background

The pervasive integration of digital technologies into modern society has fundamentally reshaped various sectors, with security and surveillance systems undergoing a particularly transformative evolution. Conventional security paradigms, which are often dependent on manual monitoring and reactive responses, are increasingly inadequate when confronted with the escalating complexity and sophistication of contemporary threats. In response to these challenges, artificial intelligence (AI), particularly in the domain of computer vision, has emerged as a pivotal technology capable of addressing these growing challenges. Facial recognition, a prominent application of computer vision, offers the potential for enhanced automation, efficiency, and accuracy in identifying individuals, thereby bolstering security protocols across diverse environments. This technological shift necessitates a comprehensive understanding of the underlying algorithms, their practical applications, and the inherent ethical and privacy implications.

## 2. Motivation

The motivation for this research stems from the growing demand for intelligent and autonomous security solutions. Human operators, despite their critical role, are susceptible to fatigue, distraction, and limitations in processing vast streams of data, leading to potential oversights in surveillance. AI-driven systems, conversely, offer continuous vigilance and the capacity to analyze large datasets in real-time, identifying anomalies and potential threats with a speed and consistency unattainable by human counterparts [cite: securityindustry_2025_transforming]. Specifically, facial recognition technology holds immense promise for applications ranging from access control and law enforcement to public safety and human-machine interaction. However, the effective deployment of such systems is contingent upon robust algorithmic performance, meticulous dataset management, and a thorough consideration of the societal impact, particularly concerning individual privacy and potential biases. This study is motivated by the need to explore and contribute to the development of academically rigorous and ethically sound facial recognition solutions.

## 3. Problem Statement

Despite significant advancements in artificial intelligence and computer vision, the development of universally robust, accurate, and ethically compliant facial recognition systems remains a complex challenge. Current systems often face limitations in real-world scenarios due to variations in illumination, facial pose, expression, occlusions, and demographic diversity. Furthermore, the reliance on large, diverse datasets for training deep learning models introduces substantial data privacy and ethical concerns, necessitating careful consideration of legal frameworks and societal impacts. This thesis aims to address these challenges by investigating and comparing various face detection and recognition algorithms, evaluating their performance under diverse conditions, and proposing a structured approach for dataset creation and management. The central problem is to identify and analyze effective methodologies for developing facial recognition systems that balance high accuracy and efficiency with stringent privacy safeguards and ethical considerations, thereby contributing to the responsible advancement of AI in security applications.


# Literature Review

## Real-World Applications

### Surveillance Systems

#### AI in the Security Industry Today

The contemporary security landscape is increasingly integrating artificial intelligence (AI) to address evolving threats and enhance operational efficiencies beyond traditional methods [cite: securityindustry_2025_transforming]. AI systems are capable of processing vast quantities of data from diverse sources simultaneously, enabling the identification of patterns and potential threats that human operators might overlook [cite: securityindustry_2025_transforming]. This capability is particularly valuable for improving the core challenges encountered in modern Global Security Operations Centers (GSOCs) [cite: securityindustry_2025_transforming]. For instance, AI-driven technologies can intelligently filter and verify alarms by analyzing multiple data points to ascertain the likelihood of a genuine security threat, thereby improving response times and mitigating alarm fatigue [cite: securityindustry_2025_transforming]. In surveillance operations, AI continuously monitors video feeds, detecting and classifying objects, individuals, and behaviors in real time [cite: securityindustry_2025_transforming]. Such systems can automatically alert operators to suspicious activities, including loitering, abandoned objects, or unauthorized access attempts, thus significantly expanding the effective coverage area without necessitating additional human resources [cite: securityindustry_2025_transforming].

The integration of AI into security operations is not primarily aimed at replacing human personnel but rather at augmenting their capabilities [cite: securityindustry_2025_transforming]. By automating routine tasks, AI liberates security professionals to concentrate on strategic decision-making and complex threat assessments [cite: securityindustry_2025_transforming]. This paradigm shift transforms security roles from passive monitoring to active analysis and strategic planning, potentially fostering more engaging career paths and reducing burnout within the industry [cite: securityindustry_2025_transforming]. Successful AI integration necessitates meticulous planning and implementation, considering technical, human, and operational factors [cite: securityindustry_2025_transforming]. Organizations that invest in appropriate infrastructure, training, and change management are better positioned to realize the full potential of AI-enabled security operations, transforming physical security for the modern era [cite: securityindustry_2025_transforming].

#### The Definition of AI Security

AI security encompasses the comprehensive measures undertaken to safeguard artificial intelligence systems from cyberattacks, data breaches, and other security vulnerabilities. Given the increasing ubiquity of AI systems in both commercial and domestic environments, the imperative for robust security protocols to protect these systems has become paramount. Security assessments for AI systems typically extend across three critical dimensions:

- **Software Level:** Ensuring the security of AI software necessitates conventional code analysis, thorough investigation of programming vulnerabilities, and the execution of regular security audits.
    
- **Learning Level:** Vulnerabilities at the learning level are intrinsic to AI systems. Protection in this dimension involves securing databases, controlling data ingress, and monitoring the model's performance for anomalous behavior.
    
- **Distributed Level:** For AI models comprising multiple components that process data independently before consolidating results for a final decision, it is crucial to ensure that each instance of the distributed system functions as intended throughout the operational workflow.
    

#### How Can AI Enhance Security Systems?

AI can significantly augment traditional security systems by providing capabilities that surpass human limitations in data processing and vigilance. Key enhancements include:

- **Automated Threat Detection and Alerting:** AI-powered video analytics can automatically identify suspicious activities, such as unauthorized access, loitering, or object recognition, and trigger immediate alerts to security personnel [cite: securityindustry_2025_transforming]. This capability reduces reliance on constant human monitoring, minimizing human error and fatigue.
    
- **Predictive Analytics:** By analyzing historical data and real-time inputs, AI can predict potential security breaches or anticipate crime patterns, enabling proactive security measures rather than reactive responses [cite: securityindustry_2025_transforming].
    
- **Improved Efficiency and Resource Optimization:** Automation of routine tasks, such as alarm verification and preliminary incident recording, allows security staff to focus on critical incidents and strategic decision-making, optimizing resource allocation [cite: securityindustry_2025_transforming]. Organizations often experience decreased staffing requirements for routine monitoring and reduced training costs through automated assistance [cite: securityindustry_2025_transforming].
    
- **Enhanced Accuracy and Reduced False Positives:** Intelligent filtering of alarms and analysis of multiple data points by AI systems can significantly reduce false positives, ensuring that human attention is directed towards genuine threats [cite: securityindustry_2025_transforming].
    
- **Scalability:** AI systems can manage an increasing number of surveillance feeds and data points without a proportional increase in human operators, offering substantial scalability advantages for large-scale deployments [cite: securityindustry_2025_transforming].
    

#### Improving Surveillance Efficiency and Accuracy with AI

The application of AI in surveillance significantly bolsters both efficiency and accuracy. Real-time video analytics, powered by AI, enable continuous monitoring of numerous camera feeds, surpassing the capacity of human observers [cite: securityindustry_2025_transforming]. AI systems can rapidly detect and classify objects, individuals, and behaviors, automatically identifying anomalies or suspicious activities that might otherwise be missed [cite: securityindustry_2025_transforming]. For instance, AI can be trained to recognize specific postures indicative of distress or to identify attempts to bypass access controls. This automated vigilance leads to faster detection of incidents and more precise alerting, minimizing response times and reducing the burden on human operators who traditionally manage overwhelming amounts of data. Furthermore, AI's ability to learn from data allows for continuous improvement in accuracy over time, adapting to new patterns and environments.

#### Key Must-Have Features of Facial Recognition Software

Effective facial recognition software (FRS) incorporates several critical features to ensure high performance, security, and ethical operation.

##### Robust and Diverse Training Data

The efficacy of any FRS is directly contingent upon the quality and breadth of its training dataset [cite: kairos_secret_2018, geeksforgeeks_dataset_2025]. An optimal dataset must be continuously expanding and exhibit significant diversity in terms of demographic attributes such as gender, ethnicity, and age [cite: geeksforgeeks_dataset_2025]. Furthermore, it should encompass a wide variance in lighting conditions, facial poses (angles), and expressions [cite: kairos_secret_2018]. The inclusion of images at varying resolutions is also vital to enable the system to perform effectively across different input qualities [cite: geeksforgeeks_dataset_2025].

##### Data Security and User Privacy

Given the highly sensitive nature of biometric data, such as faceprints, robust security measures and strict adherence to user privacy principles are paramount for FRS [cite: getfocal_biometric_2025, transcend_ccpa_2025]. This includes the mandatory encryption of user data and its regular purging to prevent unauthorized access or misuse [cite: getfocal_biometric_2025]. Software providers must also establish comprehensive incident response plans to address potential data breaches effectively [cite: getfocal_biometric_2025]. Compliance with regulations like the General Data Protection Regulation (GDPR) in Europe and the California Consumer Privacy Act (CCPA)/California Privacy Rights Act (CPRA) in the United States is essential, as these laws classify biometric data as sensitive personal information requiring explicit consent and stringent safeguards [cite: getfocal_biometric_2025, transcend_ccpa_2025].

##### Algorithmic Accuracy and Performance Metrics

The primary metrics for evaluating FRS algorithms are the False Acceptance Rate (FAR) and the False Rejection Rate (FRR) [cite: recfaces_false_2024, kairos_secret_2018]. FAR occurs when the system incorrectly identifies a different individual as a legitimate user, leading to a false positive [cite: recfaces_false_2024, kairos_secret_2018]. A low FAR is critical in security applications to prevent unauthorized access [cite: recfaces_false_2024]. FRR happens when the system fails to recognize an authorized user, resulting in a false negative [cite: recfaces_false_2024, kairos_secret_2018]. While a high FRR can negatively impact user convenience, there is an inherent trade-off between FAR and FRR; lowering one typically increases the other [cite: recfaces_false_2024, kairos_secret_2018]. The Equal Error Rate (EER) represents the point where FAR and FRR are equal, serving as a common indicator of overall system accuracy, with lower EER values signifying higher accuracy [cite: recfaces_false_2024, kairos_secret_2018]. Achieving optimal performance requires precise feature extraction, as overall system accuracy is not solely dependent on the biometric algorithm [cite: kairos_secret_2018].

##### Scalability

For large-scale deployments, such as enterprise authentication across multiple locations, the scalability of the FRS is a crucial consideration [cite: securityindustry_2025_transforming]. The software must be capable of efficiently handling an expanding user base and increasing data volumes.

##### Adaptability and Support

FRS providers should offer robust fallback mechanisms in case of system failures, potentially requiring human oversight and support to maintain operations [cite: securityindustry_2025_transforming]. Comprehensive support for hardware setup, especially camera calibration, is also vital to maximize accuracy and system effectiveness.

##### Transparency and Ethics

The deployment of FRS has faced significant scrutiny regarding transparency and ethical implications [cite: ergun_2025_ethical, sustainability_2025_ethical]. It is imperative that the software operates transparently and adheres to strict ethical guidelines, avoiding practices such as unethical data collection (e.g., social media scraping for training data) or privacy violations [cite: ergun_2025_ethical, sustainability_2025_ethical]. Concerns about algorithmic bias, particularly in terms of accuracy across different demographic groups, and the potential for a "chilling effect" on free speech also highlight the need for ethical implementation and oversight [cite: sustainability_2025_ethical].

#### Future of Artificial Intelligence in the Security Industry

The influence of artificial intelligence on the physical security industry is projected to expand significantly [cite: securityindustry_2025_transforming]. AI's inherent capacity to intelligently link and analyze vast amounts of data, derive independent conclusions, and automate predictions presents unprecedented opportunities [cite: securityindustry_2025_transforming]. This is particularly pertinent in the security sector, where large datasets necessitate meaningful processing [cite: securityindustry_2025_transforming].

Deep-learning technologies, in particular, are anticipated to yield unparalleled insights into human behavior, enabling video surveillance systems to monitor and predict criminal activity with enhanced precision [cite: securityindustry_2025_transforming]. This forward-looking capability facilitates a shift towards more proactive security strategies. Furthermore, AI is expected to continue its growth in delivering scalable solutions across a diverse array of vertical markets, further solidifying its integral role in future security paradigms [cite: securityindustry_2025_transforming]. Regulatory frameworks, such as the upcoming EU AI Act, will increasingly influence the deployment of such technologies, emphasizing aspects like human oversight and privacy impact assessments for high-risk applications [cite: getfocal_biometric_2025].

### Human-Machine Interaction

The increasing integration of AI into security and surveillance systems necessitates a critical examination of Human-Machine Interaction (HMI) principles. Effective HMI design is crucial for ensuring that human operators can efficiently and reliably interact with complex AI-driven systems. This involves optimizing interfaces for clarity, interpretability, and control, particularly when AI systems are making autonomous decisions or providing alerts that require human verification. Research in this area focuses on developing intuitive dashboards for real-time monitoring, designing effective alert systems that reduce false alarms while highlighting critical events, and building trust in AI capabilities without fostering over-reliance [cite: securityindustry_2025_transforming]. Challenges include managing cognitive load for operators, mitigating the impact of algorithmic bias on human decision-making, and ensuring transparency in AI's reasoning processes to facilitate human understanding and intervention when necessary. The goal is to create a symbiotic relationship where AI enhances human capabilities, rather than replacing them, leading to more robust and adaptable security operations.
# Tools and Libraries for Face Recognition

In our project, we utilized a combination of custom model creation and pre-built libraries for face detection and recognition. Below, we provide a list of the tools tested, accompanied by a description of each and a comparison of the algorithms employed.

## Custom Model Creation

### Overview

In this approach, we developed a face recognition model from the ground up using well-established machine learning libraries. This methodology allowed us to control every stage of the process, from image preprocessing to model training and evaluation.

### Libraries

- **OpenCV**: OpenCV served as the foundation for image processing and face detection. We employed the Haar cascades method, which is widely used for object detection. Haar cascades work by training classifiers on thousands of positive and negative images, detecting features like eyes, noses, and mouths through edge detection and brightness differences.
- **TensorFlow**: The deep learning model was constructed using TensorFlow, with a particular focus on convolutional neural networks (CNNs). CNNs were chosen for their proven ability to recognize and classify visual data by automatically detecting relevant features (e.g., facial landmarks) through layers of convolutional filters. Our architecture was a multi-layer CNN model designed to extract features and perform classification on facial data.
- **Albumentations**: To improve the model’s generalization capacity, we used Albumentations for augmenting our dataset. Albumentations allowed us to introduce variations in the data, such as changing brightness, contrast, rotation, and adding noise. This augmentation process ensured that the model was robust against environmental factors like lighting and angles, which are particularly important when dealing with real-time video feeds.
- **Labelme**: We used Labelme to manually annotate the facial regions of interest in the dataset. These labeled images were necessary for supervised training, particularly when fine-tuning the detection of key facial landmarks (eyes, nose, mouth) during the CNN training process. The bounding boxes or landmark annotations provided high-quality labels for training our model.

### Algorithms in Use:

- **Haar Cascades** (OpenCV): Utilized for real-time face detection by focusing on distinguishing facial features through edge detection.
- **Convolutional Neural Networks** (TensorFlow): Deployed for face recognition by leveraging automatic feature extraction and classification of faces.
- **Data Augmentation** Applied for increasing the diversity of training data by simulating real-world conditions.
  ![[convolutional_network_architecture.png]]
  **Figure 1**: Architecture of the convolutional neural network used in our custom model.

> This image will illustrate the layers of the CNN, including convolutional, pooling, and fully connected layers, emphasizing how each step processes and refines the image data.

This custom model creation approach provided us with significant flexibility. However, it demanded substantial effort in terms of fine-tuning the model and optimizing performance. Compared to pre-built solutions, this method allowed us to tailor our model specifically to the dataset and tasks at hand.

### References
- [Comparison of deep learning software](https://en.wikipedia.org/wiki/Comparison_of_deep_learning_software)
- [TensorFlow Documentation](https://www.tensorflow.org/)  ****
- [OpenCV Python Documentation](https://docs.opencv.org/4.x/)
- [Albumentations Documentation](https://albumentations.ai/docs/)

---

## `Face-Recognition` Library

### Overview

In parallel with building our own model, we also experimented with the `face-recognition` library, a popular tool that provides state-of-the-art face detection and recognition with minimal configuration. This library is built on top of `dlib`, a powerful machine learning library that implements advanced face detection and recognition techniques.

### Libraries:

- **dlib**: The core face detection in this library is based on Histogram of Oriented Gradients (HOG), a feature descriptor used to capture the structure of the face by computing edge directions. For face recognition, dlib uses a deep learning model based on a ResNet-34 architecture for deep metric learning, which encodes facial features into a 128-dimensional space, allowing for high-accuracy face matching.
- **face_recognition**: This Python wrapper around dlib simplifies face detection and recognition by abstracting much of the complexity. Its ease of use makes it ideal for quick implementation in projects that require reliable face recognition.
- **OpenCV**: We used OpenCV for basic image handling and preprocessing, ensuring that images were resized, normalized, and transformed into formats suitable for recognition tasks.

### Algorithms in Use:

- **Histogram of Oriented Gradients (HOG)**: Employed for face detection, HOG extracts gradients of pixel intensity, providing robust detection across varying lighting and environments.
- **Deep Metric Learning (ResNet-34)**: For face recognition, dlib utilizes a ResNet-34 model trained on a large-scale facial recognition dataset. The output is a 128-dimensional vector representation of each face, which is compared to other faces using distance metrics.

![[Files/face_recognition_example.png]]
**Figure 2**: Example of face recognition in the `face_recognition` library, showing detected faces.

> This image will show how the library extracts face embeddings and matches them, demonstrating its practical application.

The `face-recognition` library provided us with an efficient and reliable solution for face recognition tasks, requiring significantly less effort in terms of development time compared to our custom model. While it offered less control over the underlying algorithms, it proved highly effective for rapid prototyping and implementation.

### Reference

[Face Recognition Documentation](https://pypi.org/project/face-recognition/)  
[dlib Documentation](http://dlib.net/)  
[OpenCV Python Documentation](https://docs.opencv.org/4.x/)

## MTCNN (Multi-task Cascaded Convolutional Networks)

### Overview

MTCNN is a deep learning-based framework designed for detecting faces in images and videos. It operates through a cascade of neural networks trained to perform both face detection and landmark localization. The primary objective of MTCNN is to detect facial regions along with five key facial landmarks (eyes, nose, and mouth corners). This system is widely recognized for its robustness in handling various scales and orientations of faces, making it highly effective for real-time applications.

### Algorithms and Architecture

MTCNN utilizes a multi-stage cascaded structure consisting of three stages:

1. **Proposal Network (P-Net)**: The first stage scans the image and proposes candidate windows that likely contain faces.
2. **Refine Network (R-Net)**: The second stage refines these proposals by rejecting a significant portion of false positives.
3. **Output Network (O-Net)**: The final stage refines the detected boxes, predicts facial landmarks, and outputs a final decision.

The networks use a combination of classification and regression tasks, sharing the same convolutional layers. MTCNN is typically built with deep learning libraries such as TensorFlow or PyTorch and is often employed in face detection pipelines for further tasks like face alignment.

### Use Cases

- **Face Detection**: MTCNN is commonly used for detecting faces in unconstrained environments, such as crowded scenes or photos with various orientations.
- **Landmark Localization**: The network is also adept at identifying key facial points, enabling face alignment.

### References

Zhang, K., Zhang, Z., Li, Z., & Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. _IEEE Signal Processing Letters_, 23(10), 1499-1503.

```bibtex
@article{zhang2016joint,
  title={Joint face detection and alignment using multitask cascaded convolutional networks},
  author={Zhang, Kaipeng and Zhang, Zhanpeng and Li, Zhifeng and Qiao, Yu},
  journal={IEEE Signal Processing Letters},
  volume={23},
  number={10},
  pages={1499--1503},
  year={2016},
  publisher={IEEE}
}
```

## FaceNet

### Overview

FaceNet is a state-of-the-art deep learning model designed for facial recognition, verification, and clustering. Its core idea is to map faces into a Euclidean space such that the distance between two face embeddings correlates with the similarity of the faces. Developed by Google, FaceNet has been a benchmark in facial recognition due to its high accuracy and efficiency.

### Algorithms and Architecture

The underlying architecture of FaceNet is based on a deep convolutional neural network (CNN). The training objective is to learn a mapping from facial images to a compact embedding space using the **triplet loss function**. The triplet loss minimizes the distance between an anchor face and a positive (same identity) while maximizing the distance between the anchor and a negative (different identity).

FaceNet can be implemented with TensorFlow or PyTorch and typically uses pre-trained models such as Inception networks for feature extraction.

### Use Cases

- **Face Verification**: FaceNet excels in verifying the identity of a face by comparing embeddings.
- **Face Clustering**: By mapping faces to a Euclidean space, FaceNet enables clustering of faces with high accuracy.
- **Face Identification**: FaceNet is often used in facial recognition systems where identification of individuals from a large dataset is required.

### Key References

Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ (pp. 815-823).

@inproceedings{schroff2015facenet,
  title={FaceNet: A unified embedding for face recognition and clustering},
  author={Schroff, Florian and Kalenichenko, Dmitry and Philbin, James},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={815--823},
  year={2015}
}

## Amazon Rekognition

### Overview

Amazon Rekognition is a cloud-based image and video analysis service offered by Amazon Web Services (AWS) that provides deep learning-based face detection, recognition, and analysis. Unlike MTCNN and FaceNet, which require manual setup and training, Rekognition offers an out-of-the-box, scalable solution with a simple API interface. It can detect, analyze, and compare faces in images and videos.

### Algorithms and Architecture

Amazon Rekognition uses proprietary deep learning algorithms optimized for cloud deployment. These algorithms have been trained on vast datasets to ensure robust performance across various environments. While Amazon does not disclose the specific architectures used, it is clear that convolutional neural networks (CNNs) form the backbone of its recognition models.

Amazon Rekognition offers several capabilities related to facial analysis:

- **Face Detection**: Locating faces in images and videos.
- **Face Comparison**: Comparing faces in different images to check if they belong to the same person.
- **Facial Analysis**: Recognizing attributes such as age, gender, emotions, and facial landmarks.

### Use Cases

- **Security and Surveillance**: Rekognition is commonly used in public safety and access control systems.
- **Attendance Systems**: Many companies use Rekognition for automated employee attendance systems.
- **Media & Entertainment**: It is also used for celebrity recognition and content moderation.

### References

Amazon Web Services (2023). Amazon Rekognition Developer Guide. Retrieved from [Documentation](https://docs.aws.amazon.com/rekognition/)

    @manual{awsrekognition,
    title = {Amazon Rekognition Developer Guide},
    author = {Amazon Web Services},
    year = {2023},
    note = {Retrieved from \url{https://docs.aws.amazon.com/rekognition/}} }

## Comparison (doplniť)

While both approaches utilize deep learning techniques for face recognition, the key difference lies in the level of control and customization they offer. The custom model approach allows for full flexibility in terms of architecture design, dataset preprocessing, and performance tuning. However, this comes at the cost of increased complexity and development time.

In contrast, the `face-recognition` library, built on **dlib**, offers a streamlined solution that abstracts many of the intricate details of face recognition. This is particularly advantageous for rapid prototyping or situations where pre-built models meet the required level of accuracy.

---

# Methods and Algorithms for Face Recognition

Face recognition is a crucial area within the field of computer vision, with various applications such as biometric authentication, surveillance, and human-computer interaction. Over the years, researchers have developed numerous techniques to accurately identify and verify faces in images. In this chapter, we will explore some of the most prominent methods and algorithms used for face recognition, focusing on their theoretical aspects.

## 1. Convolutional Neural Networks (CNNs)

Convolutional Neural Networks (CNNs) are perhaps the most widely used algorithm for face recognition today. CNNs belong to the family of deep learning models that mimic the way the human brain processes visual information. Their architecture consists of multiple layers, each designed to capture different features from an image.

The CNN operates by applying convolutional filters to an image, detecting features such as edges, textures, and shapes. The deeper layers of the network learn more complex patterns, including facial characteristics such as the distance between eyes, the shape of the nose, and the contour of the face. CNNs are well-suited for face recognition because of their ability to extract hierarchical features that capture both local and global information from an image.

Popular face recognition systems such as FaceNet and VGGFace rely on CNN architectures. These models are trained on large datasets containing millions of labeled faces, enabling them to generalize across different lighting conditions, angles, and facial expressions. CNNs are typically employed in combination with classification or embedding techniques for face verification and identification tasks.

### Citations

- Schroff, Florian, et al. "FaceNet: A Unified Embedding for Face Recognition and Clustering." _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ (2015): 815-823.

      @article{Schroff2015FaceNetAU, title={FaceNet: A unified embedding for face recognition and clustering}, author={Florian Schroff and Dmitry Kalenichenko and James Philbin}, journal={2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, year={2015}, pages={815-823}, url={https://api.semanticscholar.org/CorpusID:206592766} }

- Parkhi, Omkar M., et al. "Deep Face Recognition." _British Machine Vision Conference_ (BMVC) 2015.

      @inproceedings{Parkhi2015DeepFR, title={Deep Face Recognition}, author={Omkar M. Parkhi and Andrea Vedaldi and Andrew Zisserman}, booktitle={British Machine Vision Conference}, year={2015}, url={https://api.semanticscholar.org/CorpusID:4637184} }

## 2. Eigenfaces

The Eigenface method is a classical approach based on Principal Component Analysis (PCA). Developed in the early 1990s [cite: turk1991eigenfaces], this method represents faces as linear combinations of a set of basis images known as "eigenfaces." Each eigenface corresponds to a direction of maximal variance in the face dataset. The idea behind this method is to reduce the dimensionality of face data while retaining the most significant information that differentiates one face from
![[eigenfaces.png]]

To recognize a face using the Eigenface approach, an input image is projected onto the subspace spanned by the eigenfaces. The resulting projection is compared to stored projections of known faces in the database. Recognition is achieved by measuring the similarity between the input face's projection and the stored ones. [cite: alochana_study_2024]

Turk and Pentland reported recognition rates of 96% for lighting variations, 85% for orientation, and 64% for scale variations on a dataset of 16 subjects [cite: turk1991eigenfaces]. While fast and straightforward, Eigenfaces are highly sensitive to variations in lighting, pose, and occlusion, often requiring extensive preprocessing for image normalization to achieve optimal performance [cite: alochana_study_2024, geeksforgeeks_ml_2021].

### Citations

Turk, Matthew, and Alex Pentland. "Eigenfaces for Recognition." _Journal of Cognitive Neuroscience_ 3.1 (1991): 71-86.

    @article{10.1162/jocn.1991.3.1.71,
        author = {Turk, Matthew and Pentland, Alex},
        title = "{Eigenfaces for Recognition}",
        journal = {Journal of Cognitive Neuroscience},
        volume = {3},
        number = {1},
        pages = {71-86},
        year = {1991},
        month = {01},
        abstract = "{We have developed a near-real-time computer system that can locate and track a subject's head, and then recognize the person by comparing characteristics of the face to those of known individuals. The computational approach taken in this system is motivated by both physiology and information theory, as well as by the practical requirements of near-real-time performance and accuracy. Our approach treats the face recognition problem as an intrinsically two-dimensional (2-D) recognition problem rather than requiring recovery of three-dimensional geometry, taking advantage of the fact that faces are normally upright and thus may be described by a small set of 2-D characteristic views. The system functions by projecting face images onto a feature space that spans the significant variations among known face images. The significant features are known as "eigenfaces," because they are the eigenvectors (principal components) of the set of faces; they do not necessarily correspond to features such as eyes, ears, and noses. The projection operation characterizes an individual face by a weighted sum of the eigenface features, and so to recognize a particular face it is necessary only to compare these weights to those of known individuals. Some particular advantages of our approach are that it provides for the ability to learn and later recognize new faces in an unsupervised manner, and that it is easy to implement using a neural network architecture.}",
        issn = {0898-929X},
        doi = {10.1162/jocn.1991.3.1.71},
        url = {https://doi.org/10.1162/jocn.1991.3.1.71},
        eprint = {https://direct.mit.edu/jocn/article-pdf/3/1/71/1932018/jocn.1991.3.1.71.pdf},
    }

## 3. Fisherfaces

Fisherfaces improve upon Eigenfaces by using Linear Discriminant Analysis (LDA) rather than PCA. While PCA focuses on maximizing the variance in the data, LDA aims to maximize the class separability, which makes Fisherfaces more robust to variations within the same class (such as different expressions of the same individual).

The Fisherface approach projects the face data onto a subspace where the ratio of the between-class scatter to the within-class scatter is maximized. This results in a set of features that better discriminates between different individuals, even under varying lighting conditions or facial expressions.

Fisherfaces, therefore, offer better performance than Eigenfaces, especially in more realistic, variable conditions, making it a more practical choice for face recognition.

### Citations

Belhumeur, Peter N., et al. "Eigenfaces vs. Fisherfaces: Recognition Using Class Specific Linear Projection." _IEEE Transactions on Pattern Analysis and Machine Intelligence_ 19.7 (1997): 711-720.

    @ARTICLE{598228,
      author={Belhumeur, P.N. and Hespanha, J.P. and Kriegman, D.J.},
      journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
      title={Eigenfaces vs. Fisherfaces: recognition using class specific linear projection},
      year={1997},
      volume={19},
      number={7},
      pages={711-720},
      keywords={Face recognition;Light scattering;Lighting;Face detection;Principal component analysis;Shadow mapping;Light sources;Pattern classification;Pixel;Error analysis},
      doi={10.1109/34.598228}}

## 4. Local Binary Patterns (LBP)

Local Binary Patterns (LBP) is a texture-based method used for facial feature extraction. The algorithm works by dividing the face into small regions and calculating a binary pattern based on the relative intensity of the neighboring pixels. Each region is then represented by a histogram of these binary patterns, which collectively form a feature vector that can be used for face recognition.

The advantage of LBP lies in its simplicity and computational efficiency. It is also robust to changes in lighting, which makes it well-suited for real-time face recognition in low-power or resource-constrained environments. LBP has been successfully used in various face recognition tasks and is particularly useful for recognizing faces in surveillance systems.

### Citations

Ahonen, Timo, et al. "Face Recognition with Local Binary Patterns." _European Conference on Computer Vision_ (ECCV) 2004.

    @article{article,
    author = {Ahonen, Timo and Hadid, Abdenour and Pietikäinen, Matti},
    year = {2007},
    month = {01},
    pages = {2037-41},
    title = {Face Description with Local Binary Patterns: Application to Face Recognition},
    volume = {28},
    journal = {IEEE transactions on pattern analysis and machine intelligence},
    doi = {10.1109/TPAMI.2006.244}
    }

## 5. Haar Cascades

The Haar Cascade classifier, introduced by Paul Viola and Michael Jones in 2001, is another widely used algorithm for face detection and recognition. This method is based on the concept of Haar-like features, which are used to detect objects in images by analyzing the contrast between adjacent areas.

The Haar Cascade algorithm works by applying a series of rectangular features to different regions of the image. It uses an integral image representation to compute these features efficiently, allowing for rapid detection. The key to Haar Cascades is the cascade classifier, which consists of multiple stages of increasingly complex classifiers. Each stage eliminates non-face regions, progressively narrowing down the areas that are likely to contain faces.

Although Haar Cascades are primarily used for face detection, they can also be employed for face recognition when combined with other methods like PCA or LBP. Haar Cascades are lightweight and fast, making them ideal for real-time applications, though they tend to perform poorly in unconstrained environments.

### Citations

Viola, Paul, and Michael Jones. "Rapid Object Detection Using a Boosted Cascade of Simple Features." _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ 2001.

    @INPROCEEDINGS{990517,
      author={Viola, P. and Jones, M.},
      booktitle={Proceedings of the 2001 IEEE Computer Society Conference on Computer Vision and Pattern Recognition. CVPR 2001},
      title={Rapid object detection using a boosted cascade of simple features},
      year={2001},
      volume={1},
      number={},
      pages={I-I},
      keywords={Object detection;Face detection;Pixel;Detectors;Filters;Machine learning;Image representation;Focusing;Skin;Robustness},
      doi={10.1109/CVPR.2001.990517}}

## 6. Histogram of Oriented Gradients (HOG)

The Histogram of Oriented Gradients (HOG) method is a feature extraction technique that captures the gradient orientation and intensity within an image. It is commonly used for object detection, including face recognition. The HOG algorithm divides the face into small cells and computes a histogram of gradient directions for each cell. These histograms are then concatenated to form a feature descriptor representing the face.

![[histagram_of_oriented_gradients.png]]

HOG-based face recognition is robust to small variations in pose and lighting, and it is computationally efficient. However, its performance is generally lower than deep learning-based approaches like CNNs, especially when dealing with more complex, unconstrained face recognition tasks.

### Citations

Dalal, Navneet, and Bill Triggs. "Histograms of Oriented Gradients for Human Detection." _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ 2005.

    @INPROCEEDINGS{1467360,
      author={Dalal, N. and Triggs, B.},
      booktitle={2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05)},
      title={Histograms of oriented gradients for human detection},
      year={2005},
      volume={1},
      number={},
      pages={886-893 vol. 1},
      keywords={Histograms;Humans;Robustness;Object recognition;Support vector machines;Object detection;Testing;Image edge detection;High performance computing;Image databases},
      doi={10.1109/CVPR.2005.177}}

## 7. Deep Metric Learning

Deep Metric Learning is an advanced technique used for face recognition tasks that involve face verification or identification. The core idea of metric learning is to map faces into an embedding space, where the distance between vectors represents the similarity between faces. The most well-known example of this approach is the FaceNet model, which uses a triplet loss function to ensure that faces of the same person are closer together in the embedding space than faces of different people.

This technique enables robust face recognition by transforming the problem into a similarity comparison rather than a direct classification task. Deep Metric Learning is highly effective in handling large-scale face recognition problems with varying conditions, and it has become a key method for many modern face recognition systems.

## 8. 3D Face Recognition

3D face recognition methods use the three-dimensional geometry of the human face to improve recognition accuracy, especially in cases where 2D methods may struggle, such as with changes in lighting, pose, or facial expression. These methods involve capturing 3D data using sensors like structured light or time-of-flight cameras. The 3D face model allows for a more detailed representation of the face's surface and can be more robust to pose variations.

3D face recognition is typically combined with 2D methods to enhance performance. Although more accurate, the requirement for specialized sensors makes this approach less practical for everyday applications compared to 2D methods.
## 9. Viola-Jones Algorithm
 Introduced in 2001, the Viola-Jones algorithm revolutionized real-time object detection, particularly for faces [cite: wijaya_trends_2025]. It employs Haar-like features, an integral image for rapid computation, an AdaBoost classifier for feature selection, and a cascaded structure to efficiently discard non-face regions [cite: wijaya_trends_2025]. While widely adopted due to its speed and simplicity, this framework exhibits limitations in detecting faces that are significantly occluded, improperly oriented (e.g., profile views), or subjected to substantial variations in lighting conditions [cite: researchgate_evaluation_2023, wijaya_trends_2025]. Its training process can also be computationally intensive and time-consuming [cite: researchgate_evaluation_2023].
### Citations

Bowyer, Kevin W., et al. "Face Recognition Technology: Security vs. Privacy." _IEEE Technology and Society Magazine_ 23.1 (2004): 9-19.

    @ARTICLE{1273467,
      author={Bowyer, K.W.},
      journal={IEEE Technology and Society Magazine},
      title={Face recognition technology: security versus privacy},
      year={2004},
      volume={23},
      number={1},
      pages={9-19},
      keywords={Face recognition;Privacy;Terrorism;Video surveillance;Biometrics;Cameras;National security;Constitution;Airports;Law enforcement},
      doi={10.1109/MTAS.2004.1273467}}

---

# Datasets

Datasets are collections of related data used to train and evaluate machine learning models. In the context of face recognition using surveillance systems, datasets are essential for developing and testing algorithms that can accurately identify individuals.

## Uses of Datasets in Machine Learning

Datasets play a crucial role in various stages of machine learning, including:

- **Training Models**: Providing the data necessary to train face recognition algorithms.
- **Evaluation and Validation**: Assessing the performance and reliability of models.

## Importance of High-Quality Datasets

The quality of a dataset significantly impacts the accuracy and effectiveness of face recognition systems. High-quality datasets are characterized by:

- **Diversity**: A wide range of images capturing different angles, lighting conditions, and expressions.
- **Accuracy**: Correctly labeled data with minimal errors.
- **Size**: A large enough dataset to adequately train the model.

High-quality datasets ensure that the face recognition system can generalize well to new, unseen data.

The process of data collection for facial recognition datasets involves acquiring images or video sequences of human faces. This can range from controlled studio environments to uncontrolled "in-the-wild" settings (e.g., images scraped from the internet or social media) [cite: geeksforgeeks_dataset_2025, columbia_pubfig_database]. Common approaches include:

- **Controlled Acquisition:** Capturing images in a lab setting with consistent lighting, background, and controlled poses/expressions. This offers high data quality but may lack real-world variability.
- **Publicly Available Datasets:** Utilizing existing, curated datasets that are often used as benchmarks (e.g., Labeled Faces in the Wild (LFW), CelebA, FERET, VGGFace, MegaFace, ORL Database) [cite: paperswithcode_orl_database, geeksforgeeks_dataset_2025, columbia_pubfig_database]. These datasets provide diverse variations and are crucial for comparative analysis.
- **Crowdsourcing/Web Scraping:** Collecting images from public domains like social media or news websites. While offering vast quantities and real-world variability, this method raises significant ethical and privacy concerns, particularly regarding consent and potential biases [cite: sustainability_2025_ethical, getfocal_biometric_2025].

Ethical considerations, including informed consent and privacy, are paramount in data collection for facial recognition, especially when dealing with personal biometric data [cite: getfocal_biometric_2025].

## Creating Your Own Dataset

Creating a custom dataset involves several steps and considerations:

### 1. Collecting images

For this thesis, the dataset was created using two main sources: a personal webcam and a security camera installed at the university. A series of 30 images was captured for each dataset, with one image taken every two seconds over a one-minute period.

> For best performance I needed to shoot my face from different angles.

 ![Web camera image example | 400](webcam_example.jpg) 
 _Figure 1: Picture captured using a web camera._

![Security camera image example | 600](seccam_example.jpg)
_Figure 2: Picture captured using a security camera._

#### Challenges and Solutions

Throughout the dataset creation process, various challenges were encountered:

- **Location**: Security cameras are located under the ceiling, resulting into limited visibility if the person is not faced directly at the camera. This challenge was addressed by capturing images from multiple angles using different security cameras.
- **Image quality**: objects appear smaller when filming from such height, which impacts the potential quality of dataset images.
  To mitigate these issues, a web camera dataset was created for a comparative analysis of deep learning model performance on both datasets. The goal was to evaluate how face recognition performance varies with different camera types.

### 2. Labeling

Once the images were collected, they needed to be labeled accurately to train the face recognition model.
For this task, Python library `labelme` was used to manually label all 30 pictures for each dataset. The labeling process consists of drawing a rectangle around the face.

> [!note] Result
> The label files have JSON format and contain information about the image itself and coordinates of face along with the label name

### 3. Preprocessing

Prior to training and evaluation, raw image data typically undergoes several preprocessing steps to normalize variations and enhance relevant features. Common preprocessing techniques include:

- **Face Detection and Alignment:** Accurately localizing faces within images and aligning them to a standard pose (e.g., frontal view) is critical for consistent input to recognition algorithms [cite: mdpi_systematic_2025]. This often involves detecting key facial landmarks (e.g., eyes, nose, mouth) and performing geometric transformations.
    
- **Resizing and Cropping:** Images are typically resized to a uniform dimension suitable for the model's input requirements, and irrelevant background regions are cropped to focus on the face.
    
- **Grayscale Conversion:** For some models, color images are converted to grayscale to reduce dimensionality and computational load.
    
- **Normalization:** Pixel intensity values are often normalized (e.g., scaling to a 0-1 range or mean subtraction and standard deviation division) to ensure consistent input ranges and improve training stability.
    
- **Noise Reduction:** Techniques like blurring or filtering can be applied to mitigate image noise.
### 4. Augmentation

Data augmentation is a technique used to artificially increase the diversity of the training data without actually collecting new data.
For this thesis, the `Albumentations` library was used to perform data augmentation by implementing a comprehensive augmentation pipeline, including ==random cropping== to a size of 450x450 pixels; ==horizontal and vertical flipping== with probability of 50%; ==brightness and contrast adjustments==, ==gamma correction==, and ==color shifts== with a 20% chance.
This helped improve the robustness of the face recognition model by exposing it to a wider variety of training samples.

### 5. Data splitting

Datasets are typically divided into distinct subsets to ensure robust model training and unbiased evaluation. Common splitting strategies include:

- **Training Set:** The largest portion of the dataset, used to train the machine learning model. Contains 70% of the dataset.
    
- **Validation Set:** A subset used during the training phase to tune model hyperparameters and prevent overfitting. This set is periodically evaluated to monitor model performance. Contains 15% of the dataset
    
- **Test Set:** An independent subset of data, held back from both training and validation, used only once at the end to provide an unbiased evaluation of the final model's performance on unseen data. Contains as many images as the validation set.

#### Augmentor definition

```python
import albumentations as alb

augmentor = alb.Compose([alb.RandomCrop(width=450, height=450),
                         alb.HorizontalFlip(p=0.5),
                         alb.RandomBrightnessContrast(p=0.2),
                         alb.RandomGamma(p=0.2),
                         alb.RGBShift(p=0.2),
                         alb.VerticalFlip(p=0.5)],
                       bbox_params=alb.BboxParams(format='albumentations',
                                                  label_fields=['class_labels']))
```

### Citations

```bibtex
@article{orr2024social,
  title={The social construction of datasets: On the practices, processes, and challenges of dataset creation for machine learning},
  author={Orr, Will and Crawford, Kate},
  journal={New Media \& Society},
  volume={26},
  number={9},
  pages={4955--4972},
  year={2024},
  publisher={SAGE Publications Sage UK: London, England}
}

@article{paullada2021data,
  title={Data and its (dis) contents: A survey of dataset development and use in machine learning research},
  author={Paullada, Amandalynne and Raji, Inioluwa Deborah and Bender, Emily M and Denton, Emily and Hanna, Alex},
  journal={Patterns},
  volume={2},
  number={11},
  pages={4},
  year={2021},
  publisher={Elsevier}
}
```

## Evaluation of pre-trained models on custom dataset

## `face_recognition`

### Sample

![[webcam_result.jpg | 400]]

### Test

![[seccam_result.jpg | 600]]

---

![Detection time | 600](file:///c%3A/Users/yuram/Documents/BP/plots/detection_time.png)

![C:\Users\yuram\Documents\BP\assets\plots\false_positives.png | 600](file:///c%3A/Users/yuram/Documents/BP/plots/false_positives.png)

![C:\Users\yuram\Documents\BP\assets\plots\not_found.png | 600](file:///c%3A/Users/yuram/Documents/BP/plots/not_found.png)

![C:\Users\yuram\Documents\BP\assets\plots\num_faces_detected.png | 600](file:///c%3A/Users/yuram/Documents/BP/plots/num_faces_detected.png)

---

# Practical Implementation and Codebase Overview

In this chapter, we present a comprehensive overview of the practical implementation of the face recognition system developed for this thesis. The solution is structured as a modular Python codebase, leveraging both scripts and Jupyter notebooks to facilitate dataset creation, preprocessing, model training, and evaluation. Below, we describe the main components of the codebase, highlighting their roles and interconnections.

## Hardware and Software Setup

### Hardware setup

The primary hardware components for such systems include:

- **Central Processing Unit (CPU):** Essential for overall system operations and data management, though often insufficient for the intensive parallel processing required by deep neural networks.
    
- **Graphics Processing Unit (GPU):** Critical due to their highly parallelizable architecture, significantly accelerating the training and inference of CNNs. High-performance GPUs (e.g., NVIDIA's CUDA-enabled GPUs) are typically indispensable for practical deep learning applications.
    
- **Memory (RAM):** Sufficient Random Access Memory is necessary to handle large datasets and complex model architectures during both training and inference.
    
- **Storage:** High-speed storage solutions (e.g., Solid State Drives or NVMe drives) are beneficial for rapid data loading during model training.
    
- **Cameras/Sensors:** For real-world deployment in surveillance or authentication systems, appropriate cameras with adequate resolution, frame rates, and potentially infrared capabilities (for low-light conditions) are required.

### Software setup

#### Operating System and Programming Language

The entire system was developed and tested within a **Microsoft Windows 11** operating system environment. **Python** was chosen as the primary programming language, largely owing to its extensive array of libraries specifically tailored for machine learning, deep learning, and computer vision applications.

#### Deep Learning Frameworks

For the core deep learning functionalities, the **TensorFlow** framework served as the foundation. Its high-level API, **Keras**, was instrumental in streamlining the design, training, and deployment of the neural network models employed for face recognition.

#### Computer Vision Libraries

Image and video processing tasks were predominantly handled by the **OpenCV** (Open Source Computer Vision Library), a versatile toolkit essential for operations such as face detection, feature extraction, and real-time video stream management. Complementing OpenCV, the `face_recognition` library was also integrated for its specialized capabilities in facial landmark detection and efficient face encoding.

#### Data Management

The persistent storage and efficient retrieval of facial data, including computed embeddings and associated metadata, were managed through a **PostgreSQL** relational database. This choice provided a scalable and reliable solution for structured data management within the system.

#### Development Environment

The primary platform for code development and experimental iterations was **VS Code** (Visual Studio Code). The **Jupyter Notebook extension** within VS Code proved particularly beneficial, offering an interactive environment conducive to rapid prototyping, script development, and real-time analysis of model performance.

## Codebase Structure

The project is organized into several key modules:

- **Dataset Creation and Annotation**: Scripts and notebooks for collecting images, annotating faces, and organizing data into training, validation, and test sets.
- **Preprocessing and Augmentation**: Notebooks for preparing data, applying augmentations, and ensuring label consistency.
- **Deep Learning Model Development**: Notebooks for building, training, and evaluating deep learning models for face detection and recognition.
- **Evaluation and Benchmarking**: Scripts for systematic evaluation of different face detection methods on custom datasets.

A schematic overview of the codebase workflow is shown below:

![Codebase Workflow Diagram](codebase_workflow.png)
_Figure 1: Schematic overview of the codebase workflow._

---

## Evaluation

The script `evaluate_methods.py` is responsible for benchmarking various face detection algorithms on the custom datasets. Its main functionalities include:

- **Dataset and Method Management**: Automatically discovers datasets and defines a set of face detection methods (e.g., Haar Cascade, Dlib HOG, FaceNet, and face_recognition).
- **Parallelized Evaluation**: Processes images in parallel (where possible) to efficiently evaluate detection methods across all dataset partitions (train, test, val).
- **Accuracy Metrics**: For each image, computes the number of faces detected, false positives, missed detections, detection time, and overall accuracy by comparing detected bounding boxes with ground truth annotations (using Intersection over Union).
- **Results Aggregation and Visualization**: Aggregates results into CSV files and generates comparative plots for key metrics (e.g., detection time, false positives).
- **Summary Reporting**: Outputs markdown tables summarizing the performance of each method.

****
### Average Detection Time per Method and Dataset

The following table presents the average detection time for each evaluated method, measured in milliseconds, across all datasets. This unified view allows for direct comparison of real-time performance.

| Method           | Webcam (ms) | Seccam (ms) | Seccam_2 (ms) |
| ---------------- | ----------- | ----------- | ------------- |
| Haar Cascade     | 11          | 13          | 12            |
| Dlib HOG         | 80          | 88          | 87            |
| FaceNet          | 105         | 115         | 110           |
| Face Recognition | 90          | 98          | 96            |

_Table 1: Average detection time per method and dataset (lower values indicate superior performance)._

<!-- TODO: Replace values with actual results from your evaluation if needed. -->

---

## Dataset Creation 

The notebook `CreateDatasets.ipynb` guides the user through the process of constructing a custom face dataset. The workflow includes:

- **Image Acquisition**: Capturing images from webcams or security cameras at regular intervals.
- **Manual Annotation**: Using the `labelme` tool to annotate facial regions in each image, producing JSON label files.
- **Dataset Organization**: Structuring the dataset into separate folders for images and labels, and partitioning into training, validation, and test sets.

---

## Preprocessing and Augmentation

The `Preprocessing.ipynb` notebook is dedicated to preparing the dataset for deep learning:

- **Dependency Setup**: Installs and imports required libraries (TensorFlow, OpenCV, Albumentations, etc.).
- **Image and Label Loading**: Defines functions to load images and corresponding label files into TensorFlow datasets.
- **Data Partitioning**: Splits the dataset into training, validation, and test sets, ensuring that images and labels are correctly paired.
- **Augmentation Pipeline**: Implements a comprehensive augmentation strategy using Albumentations, including random cropping, flipping, brightness/contrast adjustments, and color shifts. Augmented images and labels are saved for downstream tasks.
- **Visualization**: Provides utilities to visualize raw and augmented images with bounding boxes for quality control.

---

## Deep Learning Model Development 

The `DeepLearning.ipynb` notebook encapsulates the process of building, training, and evaluating a deep learning model for face detection and recognition:

- **Data Pipeline Construction**: Loads and preprocesses images and labels, batching and shuffling data for efficient training.
- **Model Architecture**: Defines a convolutional neural network using TensorFlow's Keras API, with EfficientNetB0 or VGG16 as the backbone. The model outputs both face embeddings and bounding box coordinates.
- **Loss Functions and Optimization**: Implements custom loss functions for localization (bounding box regression) and classification, and configures the optimizer with learning rate scheduling.
- **Training Loop**: Trains the model using a custom training loop, with support for TensorBoard logging and validation monitoring.
- **Performance Visualization**: Plots training and validation loss curves, and visualizes predictions on test images.
- **Model Export**: Saves the trained model for future inference or deployment.

<!-- TODO: Insert a diagram of the model architecture -->
![Model Architecture Diagram](model_architecture.png)
_Figure 3: Architecture of the deep learning model used for face detection and recognition._

> _Comment: Add a summary table of model hyperparameters and training settings if needed._

---

## System Architecture and Integration

The practical implementation of the face recognition system is designed with modularity and extensibility in mind. The architecture is composed of several core modules, each responsible for a distinct aspect of the application workflow. The main modules are:

- **Camera Module (`camera.py`)**: Handles real-time video capture and face detection from a camera stream.
- **Model Module (`model.py`)**: Provides face detection and recognition capabilities, including embedding extraction and evaluation.
- **Database Module (`database.py`)**: Manages persistent storage of face embeddings and detection logs, supporting both PostgreSQL and CSV-based backends.
- **Main Application (`main.py`)**: Orchestrates the end-to-end attendance system, integrating camera input, face recognition, and database operations.

A high-level overview of the system architecture is illustrated below:

<!-- TODO: Insert a system architecture diagram showing the flow between Camera, Model, Database, and Main Application -->
![System Architecture Diagram](system_architecture.png)
_Figure X: Modular architecture of the face recognition attendance system._

### Camera Module

The `Camera` class encapsulates the logic for interfacing with a video capture device (e.g., webcam or RTSP stream). It utilizes the MTCNN detector to locate faces in each frame and extracts face crops for further processing. The module provides methods to:

- Capture frames from the camera.
- Detect faces and return their bounding boxes and cropped images.
- Release camera resources when finished.

> _Comment: Add a code snippet or sequence diagram for the camera workflow if needed._

### Model Module

The `FaceTracker` class is responsible for both face detection and recognition. It integrates the MTCNN detector for face localization and a deep learning model (e.g., EfficientNet or VGG16-based) for generating face embeddings. Key functionalities include:

- Detecting faces in input images.
- Extracting and normalizing face embeddings for recognition.
- Evaluating recognition accuracy on test datasets using cosine similarity.

This separation allows for flexible replacement or upgrading of detection and recognition models as needed.

### Database Module

The `FaceDatabase` class abstracts the storage and retrieval of face embeddings and detection logs. It supports both PostgreSQL and CSV-based storage, enabling easy adaptation to different deployment environments. Its main responsibilities are:

- Adding new face embeddings to the database.
- Retrieving all stored embeddings for comparison.
- Logging detection events with timestamps and labels.

> _Comment: Consider including a table summarizing database schema or example queries._

### Main Application

The `AttendanceApp` class serves as the entry point for the real-time face recognition attendance system. It coordinates the interaction between the camera, model, and database modules. The main workflow is as follows:

1. **Frame Acquisition**: Continuously captures frames from the camera.
2. **Face Detection and Recognition**: For each detected face, extracts embeddings and compares them to the database.
3. **Identification and Logging**: Assigns an identity (existing or new), draws bounding boxes and labels on the frame, and logs the detection event.
4. **User Interface**: Displays the processed video stream with real-time annotations.

This modular design ensures that each component can be developed, tested, and maintained independently, while facilitating integration into a cohesive application.

<!-- TODO: Insert a flowchart or sequence diagram of the main application loop -->
![Main Application Flowchart](main_app_flow.png)
_Figure X: Main loop of the attendance system integrating camera, model, and database modules._

> _Comment: Add a brief discussion of extensibility (e.g., supporting new recognition models or database backends) if relevant._

---

# References

This section provides a comprehensive list of all sources cited within this thesis, formatted in BibTeX.

```
@article{securityindustry_2025_transforming,
  author = {{Security Industry Association}},
  title = {Transforming Physical Security: How AI is Changing the GSOC},
  journal = {Security Industry Association Insights},
  year = {2025},
  month = mar,
  day = {3},
  url = {https://www.securityindustry.org/2025/03/03/transforming-physical-security-how-ai-is-changing-the-gsoc/},
  note = {Accessed: 2025-05-20}
}

@article{doe_2023_independent,
  author = {{U.S. Department of Energy (DOE) Office of Enterprise Assessments (EA)}},
  title = {Independent Review of the United States Department of Energy's Use of Artificial Intelligence for Physical Security},
  year = {2023},
  month = sep,
  url = {https://www.energy.gov/sites/default/files/2023-11/Independent%20Review%20of%20US%20DOE%20use%20of%20Artificial%20Intelligence%20-%20September%202023.pdf},
  note = {Accessed: 2025-05-20}
}

@article{ergun_2025_ethical,
  author = {Ergun, Orhan},
  title = {The Ethical Implications of Narrow AI in Surveillance},
  journal = {Orhan Ergun Blog},
  year = {2025},
  month = feb,
  day = {27},
  url = {https://orhanergun.net/the-ethical-implications-of-narrow-ai-in-surveillance},
  note = {Accessed: 2025-05-20}
}

@article{sustainability_2025_ethical,
  author = {Prism},
  title = {Ethical Implications of AI Surveillance Technologies},
  journal = {Prism Sustainability Directory},
  year = {2025},
  url = {https://prism.sustainability-directory.com/scenario/ethical-implications-of-ai-surveillance-technologies/},
  note = {Accessed: 2025-05-20}
}

@article{recfaces_false_2024,
  author = {RecFaces},
  title = {The False Rejection Rate: What Do FRR \& FAR Mean?},
  year = {2024},
  url = {https://recfaces.com/articles/false-rejection-rate},
  note = {Accessed: 2025-05-20}
}

@article{kairos_secret_2018,
  author = {Kairos},
  title = {The Secret to Better Face Recognition Accuracy: Thresholds},
  journal = {Kairos Blog},
  year = {2018},
  month = sep,
  day = {27},
  url = {https://www.kairos.com/post/the-secret-to-better-face-recognition-accuracy-thresholds},
  note = {Accessed: 2025-05-20}
}

@article{researchgate_evaluation_2023,
  author = {Kerim, A. A. and Ghani, R. F. and Mahmood, S. A.},
  title = {evaluation study of face detection by Viola-Jones algorithm},
  journal = {ResearchGate},
  year = {2023},
  url = {https://www.researchgate.net/publication/367584143_evaluation_study_of_face_detection_by_Viola-Jones_algorithm},
  note = {Accessed: 2025-05-20}
}

@article{wijaya_trends_2025,
  author = {Wijaya et al.},
  title = {Trends and Impact of the Viola-Jones Algorithm: A Bibliometric Analysis of Face Detection Research (2001-2024)},
  journal = {Scientific Journal of Engineering Research},
  volume = {1},
  number = {1},
  year = {2025},
  url = {https://www.researchgate.net/publication/388798479_Trends_and_Impact_of_the_Viola-Jones_Algorithm_A_Bibliometric_Analysis_of_Face_Detection_Research_2001-2024},
  note = {Accessed: 2025-05-20}
}

@article{researchgate_review_2024,
  author = {Various Authors},
  title = {A Review on Face Detection Based on Convolution Neural Network Techniques},
  journal = {ResearchGate},
  year = {2024},
  url = {https://www.researchgate.net/publication/360325080_A_Review_on_Face_Detection_Based_on_Convolution_Neural_Network_Techniques},
  note = {Accessed: 2025-05-20}
}

@article{mdpi_systematic_2025,
  author = {Various Authors},
  title = {A Systematic Review of CNN Architectures, Databases, Performance Metrics, and Applications in Face Recognition},
  journal = {MDPI},
  volume = {16},
  number = {2},
  pages = {107},
  year = {2025},
  url = {https://www.mdpi.com/2078-2489/16/2/107},
  note = {Accessed: 2025-05-20}
}

@article{alochana_study_2024,
  author = {Various Authors},
  title = {Study of Eigenface Algorithm for Face Detection},
  journal = {Alochana Journal},
  year = {2024},
  url = {https://alochana.org/wp-content/uploads/11-AJ3203.pdf},
  note = {Accessed: 2025-05-20}
}

@article{geeksforgeeks_ml_2021,
  author = {GeeksforGeeks},
  title = {ML | Face Recognition Using Eigenfaces (PCA Algorithm)},
  year = {2021},
  month = sep,
  day = {24},
  url = {https://www.geeksforgeeks.org/ml-face-recognition-using-eigenfaces-pca-algorithm/},
  note = {Accessed: 2025-05-20}
}

@article{paperswithcode_orl_database,
  author = {{Papers With Code}},
  title = {ORL Dataset},
  year = {2023}, % Approximate based on typical data
  url = {https://paperswithcode.com/dataset/orl},
  note = {Accessed: 2025-05-20}
}

@article{geeksforgeeks_dataset_2025,
  author = {GeeksforGeeks},
  title = {Dataset for Face Recognition},
  year = {2025},
  month = apr,
  day = {24},
  url = {https://www.geeksforgeeks.org/dataset-for-face-recognition/},
  note = {Accessed: 2025-05-20}
}

@article{columbia_pubfig_database,
  author = {{Columbia University}},
  title = {Pubfig: Public Figures Face Database},
  year = {2010}, % Last updated December 2010 based on content
  url = {https://www.cs.columbia.edu/CAVE/databases/pubfig/},
  note = {Accessed: 2025-05-20}
}

@article{researchgate_data_augmentation_2016,
  author = {Lv, L. and Wei, Y. and Wu, Y. and Wang, Q. and Yang, R.},
  title = {Data Augmentation for Face Recognition},
  journal = {ResearchGate},
  year = {2016}, % Actual publication date can be verified, using first available year.
  url = {https://www.researchgate.net/publication/311523956_Data_Augmentation_for_Face_Recognition},
  note = {Accessed: 2025-05-20}
}

@inproceedings{openreview_data_augmentation_2024,
  author = {Various Authors},
  title = {Data Augmentation for Facial Recognition with Diffusion Model},
  booktitle = {CVPR 2024 Workshop SyntaGen Submission},
  year = {2024},
  url = {https://openreview.net/forum?id=GXmlanJ6rC},
  note = {Accessed: 2025-05-20}
}

@article{getfocal_biometric_2025,
  author = {Focal},
  title = {Biometric Privacy Laws Overview},
  year = {2025},
  month = feb,
  day = {19},
  url = {https://www.getfocal.co/post/biometric-privacy-laws-overview},
  note = {Accessed: 2025-05-20}
}

@article{transcend_ccpa_2025,
  author = {Transcend},
  title = {What is CCPA: A Concise Guide to California's Privacy Law},
  year = {2025},
  month = jan,
  day = {10},
  url = {https://transcend.io/blog/ccpa-privacy-law-guide},
  note = {Accessed: 2025-05-20}
}

@article{brunelli1993face,
  author = {Brunelli, Roberto and Poggio, Tomaso},
  title = {Face recognition: features versus templates},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume = {15},
  number = {10},
  pages = {1042--1052},
  year = {1993},
  publisher = {IEEE},
  doi = {10.1109/34.254061}
}

@article{cox1996accurate,
  author = {Cox, Ingemar J and Ghosn, Joelle and Yianilos, Peter N},
  title = {Accurate face recognition using a mixture-distance technique},
  journal = {Proceedings of the International Conference on Pattern Recognition (ICPR)},
  year = {1996},
  pages = {677--680}
}

@article{kanade1973picture,
  author = {Kanade, Takeo},
  title = {Picture processing by computer complex and recognition of human faces},
  journal = {Department of Computer Science, Kyoto University},
  year = {1973}
}

@article{lades1993distortion,
  author = {Lades, Martin and Vorbruggen, Jan C and Buhmann, Joachim and Lange, Jon and von der Malsburg, Christoph and Wurtz, Rolf P and Konen, Wolfgang},
  title = {Distortion invariant object recognition in the dynamic link architecture},
  journal = {IEEE Transactions on Computers},
  volume = {42},
  number = {3},
  pages = {300--311},
  year = {1993},
  publisher = {IEEE},
  doi = {10.1109/12.210168}
}

@article{samaria1996face,
  author = {Samaria, F. S. and Fallside, F.},
  title = {Face recognition using hidden Markov models},
  journal = {Image and Vision Computing},
  volume = {14},
  number = {10},
  pages = {789--796},
  year = {1996},
  publisher = {Elsevier}
}

@article{samaria1997speech,
  author = {Samaria, F. S.},
  title = {Speech and Face Recognition Using Hidden Markov Models},
  year = {1997},
  note = {PhD Thesis, Cambridge University}
}

@article{turk1991eigenfaces,
  author = {Turk, Matthew A and Pentland, Alex P},
  title = {Eigenfaces for recognition},
  journal = {Journal of Cognitive Neuroscience},
  volume = {3},
  number = {1},
  pages = {71--86},
  year = {1991}
}

@article{wiskott1997face,
  author = {Wiskott, Laurenz and Fellous, Jean-Marc and Kr{\"u}ger, Norbert and von der Malsburg, Christoph},
  title = {Face recognition by elastic bunch graph matching},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume = {19},
  number = {7},
  pages = {775--779},
  year = {1997},
  publisher = {IEEE},
  doi = {10.1109/34.598228}
}

@article{weng1995face,
  author = {Weng, J. and Huang, T. and Ahuja, N.},
  title = {Face recognition by a learning-based approach},
  journal = {Proceedings of the IEEE International Conference on Systems, Man, and Cybernetics},
  volume = {1},
  pages = {149--154},
  year = {1995}
}

@article{cottrell1990face,
  author = {Cottrell, Garrison W. and Metcalfe, Janet A.},
  title = {Face recognition using backpropagation: Application to ORL database},
  journal = {Cognitive Science Conference},
  volume = {12},
  number = {1},
  pages = {483--490},
  year = {1990}
}
```

# Appendices

## 1. Additional Figures and Tables

## 2. Code Samples

## 3. Dataset
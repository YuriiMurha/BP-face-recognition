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
  ![[Pasted image 20241008150151.png]]
  **Figure 1**: Architecture of the convolutional neural network used in our custom model.

> This image will illustrate the layers of the CNN, including convolutional, pooling, and fully connected layers, emphasizing how each step processes and refines the image data.

This custom model creation approach provided us with significant flexibility. However, it demanded substantial effort in terms of fine-tuning the model and optimizing performance. Compared to pre-built solutions, this method allowed us to tailor our model specifically to the dataset and tasks at hand.

### References
- [Comparison of deep learning software](https://en.wikipedia.org/wiki/Comparison_of_deep_learning_software)
- [TensorFlow Documentation](https://www.tensorflow.org/)  
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

![[Files/Pasted image 20241008151235.png]]
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

    @inproceedings{schroff2015facenet,   title={FaceNet: A unified embedding for face recognition and clustering},   author={Schroff, Florian and Kalenichenko, Dmitry and Philbin, James},   booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},   pages={815--823},   year={2015} }

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

    @manual{awsrekognition,   title = {Amazon Rekognition Developer Guide},   author = {Amazon Web Services},   year = {2023},   note = {Retrieved from \url{https://docs.aws.amazon.com/rekognition/}} }

## Comparison (doplniť)

While both approaches utilize deep learning techniques for face recognition, the key difference lies in the level of control and customization they offer. The custom model approach allows for full flexibility in terms of architecture design, dataset preprocessing, and performance tuning. However, this comes at the cost of increased complexity and development time.

In contrast, the `face-recognition` library, built on dlib, offers a streamlined solution that abstracts many of the intricate details of face recognition. This is particularly advantageous for rapid prototyping or situations where pre-built models meet the required level of accuracy.

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

The Eigenface method is a classical approach based on Principal Component Analysis (PCA). Developed in the early 1990s, this method represents faces as linear combinations of a set of basis images known as "eigenfaces." Each eigenface corresponds to a direction of maximal variance in the face dataset. The idea behind this method is to reduce the dimensionality of face data while retaining the most significant information that differentiates one face from
![[Pasted image 20241017002332.png]]

To recognize a face using the Eigenface approach, an input image is projected onto the subspace spanned by the eigenfaces. The resulting projection is compared to stored projections of known faces in the database. Recognition is achieved by measuring the similarity between the input face's projection and the stored ones.

While the Eigenface method was a significant advancement at the time, its main limitation lies in its sensitivity to variations in lighting, pose, and expression. It also struggles with facial recognition in unconstrained environments.

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

![[Pasted image 20241017003933.png]]

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

## Creating Your Own Dataset

Creating a custom dataset involves several steps and considerations:

### 1. Collecting images

For this thesis, the dataset was created using two main sources: a personal webcam and a security camera installed at the university. A series of 30 images was captured for each dataset, with one image taken every two seconds over a one-minute period.

> For best performance I needed to shoot my face from different angles.
> ![Web camera image example | 400](webcam_example.jpg) > _Figure 1: Picture captured using a web camera._

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

### 3. Augmentation

Data augmentation is a technique used to artificially increase the diversity of the training data without actually collecting new data.
For this thesis, the `Albumentations` library was used to perform data augmentation by implementing a comprehensive augmentation pipeline, incuding ==random cropping== (450x450 pixels), ==horizontal and vertical flipping==, ==brightness and contrast adjustments==, ==gamma correction==, and ==color shifts==.
This helped improve the robustness of the face recognition model by exposing it to a wider variety of training samples.

**TODO**: add parameter values to text, note that during augmentation we also augmented labels.

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


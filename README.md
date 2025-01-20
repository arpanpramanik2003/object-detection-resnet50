# CIFAR-10 Image Classification using ResNet50

## Introduction
This project focuses on classifying images from the CIFAR-10 dataset using a deep learning model based on the ResNet50 architecture. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. Leveraging the power of the pre-trained ResNet50 model, the project aims to achieve high accuracy through transfer learning.

## Project Overview
The steps involved in the project are as follows:

1. **Dataset Preparation**
   - Download the CIFAR-10 dataset using Kaggle API.
   - Extract the dataset and perform necessary preprocessing.
   - Apply One-Hot Encoding to the labels.

2. **Data Preprocessing**
   - Load image data and convert them into numpy arrays.
   - Normalize pixel values by scaling them to the range [0,1].
   - Split the dataset into training and testing sets.

3. **Model Development**
   - Utilize the ResNet50 pre-trained model with `imagenet` weights.
   - Add additional layers such as upsampling, dense, dropout, and batch normalization.
   - Compile the model using RMSprop optimizer and categorical crossentropy loss function.

4. **Model Training and Evaluation**
   - Train the model with the training dataset and validate using a validation split.
   - Evaluate model performance using accuracy and loss metrics.

5. **Results Visualization**
   - Plot training vs validation loss.
   - Plot training vs validation accuracy.

## Requirements
To run this project, you need the following dependencies:

```bash
pip install numpy pandas matplotlib tensorflow keras opencv-python PIL scikit-learn kaggle
```

## Execution Steps
1. Ensure you have the `kaggle.json` file configured to download the dataset.
2. Run the Python script to download, preprocess, train, and evaluate the model.
3. Observe the evaluation results and plots to analyze model performance.

## Model Performance
The model achieved a good test accuracy, showing the effectiveness of transfer learning using ResNet50. Further improvements can be made by tuning hyperparameters and using data augmentation techniques.

## Conclusion
This project demonstrates the application of deep learning for image classification using pre-trained models. ResNet50, with its powerful feature extraction capabilities, enhances the classification accuracy, making it suitable for practical applications.

## Author
Arpan Pramanik

## Acknowledgments
- Kaggle for the CIFAR-10 dataset.
- TensorFlow and Keras libraries for deep learning implementation.

## License
This project is licensed under the Apache License.


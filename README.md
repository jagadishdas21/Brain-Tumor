# Brain Tumor Detection
This project aims to detect brain tumor directly from the MRI images of patients using Convolutional Neural Network (CNN).

## Overview
Brain tumors can pose significant health risks, and early detection plays a crucial role in treatment and prognosis. This project focuses on building a deep learning model using Convolutional Neural Networks (CNN) to detect brain tumors directly from MRI images. By processing these images, the model can identify patterns and features associated with tumors, aiding in quicker diagnosis and potentially improving patient outcomes.

The goal of the project is to automate the detection process, reducing the need for manual analysis by healthcare professionals and providing a reliable tool for early tumor identification.

## Technologies Used
- Python
- NumPy
- Matplotlib
- Scikit-Learn
- Tensorflow
- Keras

## Dataset
The dataset used in this project is sourced from Kaggle datasets.

## Methodology
1. Data Preprocessing: The MRI images are first processed for normalization and resizing to ensure uniformity across the dataset. Techniques such as grayscale conversion and noise reduction are applied to enhance image quality and clarity.

2. Model Architecture: The CNN model is designed to extract features from MRI images through multiple convolutional layers followed by pooling layers, which help in reducing dimensionality while retaining essential information. Fully connected dense layers are added at the end to make the final classification decision.

3. Training and Validation: The model is trained using labeled MRI images (tumor and non-tumor). The dataset is split into training and validation sets, and the model's performance is evaluated using metrics like accuracy, precision, recall, and F1-score.

4. Hyperparameter Tuning: Various hyperparameters, including learning rate, batch size, and number of epochs, are optimized to improve model accuracy and generalization.

5. Evaluation and Testing: After training, the model is tested on unseen MRI images to assess its real-world performance. Confusion matrix and ROC curve analysis are conducted to measure the model's predictive capabilities.

## Installation
To run this project, you will need to have Python installed. You can install the required libraries using pip: cmd > pip install numpy matplotlib scikit-learn tensorflow keras

## Usage
1. Clone this repository to your local machine:
bash $ git clone https://github.com/jagadishdas21/brain-tumor-detection.git
3. Navigate to the project directory:
bash $ cd brain-tumor-detection
5. Open the Jupyter notebook or Python script and run the code to see the predictions.

## Results
The CNN model achieved the following results during evaluation:

Accuracy: 97%
Precision: 95%
Recall: 93%
F1-Score: 93%

The model was able to successfully identify brain tumors from MRI images with a high level of accuracy and reliability. Below is a sample output of the confusion matrix and ROC curve from the test set:

Confusion Matrix: Shows true positive, true negative, false positive, and false negative rates.
ROC Curve: Demonstrates the trade-off between true positive rate and false positive rate, with an AUC score of 0.96.
These results show that the CNN model is effective in detecting brain tumors, making it a promising tool for medical professionals.

** Contributions are welcome! If you have suggestions or improvements, please feel free to open an issue or submit a pull request.

 

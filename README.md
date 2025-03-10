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

1. Tumor
   
   ![Te-me_0083](https://github.com/user-attachments/assets/1c77b4e9-6472-4333-8ea6-76d611822173)

2. Non-Tumor
   
   ![Te-noTr_0004](https://github.com/user-attachments/assets/be565426-635e-430b-a649-c218d78574ad)


## Results

The model was able to successfully identify brain tumors from MRI images with a high accuracy and precision. Below is a sample output of the confusion matrix and accuracy/loss curves from the test set:

1. ### Confusion Matrix Analysis  

This confusion matrix represents the performance of a **binary classification model**, where:  

- **Class 0** represents one category (e.g., `"No Tumor"` or `"Negative"`).  
- **Class 1** represents the other category (e.g., `"Tumor Present"` or `"Positive"`).  

(a) Breakdowns

| Metric  | Value | Description |
|---------|-------|-------------|
| **True Positives (TP)**  | 296 | Model correctly predicted Class 1 when it was actually Class 1. |
| **True Negatives (TN)**  | 306 | Model correctly predicted Class 0 when it was actually Class 0. |
| **False Positives (FP)**  | 3   | Model incorrectly predicted Class 1 when it was actually Class 0 (**Type I Error**). |
| **False Negatives (FN)**  | 6   | Model incorrectly predicted Class 0 when it was actually Class 1 (**Type II Error**). |

(b) Performance Metrics  

- **Accuracy**  
  Accuracy = (TP + TN) / (TP + TN + FP + FN)  
  = (296 + 306) / (296 + 306 + 3 + 6)  
  = **98.52%**

- **Precision (Positive Predictive Value, PPV)**  
  Precision = TP / (TP + FP)  
  = 296 / (296 + 3)  
  = **98.99%**

- **Recall (Sensitivity, True Positive Rate, TPR)**  
  Recall = TP / (TP + FN)  
  = 296 / (296 + 6)  
  = **98.01%**

- **F1 Score**  
  F1 Score = 2 × (Precision × Recall) / (Precision + Recall)  
  = **98.49%**

This model demonstrates **high accuracy, precision, recall, and F1-score**, indicating that it performs well in distinguishing between the two classes.  

   ![output](https://github.com/user-attachments/assets/3c9e5501-7a87-41d0-a06e-aa90e3eb33e9)

2. ### Accuracy/Loss Curve Analysis

(a) Left Graph – Model Accuracy:
The training accuracy (blue line) starts at 84.93% and increases steadily, reaching 98.79% by the final epoch.
The validation accuracy (orange line) begins with 70.87% but improves significantly, closely matching the training accuracy at around 98.53%.
This indicates that the model is learning effectively without major signs of overfitting.

(b) Right Graph – Model Loss:
The training loss (blue line) starts moderately high and decreases steadily, nearing zero as the epochs progress.
The validation loss (orange line) begins very high (~8), but drops significantly within the first few epochs and stabilizes near the training loss.
The alignment of training and validation loss suggests good generalization, meaning the model is learning patterns well without excessive memorization.

![output1](https://github.com/user-attachments/assets/f7460fc0-acc1-4462-bd21-78478c063ea4)

## Demonstration
https://github.com/user-attachments/assets/fbca20cc-e4ae-4798-bd83-4c9497481dd5

** Contributions are welcome! If you have suggestions or improvements, please feel free to open an issue or submit a pull request.

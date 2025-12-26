# Breast Cancer Prediction using Machine Learning

## Project Overview
This project focuses on predicting whether a breast tumor is **benign** or **malignant** using machine learning techniques. The objective is to build a reliable and interpretable classification model by combining **exploratory data analysis (EDA)**, **feature selection**, and **model evaluation with cross-validation**.

Multiple models were explored to understand performance differences, and a **Random Forest classifier** was selected as the final model based on accuracy and generalization performance.

---

## Dataset Description
The dataset consists of numerical features extracted from digitized images of breast tissue samples. These features describe properties of cell nuclei such as size, shape, texture, and boundary irregularity.

The dataset used in this project was sourced from Kaggle:
**Breast Cancer Detection Dataset**  
https://www.kaggle.com/datasets/aynaanquraishi/breast-cancer-detection

The dataset contains numerical features extracted from digitized images of breast tissue samples. These features describe characteristics of cell nuclei such as size, shape, texture, and boundary irregularity.

- **Target Variable**
  - `diagnosis`
    - `0` → Benign (B)
    - `1` → Malignant (M)

- **Feature Type**
  - All input features are numerical.
  - No categorical input features are present.

The dataset was downloaded locally and uploaded to the working environment for analysis and model development.

---

## Exploratory Data Analysis (EDA)
EDA was performed to understand:
- Feature distributions
- Differences between benign and malignant samples
- Presence of outliers and overlap between classes

Visualization techniques such as boxplots were used to study class separation.  
This step helped in identifying informative features and understanding their behavior before model training.

---

## Feature Selection
Feature selection was driven primarily by **EDA**, with **Pearson correlation** used as a supporting validation step.

The final selected features represent different tumor characteristics:

- `area_worst` – worst-case tumor size  
- `concave points_worst` – sharp inward boundary points  
- `concavity_worst` – depth of boundary irregularities  
- `compactness_worst` – shape compactness  
- `texture_mean` – overall surface roughness  

This feature set captures size, shape, boundary irregularity, and texture while avoiding excessive redundancy.

---

## Models Implemented

### Decision Tree Classifier
- Used as a baseline model
- Provides interpretability
- Demonstrates initial performance without ensembling
  
**Test Accuracy:** ~95.6%

---

### Support Vector Machine (SVM)
- Linear kernel used
- Hyperparameter `C` tuned using GridSearchCV
- Evaluated using cross-validation

**Test Accuracy:** ~96%

---

### Random Forest Classifier (Final Model)
- Ensemble of multiple decision trees
- Reduces variance compared to a single tree
- More robust to noise and feature interactions
  
**Test Accuracy:** **~97.3%**

The Random Forest model achieved the best overall performance while maintaining stable generalization.

---

## Cross-Validation
K-fold cross-validation was applied to evaluate model stability and ensure that performance was not dependent on a single train–test split.

Cross-validation results confirmed:
- Consistent performance across folds
- Reliable generalization
- No significant overfitting

---

## Model Evaluation
Models were evaluated using:
- Accuracy
- Confusion Matrix
- Precision, Recall, and F1-score

Although training accuracy reached 100% for tree-based models, the relatively small gap between training and test accuracy, along with stable cross-validation results, indicates only **mild and acceptable overfitting**, which is expected for such models.

---

## Final Conclusion
- EDA-driven feature selection played a key role in model performance.
- Cross-validation ensured reliability and robustness of results.
- The Random Forest classifier achieved the highest performance with approximately **97% test accuracy**.
- The final model generalizes well and demonstrates strong potential for breast cancer classification tasks.

---

## Technologies Used
- Python
- Pandas
- Matplotlib, Seaborn
- Scikit-learn
- T4 GPU (Google Colab)



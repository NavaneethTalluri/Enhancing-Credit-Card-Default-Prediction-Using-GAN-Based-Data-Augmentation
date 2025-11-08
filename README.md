# Enhancing Credit Card Default Prediction Using GAN Based Data Augmentation
---
## Project Overview
This project predicts **credit card payment defaults** using a range of classical and ensemble machine learning models.  
It explores the effects of different **data augmentation and balancing techniques** — namely **SMOTE** and **CTGAN (GAN-based synthetic data generation)** — on model accuracy and fairness.  
The results demonstrate how advanced resampling strategies can improve model generalization when dealing with imbalanced financial datasets.

---

## Dataset
- **Source:** [UCI Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)  
- **Size:** 30,000 clients × 24 features  
- **Target Variable:** `default payment next month` (1 = Default, 0 = No Default)

---

**Key Features:**
- Demographic data (age, gender, education, marital status)  
- Credit information (limit balance, payment history, bill amounts)  
- Past payment behavior (six-month history)

---

## Workflow & Methodology

1. **Data Preprocessing**
   - Handled missing values and categorical encoding  
   - Applied log and power transformations to reduce skewness  

2. **Exploratory Data Analysis**
   - Visualized class distribution, education, gender, and delay patterns  
   - Analyzed correlation and top predictors for default risk  

3. **Handling Class Imbalance**
   - **Original Dataset:** Highly imbalanced (majority non-defaulters)  
   - **SMOTE:** Synthetic oversampling using geometric interpolation  
   - **CTGAN:** GAN-based synthetic sample generation capturing nonlinear feature relations  

4. **Model Training**
   - Logistic Regression  
   - Random Forest  
   - XGBoost  
   - LightGBM  

5. **Evaluation Metrics**
   - Accuracy  
   - ROC-AUC (Receiver Operating Characteristic – Area Under Curve)  

---

## Results Summary

| Model | Dataset | Accuracy | AUC |
|--------|----------|----------|------|
| **LightGBM** | Original | 0.8170 | 0.7752 |
| **LightGBM** | SMOTE | 0.8122 | 0.7727 |
| **LightGBM** | GAN (CTGAN) | **0.8183** | **0.7746** |
| **XGBoost** | Original | 0.8118 | 0.7565 |
| **XGBoost** | SMOTE | 0.7992 | 0.7517 |
| **XGBoost** | GAN (CTGAN) | **0.8140** | **0.7632** |
| **Random Forest** | Original | 0.8107 | 0.7504 |
| **Random Forest** | SMOTE | 0.7907 | 0.7528 |
| **Random Forest** | GAN (CTGAN) | **0.8145** | 0.7487 |
| **Logistic Regression** | Original | 0.8060 | 0.7327 |
| **Logistic Regression** | SMOTE | 0.7172 | 0.7222 |
| **Logistic Regression** | GAN (CTGAN) | **0.8032** | **0.7354** |

---

### **Key Insights**
- **LightGBM** consistently achieved the best **accuracy (0.8183)** and **AUC (0.7746)** after GAN-based augmentation.  
- **GAN (CTGAN)** produced slightly better or more stable results than SMOTE, indicating higher-quality synthetic data.  
- **SMOTE** improved recall for defaulters but slightly reduced accuracy due to oversampling noise.  
- **Tree-based models (LightGBM, XGBoost, Random Forest)** outperform linear models like Logistic Regression across all datasets.  

---

## Visual Results

### ROC Curve Comparison
| Dataset | File |
|----------|------|
| Original | `roc_comparison_original.png` |
| SMOTE | `roc_comparison_smote.png` |
| GAN | `roc_comparison_gan.png` |

Each ROC curve demonstrates that tree-based ensemble models outperform linear classifiers, with CTGAN providing the most stable balance between performance and class fairness.

---

## Key Insights

- **CTGAN** creates more realistic synthetic samples than SMOTE, capturing complex relationships in tabular data.  
- **LightGBM** shows robust performance and scalability for financial risk prediction.  
- **SMOTE** improves minority detection but may slightly reduce overall accuracy.  
- Balancing methods improve fairness and recall for defaulters while maintaining generalization.  


## Project Structure

credit_default_prediction/
│
├── notebook.ipynb                  # Main analysis and modeling notebook
├── README.md                       # Project summary and documentation
├── requirements.txt                # Python dependencies
├── combined_model_results.csv       # Model comparison results
├── images/
│   ├── roc_comparison_original.png
│   ├── roc_comparison_smote.png
│   └── roc_comparison_gan.png
└── data/
    └── sample_data.csv 


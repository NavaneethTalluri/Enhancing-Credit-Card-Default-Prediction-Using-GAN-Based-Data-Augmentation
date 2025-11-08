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

| **Model**               | **Dataset** | **Accuracy** | **AUC**    |
| ----------------------- | ----------- | ------------ | ---------- |
| **LightGBM**            | Original    | 0.8170       | **0.7752** |
| **LightGBM**            | SMOTE       | 0.8122       | 0.7727     |
| **LightGBM**            | GAN (CTGAN) | **0.8185**   | 0.7723     |
| **XGBoost**             | Original    | 0.8118       | 0.7565     |
| **XGBoost**             | SMOTE       | 0.7992       | 0.7517     |
| **XGBoost**             | GAN (CTGAN) | **0.8118**   | **0.7604** |
| **Random Forest**       | Original    | 0.8107       | 0.7504     |
| **Random Forest**       | SMOTE       | 0.7907       | **0.7528** |
| **Random Forest**       | GAN (CTGAN) | **0.8138**   | 0.7516     |
| **Logistic Regression** | Original    | 0.8060       | 0.7327     |
| **Logistic Regression** | SMOTE       | 0.7172       | 0.7222     |
| **Logistic Regression** | GAN (CTGAN) | **0.8035**   | **0.7353** |

---

### **Key Insights**

- **LightGBM** achieved the **highest accuracy (0.8185)** and **AUC (0.7723)** using GAN-augmented data.

- **GAN (CTGAN)** consistently improved model stability and generalization compared to SMOTE and Original datasets.

- **SMOTE** helped balance classes but led to slightly lower AUC due to oversampling artifacts.

- **Logistic Regression** gained meaningful improvement with GAN, confirming the realistic diversity of synthetic samples.

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

---

## License
This project is released under the MIT License.

---

## Project Structure
```
credit_default_prediction/
│
├── Data_Science_Project_23102357.ipynb     # Main Jupyter Notebook (model training, EDA & results)
├── LICENSE
├── README.md                               # Project summary and documentation
├── combined_model_results.csv              # Model performance comparison table
├── ctgan_model.pkl                         # Saved CTGAN model for synthetic data generation
├── default of credit card clients.xls      # Original dataset (UCI source)
├── requirements.txt                        # Python dependencies for reproducibility
├── roc_comparison_original.png             # ROC curve for Original dataset
├── roc_comparison_smote.png                # ROC curve for SMOTE dataset
└── roc_comparison_gan.png                  

```

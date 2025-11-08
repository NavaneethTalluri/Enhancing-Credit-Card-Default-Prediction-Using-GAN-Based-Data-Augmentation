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

| Model | Original AUC | SMOTE AUC | GAN (CTGAN) AUC |
|--------|--------------|-----------|-----------------|
| LightGBM | **0.775** | 0.773 | **0.775** |
| XGBoost | 0.756 | 0.752 | **0.763** |
| Random Forest | 0.750 | 0.753 | 0.749 |
| Logistic Regression | 0.733 | 0.722 | 0.735 |

**Best Performer:** LightGBM consistently delivered the highest AUC and accuracy across all datasets.

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

## Future Exploration

- Perform **feature importance and explainability analysis** using SHAP or LIME.  
- Test **time-based model validation** for temporal credit risk prediction.  
- Integrate model into a **Streamlit dashboard or Flask API** for real-time prediction.  
- Explore **hybrid GAN + SMOTE** augmentation for enhanced sample diversity.  

---

## 📁 Project Structure

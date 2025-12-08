# Enhancing Credit Card Default Prediction Using GAN-Based Data Augmentation
---
## Project Overview
This project predicts **credit card payment defaults** using a range of classical and ensemble machine learning models.  
It explores the effects of different **data augmentation and balancing techniques** — namely **SMOTE** and **CTGAN (GAN-based synthetic data generation)** — on model accuracy and fairness.  
The results demonstrate how advanced resampling strategies can improve model generalisation when dealing with imbalanced financial datasets.

---

## Dataset
- **Source:** [UCI Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)  
- **Size:** 30,000 clients × 24 features  
- **Target Variable:** `default payment next month` (1 = Default, 0 = No Default)

---

**Key Features:**
- Demographic data (age, gender, education, marital status)  
- Credit information (limit balance, payment history, bill amounts)  
- Past payment behaviour (six-month history)

---

## Workflow & Methodology

1. **Data Preprocessing**
   - Handled missing values and categorical encoding  
   - Applied log and power transformations to reduce skewness  

2. **Exploratory Data Analysis**
   - Visualised class distribution, education, gender, and delay patterns  
   - Analysed correlation and top predictors for default risk  

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
   - Precision
   - Recall
   - F1-score
   - AUC
   - ROC-AUC (Receiver Operating Characteristic – Area Under Curve)  

---

## Results Summary
### Model Performance Comparison
| **Model**               | **Accuracy** | **Precision** | **Recall** | **F1-score** | **AUC**    |
|------------------------|--------------|---------------|------------|--------------|------------|
| LightGBM               | **0.8170**   | 0.6562        | 0.3625     | 0.4670       | **0.7752** |
| XGBoost                | 0.8118       | 0.6289        | 0.3640     | 0.4611       | 0.7565     |
| Random Forest          | 0.8107       | 0.6262        | 0.3572     | 0.4549       | 0.7504     |
| Logistic Regression    | 0.8060       | **0.6601**    | 0.2532     | 0.3660       | 0.7327     |

### LGBM Model performance comparison original dataset vs smote vs CTGAN
| **Model**           | **Dataset** | **Accuracy** | **Precision** | **Recall** | **F1-score** | **AUC**    |
|--------------------|-------------|--------------|---------------|------------|--------------|------------|
| LightGBM (Tuned)    | CTGAN       | 0.7683       | 0.4812        | 0.6081     | 0.5373       | 0.7755     |
| LightGBM (Tuned)    | Original    | 0.7622       | 0.4716        | 0.6247     | 0.5374       | **0.7798** |
| LightGBM (Tuned)    | SMOTE       | **0.7895**   | **0.5242**    | 0.5222     | 0.5232       | 0.7696     |


---

## Visual Results

### ROC Curve Comparison  
![ROC Curves](roc_comparison_original.png)

### Model Performance on Original Dataset  
![Unbalanced Results](model_performance_metrics_unbalanced.png)

### LGBM Performance Across Augmented Datasets  
![LGBM Dataset Comparison](lgbm_dataset_performance_comparison.png)

---

## Key Insights

- CTGAN-generated data provides more realistic synthetic samples than SMOTE, effectively capturing complex feature relationships in tabular credit data.

- LightGBM consistently delivers the strongest overall performance, achieving high accuracy and AUC, making it well-suited for financial risk prediction.

- SMOTE improves the detection of minority classes (defaulters) but may slightly reduce overall model accuracy due to oversampling artefacts.

- Data augmentation and balancing methods enhance fairness and recall for defaulters while maintaining generalisation across models.

---

## License
This project is released under the MIT License.

---

## Project Structure
```
Enhancing_Credit_Card_Default_Prediction_Using_GAN_Based_Data_Augmentation_/
│
├── Enhancing_Credit_Card_Default_Prediction_Using_GAN_Based_Data_Augmentation_.ipynb   # Main Jupyter Notebook (EDA, GAN training & evaluation)
├── README.md                                                                            # Project documentation & visual results
├── LICENSE                                                                              # License for open-source usage
│
├── default of credit card clients.xls                                                  # Original dataset (UCI Credit Default dataset)
├── ctgan_model.pkl                                                                      # Trained CTGAN model for synthetic sample generation
├── requirements.txt                                                                     # Package dependencies for full reproducibility
│
├── model_performance_metrics_unbalanced.png                                            # Baseline performance on original dataset
├── lgbm_dataset_performance_comparison.png                                             # Original vs SMOTE vs GAN comparison
├── roc_comparison_original.png                                                         # ROC curves showing model discrimination ability
                

```

## Problem statement
The goal is to predict whether a breast tumour is malignant or benign from measurements of cell nuclei computed from digitized images of fine needle aspirate (FNA) biopsies. This is a binary classification problem. Five models (Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM) are trained on the same data and compared using Accuracy, AUC, Precision, Recall, F1 and MCC. The models are demonstrated through an interactive Streamlit app.

## Dataset description
- **Source:** Breast Cancer Wisconsin (Diagnostic) dataset, UCI Machine Learning Repository (also bundled with scikit-learn)
- **Instances:** 569
- **Features:** 30 numeric features: 10 nucleus measurements (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), each as mean, standard error and worst value
- **Target:** `target` (0 = malignant, 1 = benign), 212 malignant and 357 benign, so moderately imbalanced
- **Preprocessing:** no missing values; features scaled with StandardScaler inside each model pipeline
- **Split:** 80/20 stratified, `random_state=42`. The test split (114 rows) is saved as `test_data.csv`.

### Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1-Score | MCC |
|:---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | **0.9825** | **0.9954** | **0.9861** | 0.9861 | **0.9861** | **0.9623** |
| Decision Tree | 0.9123 | 0.9147 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| KNN | 0.9737 | 0.9884 | 0.9600 | **1.0000** | 0.9796 | 0.9442 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Support Vector Machine (SVM) | **0.9825** | 0.9950 | **0.9861** | 0.9861 | **0.9861** | **0.9623** |


| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall performer: Accuracy 0.9825, AUC 0.9954, MCC 0.9623. Precision, recall and F1 are all 0.9861, so errors are low and balanced across both classes. This suggests the classes are almost linearly separable once the features are scaled. |
| Decision Tree | Weakest model: Accuracy 0.9123, AUC 0.9147, MCC 0.8174. The much lower AUC shows its probability estimates are coarse, since a single tree outputs only a few distinct probability values. It is also prone to overfitting the training data, even with max_depth limited to 6. |
| KNN | Strong results: Accuracy 0.9737, AUC 0.9884, MCC 0.9442. It has perfect recall (1.0), so it missed no benign cases, but its precision (0.96) is slightly lower because it produced a few false positives. Performance depends on feature scaling and on k=7. |
| Naive Bayes | Accuracy 0.9298 and MCC 0.8492 are lower than the top models, but AUC is high (0.9868), meaning it ranks cases well even when its default threshold makes more mistakes. The independence assumption is not realistic here, since many features (radius, perimeter, area) are strongly correlated. |
| Support Vector Machine | Matches Logistic Regression on Accuracy (0.9825), F1 (0.9861) and MCC (0.9623), with AUC of 0.9950. The RBF kernel can model non-linear boundaries, but here it gives no clear gain over the simpler linear model. |
| **Overall, Winner for your dataset?** | **Logistic Regression.** It ties with SVM on accuracy and MCC and has the highest AUC (0.9954). Because performance is effectively equal, the simpler, faster and more interpretable model is the better choice. MCC was used alongside accuracy because the classes are moderately imbalanced (about 37% malignant). |


## Project structure
```
project-folder/
│-- app.py                 # Streamlit app
│-- train_models.py        # trains and saves all 5 models
│-- metrics_utils.py       # metric calculation helper
│-- requirements.txt
│-- README.md
│-- test_data.csv          # held-out test data (20%) for the app
│-- model/                 # saved .joblib models, feature_names.json, metrics_table.csv
```

## Setup and configuration
- Python version: 3.13
- Create and activate a virtual environment:
  ```
  python -m venv venv
  
  venv\Scripts\activate          # Windows
  source venv/bin/activate       # Linux / Mac / BITS Virtual Lab
  ```
- Install dependencies:
  ```
  pip install -r requirements.txt
  ```
- Training configuration: 80/20 stratified train-test split, `random_state=42`, StandardScaler applied inside each model pipeline. The target column is saved as `target` in `test_data.csv`.

## How to run

1. Start the Streamlit app:
   ```
   streamlit run app.py
   ```
2. Open http://localhost:8501, upload `test_data.csv` in the sidebar, choose a model from the dropdown, and view the metrics, confusion matrix and classification report.



Live app: https://machine-learning-s5apm7ztvdcajx4fbclkvq.streamlit.app/
```
pip install -r requirements.txt
streamlit run app.py
```

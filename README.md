## a. Problem statement
The goal is to predict whether a breast tumour is malignant or benign from measurements of cell nuclei computed from digitized images of fine needle aspirate (FNA) biopsies. This is a binary classification problem. Five models (Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM) are trained on the same data and compared using Accuracy, AUC, Precision, Recall, F1 and MCC. The models are demonstrated through an interactive Streamlit app.

## b. Dataset description
- **Source:** Breast Cancer Wisconsin (Diagnostic) dataset, UCI Machine Learning Repository (also bundled with scikit-learn)
- **Instances:** 569
- **Features:** 30 numeric features: 10 nucleus measurements (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), each as mean, standard error and worst value
- **Target:** `target` (0 = malignant, 1 = benign), 212 malignant and 357 benign, so moderately imbalanced
- **Preprocessing:** no missing values; features scaled with StandardScaler inside each model pipeline
- **Split:** 80/20 stratified, `random_state=42`. The test split (114 rows) is saved as `test_data.csv`.


- [Allmodel.csv](https://github.com/user-attachments/files/34510550/Allmodel.csv)

- [classification_report.csv](https://github.com/user-attachments/files/34510602/classification_report.csv)



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

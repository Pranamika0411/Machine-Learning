# Classifier Comparison Lab

## a. Problem statement
<Describe the prediction task in 3-4 lines: what is being predicted, why it matters, binary or multi-class.>

## b. Dataset description  [1 mark]
- Source: <Kaggle/UCI link>
- Instances: <n> | Features: <n> (>=12) | Target: <name, classes, class balance>
- Preprocessing: <missing values, encoding, scaling (StandardScaler inside each pipeline)>
- Split: 80/20 stratified, random_state=42

## c. GitHub repository
<your repo link>

## d. Models used  [5 marks]
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Support Vector Machine | | | | | | |

(Paste values from `model/metrics_table.csv` after running `python train_models.py <data.csv> <target_col>`.)

### Observations  [3 marks]
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | <your own analysis> |
| Decision Tree | <...> |
| KNN | <...> |
| Naive Bayes | <...> |
| Support Vector Machine | <...> |
| **Overall winner** | <model + justification using MCC/AUC, not accuracy alone> |

## Run locally
```
pip install -r requirements.txt
python train_models.py data.csv target_col
streamlit run app.py
```
Live app: <Streamlit Community Cloud link>

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
Live app: < Streamlit https://machine-learning-s5apm7ztvdcajx4fbclkvq.streamlit.app/ >

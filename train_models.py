"""Train 5 classifiers on one dataset, save models + test data + metrics.
Usage: python train_models.py                      (uses built-in demo dataset)
       python train_models.py data.csv target_col  (your own Kaggle/UCI CSV)
"""
import sys, json, joblib
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from metrics_utils import compute_metrics

def load_data():
    if len(sys.argv) >= 3:
        df = pd.read_csv(sys.argv[1]); target = sys.argv[2]
    else:  # demo fallback so the pipeline runs end to end
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer(as_frame=True); df = d.frame; target = "target"
    df = df.dropna().reset_index(drop=True)
    X = df.drop(columns=[target]); y = df[target]
    X = pd.get_dummies(X, drop_first=True)          # encode categoricals
    if y.dtype == object or y.dtype.name == "category":
        le = LabelEncoder(); y = pd.Series(le.fit_transform(y), name=target)
        joblib.dump(le, "model/label_encoder.joblib")
    return X, y.rename("target")

X, y = load_data()
assert X.shape[1] >= 12 and len(X) >= 500, "Dataset must have >=12 features and >=500 rows"

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
}
rows = {}
for name, clf in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipe.fit(X_tr, y_tr)
    fname = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipe, f"model/{fname}")
    rows[name] = compute_metrics(y_te, pipe.predict(X_te), pipe.predict_proba(X_te))

pd.concat([X_te, y_te], axis=1).to_csv("test_data.csv", index=False)
json.dump(list(X.columns), open("model/feature_names.json", "w"))
res = pd.DataFrame(rows).T.round(4)
res.to_csv("model/metrics_table.csv")
print(res.to_markdown())

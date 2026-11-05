import json, joblib
import pandas as pd, numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from metrics_utils import compute_metrics

st.set_page_config(page_title="Classifier Comparison Lab", page_icon="🧪", layout="wide")
MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}

@st.cache_resource
def load_model(path): return joblib.load(path)

st.title("🧪 Classifier Comparison Lab")
st.caption("Upload a test CSV (features + a column named `target`), pick a saved model, and inspect its performance. No retraining happens here.")

with st.sidebar:
    st.header("Controls")
    choice = st.selectbox("Choose a model", list(MODEL_FILES))
    up = st.file_uploader("Upload test CSV", type="csv")
    st.markdown("Tip: `test_data.csv` from the repo works out of the box.")

if up is None:
    st.info("Upload a test CSV from the sidebar to begin.")
    st.stop()

df = pd.read_csv(up)
feats = json.load(open("model/feature_names.json"))
missing = [c for c in feats + ["target"] if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing[:10]}"); st.stop()

X, y = df[feats], df["target"]
model = load_model(MODEL_FILES[choice])
pred, proba = model.predict(X), model.predict_proba(X)
m = compute_metrics(y, pred, proba)

st.subheader(f"Results: {choice}")
cols = st.columns(6)
for c, (k, v) in zip(cols, m.items()): c.metric(k, f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(confusion_matrix(y, pred), annot=True, fmt="d", cmap="crest", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); st.pyplot(fig)
with right:
    st.markdown("**Classification report**")
    st.dataframe(pd.DataFrame(classification_report(y, pred, output_dict=True)).T.round(3))

st.subheader("All models side by side")
allm = {n: compute_metrics(y, load_model(p).predict(X), load_model(p).predict_proba(X))
        for n, p in MODEL_FILES.items()}
st.dataframe(pd.DataFrame(allm).T.round(4).style.highlight_max(axis=0, color="#cfe8cf"))

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("model/logistic_regression.joblib"),
        "Decision Tree": joblib.load("model/decision_tree.joblib"),
        "kNN": joblib.load("model/knn.joblib"),
        "Naive Bayes": joblib.load("model/naive_bayes.joblib"),
        "Random Forest": joblib.load("model/random_forest.joblib"),
        "XGBoost": joblib.load("model/xgboost.joblib"),
    }
    scaler = joblib.load("model/scaler.joblib")
    feature_cols = joblib.load("model/feature_columns.joblib")
    return models, scaler, feature_cols

models, scaler, feature_cols = load_models()

st.title("Students Performance Classification – ML Assignment 2")

st.sidebar.header("Options")

model_name = st.sidebar.selectbox(
    "Choose a model",
    list(models.keys())
)

uploaded_file = st.file_uploader(
    "Upload CSV (test data). It may optionally include GradeClass column.",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded data preview")
    st.dataframe(data.head())

    # Handle presence/absence of GradeClass
    if "GradeClass" in data.columns:
        y_true = data["GradeClass"]
        X_raw = data.drop(columns=["GradeClass"])
    else:
        y_true = None
        X_raw = data

    # Align columns with training one-hot structure
    X_raw = pd.get_dummies(X_raw, drop_first=True)
    # Add missing columns
    for col in feature_cols:
        if col not in X_raw.columns:
            X_raw[col] = 0
    # Ensure same column order
    X_raw = X_raw[feature_cols]

    X_scaled = scaler.transform(X_raw)

    clf = models[model_name]
    y_pred = clf.predict(X_scaled)

    st.subheader("Predictions")
    st.write(pd.DataFrame({"Predicted GradeClass": y_pred}))

    if y_true is not None:
        # Encode y_true if needed; here GradeClass is already labels used at training time.
        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_scaled)
        else:
            y_proba = None

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)

        try:
            if y_proba is not None:
                auc = roc_auc_score(y_true, y_proba, multi_class="ovr")
            else:
                auc = np.nan
        except Exception:
            auc = np.nan

        st.subheader("Evaluation Metrics")
        st.write(f"Accuracy: {acc:.4f}")
        st.write(f"AUC: {auc:.4f}" if not np.isnan(auc) else "AUC: N/A")
        st.write(f"Precision: {prec:.4f}")
        st.write(f"Recall: {rec:.4f}")
        st.write(f"F1 Score: {f1:.4f}")
        st.write(f"MCC: {mcc:.4f}")

        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y_true, y_pred))

        st.subheader("Classification Report")
        st.text(classification_report(y_true, y_pred))
    else:
        st.info("No GradeClass column found – showing predictions only. "
                "Include GradeClass to see metrics and confusion matrix.")
else:
    st.warning("Upload a CSV file to run predictions and compute metrics.")

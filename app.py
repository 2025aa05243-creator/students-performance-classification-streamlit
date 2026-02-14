
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.utils.multiclass import unique_labels

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

    # optional ID column
    if "StudentID" in data.columns:
        student_ids = data["StudentID"]
    else:
        student_ids = None

    # Handle presence/absence of GradeClass
    if "GradeClass" in data.columns:
        y_true = data["GradeClass"]
        X_raw = data.drop(columns=["GradeClass"])
    else:
        y_true = None
        X_raw = data

    # Align columns with training one-hot structure
    X_raw = pd.get_dummies(X_raw, drop_first=True)
    for col in feature_cols:
        if col not in X_raw.columns:
            X_raw[col] = 0
    X_raw = X_raw[feature_cols]

    X_scaled = scaler.transform(X_raw)

    clf = models[model_name]
    y_pred = clf.predict(X_scaled)

    # predictions table
    if student_ids is not None:
        pred_df = pd.DataFrame({
            "StudentID": student_ids,
            "Predicted GradeClass": y_pred
        })
    else:
        pred_df = pd.DataFrame({
            "Predicted GradeClass": y_pred
        })

    st.subheader("Predictions")
    st.dataframe(pred_df)

    # if true labels are present, compute metrics
    if y_true is not None:
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

        # Confusion matrix as table
        st.subheader("Confusion Matrix")
        labels = unique_labels(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(
            cm,
            index=[f"Actual {lbl}" for lbl in labels],
            columns=[f"Pred {lbl}" for lbl in labels],
        )
        st.dataframe(cm_df)

        # Classification report as table
        st.subheader("Classification Report (per class)")
        report_dict = classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0
        )
        report_df = pd.DataFrame(report_dict).T
        report_df = report_df[["precision", "recall", "f1-score", "support"]]
        st.dataframe(report_df)

          st.subheader("Classification Report (per class)")
        report_dict = classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0
        )
        report_df = pd.DataFrame(report_dict).T
        report_df = report_df[["precision", "recall", "f1-score", "support"]]
        st.dataframe(report_df)

        # One-row summary table for the selected model
        st.subheader("Overall Metrics (selected model)")

    else:
        st.info(
            "No GradeClass column found – showing predictions only. "
            "Include GradeClass to see metrics, confusion matrix, and classification report."
        )
else:
    st.warning("Upload a CSV file to run predictions and compute metrics.")

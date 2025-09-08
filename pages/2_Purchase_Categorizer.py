#!/usr/bin/env python3
"""
Streamlit page: Purchase Categorizer

Upload a labeled dataset A (Excel) and an unlabeled dataset B (Excel), train a
text+numeric model, and download B with predicted categories and confidences.

Notes:
- Expects columns: Date, Description, Amount (A must also include Category)
- Top-k predictions optional
- You can also load an existing model instead of training
"""

import io
import re
import sys
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

try:
    import joblib  # noqa: F401
except Exception:
    joblib = None  # type: ignore
import pickle


EXPECTED_CATEGORIES = [
    "Transfer",
    "Restaurants & Bars",
    "Takeout",
    "Gas",
    "Education",
    "Travel & Vacation",
    "Personal",
    "Medical",
    "Groceries",
    "Cash & ATM",
    "Gifts",
    "Other Income",
    "Living Expenses",
    "Entertainment & Recreation",
    "Going Out",
    "Coffee Shops",
    "Interest",
    "Financial Fees",
    "Clothing",
    "Pets",
    "House",
    "Subscriptions",
    "Taxi & Ride Shares",
    "Fitness",
    "Parking & Tolls",
    "Auto Maintenance",
    "Dentist",
    "Shopping",
    "Electronics",
    "Public Transit",
    "Charity",
    "Paychecks",
]

_amount_pat = re.compile(r"[^\d\-\(\)\.\,]+")


def parse_amount(x) -> Optional[float]:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.floating, np.integer)):
        return float(x)
    s = str(x).strip()
    if not s:
        return np.nan
    s_clean = _amount_pat.sub("", s)
    s_clean = s_clean.replace(",", "")
    neg = False
    if "(" in s_clean and ")" in s_clean:
        neg = True
        s_clean = s_clean.replace("(", "").replace(")", "")
    try:
        val = float(s_clean)
        if neg:
            val = -abs(val)
        return val
    except Exception:
        return np.nan


def clean_text(s: str) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s)
    s = re.sub(r"\s+", " ", s)
    s = s.lower().strip()
    return s


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Description" in df.columns:
        df["Description"] = df["Description"].apply(clean_text)
    if "Amount" in df.columns:
        df["Amount_std"] = df["Amount"].apply(parse_amount)
    else:
        df["Amount_std"] = np.nan
    if "Date" in df.columns:
        df["Date_parsed"] = pd.to_datetime(df["Date"], errors="coerce", infer_datetime_format=True)
        df["dow"] = df["Date_parsed"].dt.dayofweek.astype("float64")
        df["month"] = df["Date_parsed"].dt.month.astype("float64")
    else:
        df["dow"], df["month"] = np.nan, np.nan
    df["is_credit"] = (df["Amount_std"] > 0).astype("float64")
    df["is_debit"] = (df["Amount_std"] < 0).astype("float64")
    df["abs_amount"] = df["Amount_std"].abs()
    df["log_amount"] = np.log1p(df["abs_amount"].fillna(0.0))
    return df


def build_pipeline(random_state: int = 42, C: float = 1.0) -> Pipeline:
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        lowercase=False,
        strip_accents="unicode",
        sublinear_tf=True,
    )
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=200000,
        lowercase=False,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w+\b",
        sublinear_tf=True,
    )
    numeric_cols = [
        "Amount_std",
        "is_credit",
        "is_debit",
        "abs_amount",
        "log_amount",
        "dow",
        "month",
    ]
    feats = ColumnTransformer(
        transformers=[
            ("desc_char", char_vec, "Description"),
            ("desc_word", word_vec, "Description"),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imp", SimpleImputer(strategy="median")),
                        ("scale", MaxAbsScaler()),
                    ]
                ),
                numeric_cols,
            ),
        ],
        remainder="drop",
        n_jobs=None,
        sparse_threshold=0.3,
    )
    base_clf = LinearSVC(C=C, class_weight="balanced", random_state=random_state)
    clf = CalibratedClassifierCV(base_estimator=base_clf, method="sigmoid", cv=5)
    pipe = Pipeline(steps=[("features", feats), ("clf", clf)])
    return pipe


def topk_from_proba(proba: np.ndarray, classes: np.ndarray, k: int = 3):
    idx = np.argsort(-proba, axis=1)[:, :k]
    top_labels = [[str(classes[j]) for j in row] for row in idx]
    top_probs = [[float(proba[i, j]) for j in row] for i, row in enumerate(idx)]
    return top_labels, top_probs


def main():
    st.title("🧠 Purchase Categorizer")
    st.markdown("Upload a labeled dataset A (with Category) and an unlabeled dataset B; get B categorized with confidences.")

    with st.sidebar:
        st.header("Options")
        use_existing = st.checkbox("Load existing model", value=False)
        cv_folds = st.number_input("CV folds (0 to skip)", min_value=0, max_value=10, value=0, step=1)
        topk = st.number_input("Top-k predictions", min_value=1, max_value=5, value=3, step=1)
        random_state = st.number_input("Random state", min_value=0, max_value=10_000, value=42, step=1)
        C = st.number_input("LinearSVC C", min_value=0.01, max_value=100.0, value=1.0, step=0.25)

    colA, colB = st.columns(2)
    with colA:
        st.subheader("1) Labeled dataset A (Excel)")
        train_file = st.file_uploader("Upload A.xlsx (with Category)", type=["xlsx"], key="train_file") if not use_existing else None
        sheet_train = st.text_input("Sheet name for A (optional)", value="") if not use_existing else ""
    with colB:
        st.subheader("2) Unlabeled dataset B (Excel)")
        predict_file = st.file_uploader("Upload B.xlsx (no Category)", type=["xlsx"], key="predict_file")
        sheet_predict = st.text_input("Sheet name for B (optional)", value="")

    model_bytes = None
    if use_existing:
        st.subheader("Model input (optional)")
        model_upload = st.file_uploader("Upload trained model (.pkl/.joblib)", type=["pkl", "joblib"], key="model_in")
        if model_upload is not None:
            model_bytes = model_upload.read()

    run = st.button("🚀 Run categorization", type="primary")

    if run:
        if predict_file is None:
            st.error("Please upload dataset B (Excel).")
            st.stop()

        try:
            predict_df = pd.read_excel(predict_file, sheet_name=(sheet_predict or None))
        except Exception as e:
            st.error(f"Failed to read dataset B: {e}")
            st.stop()

        # Prepare B
        predict_df_orig = predict_df.copy()
        X_pred_df = prepare_dataframe(predict_df)

        # Load or train model
        model = None
        if use_existing and model_bytes is not None:
            try:
                model = pickle.loads(model_bytes)
            except Exception:
                if joblib is None:
                    st.error("Failed to load model with pickle, and joblib not available.")
                    st.stop()
                try:
                    model = joblib.load(io.BytesIO(model_bytes))
                except Exception as e:
                    st.error(f"Failed to load model: {e}")
                    st.stop()
        else:
            if train_file is None:
                st.error("Please upload dataset A (Excel) to train a model, or supply an existing model.")
                st.stop()
            try:
                train_df = pd.read_excel(train_file, sheet_name=(sheet_train or None))
            except Exception as e:
                st.error(f"Failed to read dataset A: {e}")
                st.stop()

            # Validate train data
            missing = [c for c in ["Date", "Description", "Amount", "Category"] if c not in train_df.columns]
            if missing:
                st.error(f"Training data missing columns: {missing}")
                st.stop()
            # Basic label sanity
            train_df["Category"] = train_df["Category"].astype(str)

            X_train_df = prepare_dataframe(train_df)
            X_cols = ["Description", "Amount_std", "is_credit", "is_debit", "abs_amount", "log_amount", "dow", "month"]
            y_train = train_df["Category"].astype(str)

            model = build_pipeline(random_state=int(random_state), C=float(C))

            if cv_folds and cv_folds > 1:
                skf = StratifiedKFold(n_splits=int(cv_folds), shuffle=True, random_state=int(random_state))
                try:
                    scores = cross_val_score(
                        model,
                        X_train_df[X_cols],
                        y_train,
                        cv=skf,
                        scoring="f1_macro",
                        n_jobs=None,
                    )
                    st.info(f"CV Macro F1 (k={cv_folds}): mean={scores.mean():.4f} ± {scores.std():.4f}")
                except Exception as e:
                    st.warning(f"Cross-validation failed: {e}")

            # Fit
            model.fit(X_train_df[X_cols], y_train)

        # Predict
        X_cols = ["Description", "Amount_std", "is_credit", "is_debit", "abs_amount", "log_amount", "dow", "month"]
        if not hasattr(model.named_steps["clf"], "predict_proba"):
            st.error("Classifier missing predict_proba; ensure calibrated classifier is used.")
            st.stop()

        proba = model.predict_proba(X_pred_df[X_cols])
        classes = model.named_steps["clf"].classes_
        y_pred = classes[np.argmax(proba, axis=1)]
        confidence = np.max(proba, axis=1)

        out_df = predict_df_orig.copy()
        out_df["PredictedCategory"] = y_pred
        out_df["Confidence"] = confidence

        k = int(max(1, topk))
        if k > 1:
            labels_k, probs_k = topk_from_proba(proba, classes, k=k)
            for i in range(k):
                out_df[f"Top{i+1}_Category"] = [row[i] if i < len(row) else "" for row in labels_k]
                out_df[f"Top{i+1}_Prob"] = [row[i] if i < len(row) else np.nan for row in probs_k]

        st.success(f"Done. Generated predictions for {len(out_df)} rows.")
        st.dataframe(out_df.head(20))

        # Prepare Excel for download
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            out_df.to_excel(writer, index=False, sheet_name="Predictions")
        st.download_button(
            label="📥 Download Categorized B.xlsx",
            data=buf.getvalue(),
            file_name="B_categorized.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


if __name__ == "__main__":
    main()



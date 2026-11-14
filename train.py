#!/usr/bin/env python3
"""
train.py - Train, evaluate and save the heart-disease classifier.

What this script does
---------------------
1. Loads the hospital dataset (Excel or CSV) and cleans the column names.
2. Builds ONE scikit-learn Pipeline:
       missing-value imputation -> One-Hot Encoding of categorical columns -> classifier
   Keeping everything in a single Pipeline means the API can pass raw patient
   values straight to the model, with no separate encoder file to keep in sync.
3. Compares Random Forest and Gradient Boosting with cross-validated grid search
   and keeps whichever has the better ROC-AUC.
4. Evaluates the winner on a held-out 20% test set:
   Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix, ROC curve.
5. Saves
       model.pkl     the trained pipeline (used by backend/main.py)
       metrics.json  metrics + schema for the dashboard (served by GET /api/metrics)

Usage
-----
    pip install -r requirements-dev.txt
    python train.py                              # data/hospital_synthetic_dataset.xlsx
    python train.py --data path/to/other.xlsx    # or point at any .xlsx / .csv file
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "hospital_synthetic_dataset.xlsx"
MODEL_PATH = ROOT / "model.pkl"
METRICS_PATH = ROOT / "metrics.json"

TARGET = "heart_disease"  # 0 = No, 1 = Yes
FEATURES = [
    "age",
    "sex",
    "blood_pressure",
    "cholesterol",
    "bmi",
    "smoker",
    "diabetes",
    "physical_activity",
    "family_history",
    "stress_level",
]
# These four are always one-hot encoded (as required by the assignment).
# Any other feature that turns out to contain text is treated the same way.
FORCED_CATEGORICAL = ["sex", "smoker", "diabetes", "family_history"]

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


# --------------------------------------------------------------------------- #
# Data loading and cleaning
# --------------------------------------------------------------------------- #
def normalise_name(name: str) -> str:
    """'Blood Pressure ' -> 'blood_pressure', 'BMI' -> 'bmi'."""
    return re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower()).strip("_")


def clean_category(value):
    """Turn a categorical cell into a tidy string ('Yes', '1', 'Male', ...)."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def to_binary(series: pd.Series) -> pd.Series:
    """Make the target a clean 0/1 column (accepts 0/1, Yes/No, True/False)."""
    text = series.astype(str).str.strip().str.lower()
    mapping = {"1": 1, "1.0": 1, "yes": 1, "y": 1, "true": 1,
               "0": 0, "0.0": 0, "no": 0, "n": 0, "false": 0}
    out = text.map(mapping)
    if out.isna().any():
        bad = sorted(text[out.isna()].unique())[:5]
        raise SystemExit(f"Target column '{TARGET}' has values I cannot read as 0/1: {bad}")
    return out.astype(int)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"Dataset not found: {path}\n"
            f"Put 'hospital_synthetic_dataset.xlsx' in the data/ folder, "
            f"or pass --data <path>."
        )
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise SystemExit(f"Unsupported file type: {path.suffix} (use .xlsx or .csv)")

    df.columns = [normalise_name(c) for c in df.columns]
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise SystemExit(
            f"These columns are missing from the dataset: {missing}\n"
            f"Columns found: {list(df.columns)}"
        )
    df = df[FEATURES + [TARGET]].copy()
    df[TARGET] = to_binary(df[TARGET])
    return df


def split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Decide which features are numeric and which are categorical (in place)."""
    numeric, categorical = [], []
    for col in FEATURES:
        if col in FORCED_CATEGORICAL:
            categorical.append(col)
            continue
        as_number = pd.to_numeric(df[col], errors="coerce")
        filled = df[col].notna().sum()
        if filled and as_number.notna().sum() < 0.5 * filled:
            categorical.append(col)  # mostly text, so treat it as a category
        else:
            df[col] = as_number
            numeric.append(col)
    for col in categorical:
        df[col] = df[col].map(clean_category).astype(object)
    return numeric, categorical


# --------------------------------------------------------------------------- #
# Schema (lets the dashboard build its form from the real data)
# --------------------------------------------------------------------------- #
ORDER_HINTS = ["none", "low", "moderate", "medium", "high", "very high"]


def category_sort_key(option: str):
    lowered = option.lower()
    return (ORDER_HINTS.index(lowered) if lowered in ORDER_HINTS else 99, lowered)


def build_schema(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> dict:
    schema = {"features": FEATURES, "numeric": {}, "categorical": {}}
    for col in numeric:
        values = df[col].dropna()
        is_integer = bool((values == values.round()).all())
        span = float(values.max() - values.min())
        schema["numeric"][col] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": round(float(values.mean()), 2),
            "median": round(float(values.median()), 2),
            "integer": is_integer,
            "step": 1 if (is_integer or span >= 100) else 0.1,
        }
    for col in categorical:
        options = sorted(df[col].dropna().unique().tolist(), key=category_sort_key)
        schema["categorical"][col] = options
    return schema


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_pipeline(numeric: list[str], categorical: list[str], classifier) -> Pipeline:
    """Imputation + One-Hot Encoding + classifier, all in one object."""
    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        # handle_unknown="ignore" keeps the API safe if it ever
                        # receives a category that was not in the training data.
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    return Pipeline([("prep", preprocess), ("clf", classifier)])


def candidate_models() -> dict:
    return {
        "Random Forest": (
            RandomForestClassifier(
                random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
            ),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 6, 10],
                "clf__min_samples_leaf": [1, 3, 5],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [2, 3],
            },
        ),
    }


def aggregate_importance(pipeline: Pipeline, categorical: list[str]) -> list[dict]:
    """Sum one-hot columns back into the original feature (sex_Male + sex_Female -> sex)."""
    names = pipeline.named_steps["prep"].get_feature_names_out()
    importances = pipeline.named_steps["clf"].feature_importances_
    totals: dict[str, float] = {}
    for name, value in zip(names, importances):
        _, _, rest = name.partition("__")
        owner = rest
        for col in sorted(categorical, key=len, reverse=True):
            if rest.startswith(col + "_"):
                owner = col
                break
        totals[owner] = totals.get(owner, 0.0) + float(value)
    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    return [{"feature": k, "importance": round(v, 4)} for k, v in ranked]


def r4(x) -> float:
    return round(float(x), 4)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Train the heart-disease classifier.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="path to .xlsx or .csv")
    args = parser.parse_args()

    print(f"Loading {args.data} ...")
    df = load_data(args.data)
    numeric, categorical = split_columns(df)
    print(f"  {len(df)} patients | numeric: {numeric} | one-hot encoded: {categorical}")

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"  train: {len(X_train)} | test: {len(X_test)} | heart disease rate: {y.mean():.1%}")

    # ---- Pick the best model with cross-validated grid search ------------- #
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results, best_name, best_search = [], None, None
    for name, (estimator, grid) in candidate_models().items():
        print(f"Tuning {name} ...")
        search = GridSearchCV(
            build_pipeline(numeric, categorical, estimator),
            grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=1,
        )
        search.fit(X_train, y_train)
        params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        results.append({"name": name, "cv_roc_auc": r4(search.best_score_), "best_params": params})
        print(f"  CV ROC-AUC {search.best_score_:.4f}  {params}")
        if best_search is None or search.best_score_ > best_search.best_score_:
            best_name, best_search = name, search

    model: Pipeline = best_search.best_estimator_
    print(f"Selected: {best_name}")

    # ---- Cross-validated scores for the winner (training data only) ------- #
    cv_scores = cross_validate(
        clone(model), X_train, y_train, cv=cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
    )
    cv_summary = {
        metric: {"mean": r4(cv_scores[f"test_{metric}"].mean()),
                 "std": r4(cv_scores[f"test_{metric}"].std())}
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]
    }

    # ---- Final evaluation on the untouched test set ----------------------- #
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y_test, proba)

    test_metrics = {
        "accuracy": r4(accuracy_score(y_test, pred)),
        "precision": r4(precision_score(y_test, pred, zero_division=0)),
        "recall": r4(recall_score(y_test, pred, zero_division=0)),
        "f1": r4(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": r4(roc_auc_score(y_test, proba)),
    }

    print("\nTest-set results")
    for key, value in test_metrics.items():
        print(f"  {key:<10} {value:.4f}")
    print(f"  confusion matrix  [[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]")

    # ---- Save the model and the dashboard payload ------------------------- #
    joblib.dump(model, MODEL_PATH, compress=3)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "model": {
            "name": best_name,
            "best_params": next(r["best_params"] for r in results if r["name"] == best_name),
            "candidates": results,
            "cv_folds": CV_FOLDS,
            "cv_scores": cv_summary,
            "threshold": 0.5,
        },
        "dataset": {
            "source": args.data.name,
            "rows": int(len(df)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "class_counts": {"no": int((y == 0).sum()), "yes": int((y == 1).sum())},
            "positive_rate": r4(y.mean()),
        },
        "metrics": test_metrics,
        "confusion_matrix": {
            "labels": ["No heart disease", "Heart disease"],
            "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
        "roc_curve": {"fpr": [r4(v) for v in fpr], "tpr": [r4(v) for v in tpr]},
        "feature_importance": aggregate_importance(model, categorical),
        "schema": build_schema(df, numeric, categorical),
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2))

    print(f"\nSaved {MODEL_PATH.name} ({MODEL_PATH.stat().st_size / 1024:.0f} KB) "
          f"and {METRICS_PATH.name}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            print(exc.code, file=sys.stderr)
        raise

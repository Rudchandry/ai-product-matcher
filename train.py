#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder
import joblib


SEP = " [SEP] "  # separator when concatenating multiple text columns


def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple text pair matcher.")
    parser.add_argument("--csv", required=True, help="Path to CSV file.")
    parser.add_argument(
        "--val_size", type=float, default=0.2, help="Validation split size (0-1)."
    )
    parser.add_argument(
        "--calibration",
        choices=["none", "sigmoid", "isotonic"],
        default="none",
        help="Probability calibration method.",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Upsample each class to at least 2 samples (and balance tiny sets).",
    )
    parser.add_argument(
        "--version",
        default="v0",
        help="Version tag for saving artifacts, e.g., v1",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--force-nonstratify",
        action="store_true",
        help="Force a non-stratified train/validation split even if stratification would be possible.",
    )
    parser.add_argument(
        "--min_val_frac",
        type=float,
        default=0.0,
        help="Minimum validation fraction to enforce for small datasets (0-1). Set 0 to disable.",
    )
    parser.add_argument(
        "--tiny_threshold",
        type=int,
        default=100,
        help="Dataset size below which --min_val_frac is applied.",
    )
    return parser.parse_args()


def detect_columns(df: pd.DataFrame):
    """
    Figure out label + text columns.
    Default: look for label column named 'label' (else last column).
    Text columns: prefer ['text_a','text_b'] if present; else any object-like columns except label.
    """
    label_col = "label" if "label" in df.columns else df.columns[-1]
    feature_cols = [c for c in df.columns if c != label_col]

    preferred_order = [
        "text_a",
        "text_b",
        "left",
        "right",
        "title1",
        "title2",
        "a",
        "b",
        "query",
        "candidate",
        "text",
        "sentence",
        "content",
        "body",
    ]
    preferred_cols = [c for c in preferred_order if c in feature_cols]

    # choose columns
    if {"text_a", "text_b"}.issubset(df.columns):
        text_cols = ["text_a", "text_b"]
    elif len(preferred_cols) >= 2:
        text_cols = preferred_cols[:2]
    elif len(preferred_cols) == 1:
        text_cols = preferred_cols
    else:
        obj_like = [
            c
            for c in feature_cols
            if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c])
        ]
        if not obj_like:
            raise ValueError(
                "Could not find suitable text columns. "
                f"Available columns: {list(df.columns)}; label column: {label_col}"
            )
        text_cols = obj_like[:2] if len(obj_like) >= 2 else obj_like[:1]

    return label_col, text_cols


def ensure_min_two_per_class(df: pd.DataFrame, label_col: str, random_state: int):
    """Upsample each class to at least 2 rows; returns a new dataframe."""
    parts = []
    for _, g in df.groupby(label_col):
        n = max(2, len(g))
        parts.append(g.sample(n=n, replace=True, random_state=random_state))
    return pd.concat(parts, ignore_index=True)


def build_pipeline(calibration: str, random_state: int, cv: int = 3):
    """Create a TF-IDF + LogisticRegression classifier, optionally calibrated."""
    base_clf = LogisticRegression(
        max_iter=2000,
        n_jobs=None,  # ensure compatibility across sklearn versions
        class_weight=None,  # you can switch to "balanced" if you prefer
        random_state=random_state,
    )

    pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", base_clf),
        ]
    )

    if calibration in ("sigmoid", "isotonic"):
        # Wrap the pipeline with a calibrated classifier for probability outputs
        # Use positional first argument to avoid sklearn version differences
        # (some versions expect `base_estimator`, others `estimator`).
        pipe = CalibratedClassifierCV(pipe, method=calibration, cv=cv)

    return pipe


def stringify_features(df: pd.DataFrame, text_cols):
    """Concatenate selected text columns into a single string feature."""
    if len(text_cols) == 1:
        return df[text_cols[0]].astype(str)
    return df[text_cols].astype(str).agg(SEP.join, axis=1)


def main(args):
    print(f"[INFO] Loading data from: {args.csv}")
    df = pd.read_csv(args.csv)

    # Detect columns for your file (text_a, text_b, label)
    label_col, text_cols = detect_columns(df)
    print(f"[INFO] Detected label column: {label_col}")
    print(f"[INFO] Using text columns: {text_cols}")

    # Minimal cleaning: drop rows with missing in required columns
    needed = [label_col] + text_cols
    before = len(df)
    df = df.dropna(subset=needed)
    after = len(df)
    if after < before:
        print(f"[WARN] Dropped {before - after} rows with missing values in {needed}.")

    # Optionally upsample tiny classes (and at least 2 per class)
    if args.balanced:
        df = ensure_min_two_per_class(df, label_col=label_col, random_state=args.random_state)
        print("[INFO] Applied minimal upsampling to ensure ≥2 samples per class.")

    # Build features (X) and labels (y)
    X_text = stringify_features(df, text_cols)
    y_raw = df[label_col]

    # Normalize label values and encode them
    # Map common strings to binary classes; otherwise LabelEncoder handles general case
    y_norm = y_raw.astype(str).str.strip()
    # common mapping for your sample data:
    mapping = {
        "match": "Match",
        "no match": "No match",
        "no_match": "No match",
        "nomatch": "No match",
        "1": "Match",
        "0": "No match",
        "true": "Match",
        "false": "No match",
    }
    y_norm = y_norm.str.lower().map(mapping).fillna(y_raw)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_norm)

    print("[INFO] Class distribution (encoded):", dict(Counter(y)))

    # Safe stratification: only if every class has ≥ 2 samples and not forced off
    class_counts = Counter(y)
    min_count = min(class_counts.values())
    strat = y if (min_count >= 2 and not args.force_nonstratify) else None
    if strat is None and not args.force_nonstratify:
        print(
            "[WARN] Too few samples per class for a stratified split. "
            "Proceeding without stratify; results may be unreliable for very small datasets."
        )
    elif strat is None and args.force_nonstratify:
        print("[INFO] '--force-nonstratify' provided: performing non-stratified split.")

    # Split
    n_samples = len(y)
    num_classes = len(class_counts)

    if strat is None:
        # No stratification possible; fall back to regular split
        X_train, X_val, y_train, y_val = train_test_split(
            X_text,
            y,
            test_size=args.val_size,
            random_state=args.random_state,
            stratify=None,
        )
    else:
        # When stratifying, ensure the test set has at least one sample per class
        # Convert val_size to absolute number of samples if it's a fraction
        if isinstance(args.val_size, float) and 0 < args.val_size < 1:
            test_n = int(np.ceil(args.val_size * n_samples))
        else:
            # If user passed an int-like value, coerce to int
            test_n = int(args.val_size)

        if test_n < num_classes:
            print(
                f"[WARN] Requested val_size results in test set size {test_n} which is < number of classes ({num_classes}). "
                f"Adjusting test set size to {num_classes} to allow stratification."
            )
            test_n = num_classes

        # Ensure at least one training sample remains
        if test_n >= n_samples:
            test_n = max(num_classes, n_samples - 1)
            print(f"[WARN] Adjusted test set size to {test_n} to leave at least one training sample.")

        # Optionally enforce a minimum fraction of data for validation on tiny datasets
        if args.min_val_frac and n_samples < args.tiny_threshold:
            min_test_n = int(np.ceil(args.min_val_frac * n_samples))
            if min_test_n > test_n:
                print(
                    f"[WARN] Enforcing min_val_frac={args.min_val_frac} for tiny dataset (n={n_samples}). "
                    f"Increasing test set size from {test_n} to {min_test_n}.")
                test_n = min_test_n

        X_train, X_val, y_train, y_val = train_test_split(
            X_text,
            y,
            test_size=test_n,
            random_state=args.random_state,
            stratify=strat,
        )

    # Decide on calibration cv based on training set class counts
    effective_calibration = args.calibration
    if args.calibration in ("sigmoid", "isotonic"):
        train_class_counts = Counter(y_train)
        min_train_count = min(train_class_counts.values())
        # For Stratified KFold used internally, n_splits cannot exceed the
        # number of members in any class. Ensure cv between 2 and 3 when possible.
        if min_train_count < 2:
            print(
                "[WARN] Not enough samples per class in the training set for calibration. "
                "Disabling probability calibration. For tiny training sets, prefer '--calibration sigmoid' or '--calibration none'."
            )
            effective_calibration = "none"
            calib_cv = None
        else:
            calib_cv = min(3, min_train_count)

        if effective_calibration == "isotonic" and len(X_train) < 100:
            print(
                "[WARN] Isotonic calibration typically needs more data and may be unreliable on very small training sets. "
                "Use '--calibration sigmoid' or '--calibration none' for more stable results on tiny datasets."
            )

    else:
        calib_cv = None

    # Build model (optionally calibrated)
    if effective_calibration in ("sigmoid", "isotonic"):
        pipe = build_pipeline(effective_calibration, args.random_state, cv=calib_cv)
    else:
        pipe = build_pipeline("none", args.random_state)

    # Train
    print("[INFO] Training...")
    pipe.fit(X_train, y_train)

    # Evaluate
    print("[INFO] Evaluating...")
    y_pred = pipe.predict(X_val)
    if hasattr(pipe, "predict_proba"):
        y_proba = pipe.predict_proba(X_val)[:, 1] if len(label_encoder.classes_) == 2 else None
    else:
        y_proba = None

    acc = accuracy_score(y_val, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_val, y_pred)
    print("\n=== Metrics ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 (wgt):  {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print("\nClassification report:")
    print(classification_report(y_val, y_pred, target_names=list(label_encoder.classes_), zero_division=0))
    print("Confusion matrix:\n", cm)

    # Save artifacts
    out_dir = Path("artifacts") / args.version
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.joblib"
    le_path = out_dir / "label_encoder.joblib"
    meta_path = out_dir / "metadata.json"

    joblib.dump(pipe, model_path)
    joblib.dump(label_encoder, le_path)

    meta = {
        "csv": args.csv,
        "version": args.version,
        "val_size": args.val_size,
        "calibration": args.calibration,
        "balanced": bool(args.balanced),
        "random_state": args.random_state,
        "label_col": label_col,
        "text_cols": text_cols,
        "classes": list(map(str, label_encoder.classes_)),
        "metrics": {
            "accuracy": float(acc),
            "f1_weighted": float(f1),
            "precision_weighted": float(prec),
            "recall_weighted": float(rec),
            "confusion_matrix": cm.tolist(),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[INFO] Saved model to: {model_path}")
    print(f"[INFO] Saved label encoder to: {le_path}")
    print(f"[INFO] Saved metadata to: {meta_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)

"""
Data loading, temporal split, and feature engineering utilities.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


def load_and_split():
    """
    Load Dataset.csv and produce a strict temporal split.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test,
             encoder, feature_names, split_info
    """
    df = pd.read_csv(DATASET_CSV)
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    X_raw = df.loc[:, FEATURE_START:FEATURE_END].copy()
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan).fillna(0)
    feature_names = X_raw.columns.tolist()

    encoder = LabelEncoder()
    encoder.classes_ = np.array(ALL_CLASSES)
    y = encoder.transform(df[LABEL_COL])

    n = len(df)
    i1 = int(n * TRAIN_RATIO)
    i2 = int(n * (TRAIN_RATIO + VAL_RATIO))

    X_train_raw = X_raw.iloc[:i1].values
    X_val_raw   = X_raw.iloc[i1:i2].values
    X_test_raw  = X_raw.iloc[i2:].values
    y_train = y[:i1]
    y_val   = y[i1:i2]
    y_test  = y[i2:]

    # Scalers fitted on train only
    scaler_minmax  = MinMaxScaler().fit(X_train_raw)
    scaler_std     = StandardScaler().fit(X_train_raw)

    split_info = {
        "n_total": n,
        "n_train": i1,
        "n_val":   i2 - i1,
        "n_test":  n - i2,
        "train_label_dist": dict(zip(*np.unique(y_train, return_counts=True))),
        "test_label_dist":  dict(zip(*np.unique(y_test,  return_counts=True))),
    }

    return (X_train_raw, X_val_raw, X_test_raw,
            y_train, y_val, y_test,
            scaler_minmax, scaler_std,
            encoder, feature_names, split_info)


def get_normal_train(X_train, y_train):
    """Return training samples labelled Normal (class 0)."""
    mask = (y_train == 0)
    return X_train[mask]


def stealthy_enrichment(y_full, y_flagged, encoder):
    """
    Compute enrichment factor for stealthy classes after Stage-1 filtering.
    """
    stealthy_ids = [list(encoder.classes_).index(c)
                    for c in STEALTHY_CLASSES
                    if c in encoder.classes_]

    def ratio(labels):
        return np.isin(labels, stealthy_ids).mean()

    r_before = ratio(y_full)
    r_after  = ratio(y_flagged)
    factor   = r_after / r_before if r_before > 0 else float("inf")
    return r_before, r_after, factor

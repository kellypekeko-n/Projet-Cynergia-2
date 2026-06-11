"""
STEP 1 — Build representative TON_IoT sample.
Reads all 23 Network dataset files, samples by class target,
preprocesses features, and saves a single ready-to-use CSV.
"""
import sys, os, io, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ton_iot_config import *

os.makedirs(METRICS_DIR, exist_ok=True)


def build_dataset():
    print("=" * 60)
    print("STEP 1 — Building TON_IoT representative sample")
    print("=" * 60)

    # ── Pass 1: collect full type inventory across all files ──────────────────
    print("\nPass 1: Scanning all files for type counts...")
    global_counts = {}
    file_type_map = {}   # {file_idx: {type: count}}

    for i in range(1, 24):
        path = os.path.join(NET_DIR, f"Network_dataset_{i}.csv")
        df_t = pd.read_csv(path, usecols=["type"])
        cnt  = df_t["type"].value_counts().to_dict()
        file_type_map[i] = cnt
        for k, v in cnt.items():
            global_counts[k] = global_counts.get(k, 0) + v

    print("\nGlobal class counts (22.3M total):")
    for cls, cnt in sorted(global_counts.items(), key=lambda x: -x[1]):
        pct = cnt / sum(global_counts.values()) * 100
        print(f"  {cls:15s}: {cnt:10,}  ({pct:5.2f}%)")

    # ── Pass 2: stratified sampling ──────────────────────────────────────────
    print("\nPass 2: Stratified sampling...")

    # Determine how many samples to draw per class per file
    # proportional to each file's share of the class
    remaining = {cls: SAMPLE_TARGET.get(cls, 0) for cls in global_counts}

    collected = {cls: [] for cls in global_counts}
    rng = np.random.default_rng(SEEDS[0])

    cols_keep = NUMERIC_FEATURES + CAT_FEATURES + ["ts", "type"]

    for i in range(1, 24):
        path = os.path.join(NET_DIR, f"Network_dataset_{i}.csv")
        file_cnt = file_type_map[i]

        # Determine how much to read from this file for each class
        needs_any = any(
            remaining.get(cls, 0) > 0
            for cls in file_cnt
        )
        if not needs_any:
            continue

        print(f"  Reading file {i:2d}...", end=" ")
        df = pd.read_csv(path, usecols=cols_keep)
        df = df.replace("-", np.nan)

        for cls, file_n in file_cnt.items():
            need = remaining.get(cls, 0)
            if need <= 0:
                continue
            subset = df[df["type"] == cls]
            take = min(need, len(subset))
            if take > 0:
                idx = rng.choice(len(subset), size=take, replace=False)
                collected[cls].append(subset.iloc[idx])
                remaining[cls] = remaining.get(cls, 0) - take

        print(f"  remaining: { {k:v for k,v in remaining.items() if v>0} }")

    # Combine all samples
    print("\nCombining samples...")
    dfs = []
    for cls, frames in collected.items():
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            dfs.append(combined)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.sort_values("ts").reset_index(drop=True)

    print(f"\nCombined dataset shape: {df_all.shape}")
    print("\nClass distribution in sample:")
    class_dist = df_all["type"].value_counts()
    for cls, cnt in class_dist.items():
        pct = cnt / len(df_all) * 100
        print(f"  {cls:15s}: {cnt:8,}  ({pct:5.2f}%)")

    # ── Feature engineering ──────────────────────────────────────────────────
    print("\nFeature engineering...")

    # Numeric: fill inf/nan with 0
    for col in NUMERIC_FEATURES:
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors='coerce')
    df_all[NUMERIC_FEATURES] = (df_all[NUMERIC_FEATURES]
                                  .replace([np.inf, -np.inf], np.nan)
                                  .fillna(0))

    # Derived features
    total_bytes  = df_all["src_bytes"] + df_all["dst_bytes"]
    total_pkts   = df_all["src_pkts"]  + df_all["dst_pkts"]
    df_all["bytes_ratio"]      = (df_all["src_bytes"]
                                    / (total_bytes + 1e-9))
    df_all["pkts_ratio"]       = (df_all["src_pkts"]
                                    / (total_pkts + 1e-9))
    df_all["bytes_per_pkt"]    = (total_bytes / (total_pkts + 1e-9))
    df_all["has_dns"]          = (df_all["dns_qtype"] > 0).astype(int)
    df_all["has_http"]         = (df_all["http_status_code"] > 0).astype(int)
    df_all["missed_ratio"]     = (df_all["missed_bytes"]
                                    / (df_all["src_bytes"] + 1e-9))
    df_all["log_src_bytes"]    = np.log1p(df_all["src_bytes"])
    df_all["log_dst_bytes"]    = np.log1p(df_all["dst_bytes"])
    df_all["log_duration"]     = np.log1p(df_all["duration"])

    DERIVED = ["bytes_ratio", "pkts_ratio", "bytes_per_pkt",
               "has_dns", "has_http", "missed_ratio",
               "log_src_bytes", "log_dst_bytes", "log_duration"]

    # Categorical: one-hot encode
    print("  One-hot encoding categorical features...")
    for col in CAT_FEATURES:
        df_all[col] = df_all[col].fillna("unknown").astype(str)
    dummies = pd.get_dummies(df_all[CAT_FEATURES], prefix=CAT_FEATURES,
                              dtype=int)

    ALL_FEATURES = NUMERIC_FEATURES + DERIVED + list(dummies.columns)
    df_features = pd.concat([df_all[["ts", "type"]],
                               df_all[NUMERIC_FEATURES + DERIVED],
                               dummies], axis=1)
    df_features = df_features.replace([np.inf, -np.inf], 0).fillna(0)

    print(f"  Total features: {len(ALL_FEATURES)}")

    # Save
    df_features.to_csv(CACHE_CSV, index=False)
    print(f"\nDataset saved: {CACHE_CSV}")
    print(f"Shape: {df_features.shape}")

    # Save metadata
    meta = {
        "n_total":        int(len(df_features)),
        "n_features":     len(ALL_FEATURES),
        "feature_names":  ALL_FEATURES,
        "class_dist":     {k: int(v) for k, v in class_dist.items()},
        "global_counts":  {k: int(v) for k, v in global_counts.items()},
        "stealthy_classes": STEALTHY_CLASSES,
        "mitre_map":      MITRE_MAP,
    }
    with open(os.path.join(METRICS_DIR, "dataset_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return df_features, ALL_FEATURES, meta


if __name__ == "__main__":
    build_dataset()

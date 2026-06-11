"""
TON_IoT experiment configuration.
Replaces config.py for the TON_IoT pipeline.
"""
import os

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET_DIR     = os.path.join(ROOT, "TON_IOT_Datasets", "TON_IoT datasets",
                           "Processed_datasets", "Processed_Network_dataset")
RESULTS_DIR = os.path.join(ROOT, "experiment", "results")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
LATEX_DIR   = os.path.join(RESULTS_DIR, "latex")
TABLES_DIR  = os.path.join(LATEX_DIR, "tables")
SECTIONS_DIR= os.path.join(LATEX_DIR, "sections")
CACHE_CSV   = os.path.join(METRICS_DIR, "ton_iot_sample.csv")

# Reproducibility
SEEDS    = [42, 123, 456, 789, 1024, 2024, 314, 999, 77, 555]   # 10 seeds for stats
N_SPLITS = 5

# Label column
LABEL_COL  = "type"
TIME_COL   = "ts"

# Attack taxonomy
ALL_CLASSES = ["normal", "backdoor", "ddos", "dos",
               "injection", "mitm", "password", "ransomware",
               "scanning", "xss"]

# Stealthy classes (low-volume, evasive, MITRE-aligned)
STEALTHY_CLASSES = ["scanning", "mitm", "backdoor", "ransomware"]

# MITRE ATT&CK for ICS mapping
MITRE_MAP = {
    "scanning":   "T0840 (Network Sniffing / Reconnaissance)",
    "mitm":       "T0830 (Adversary-in-the-Middle)",
    "backdoor":   "T0807 (Command-Line Interface / Backdoor)",
    "ransomware": "T0826 (Loss of Availability)",
    "dos":        "T0814 (Denial of Service)",
    "ddos":       "T0814 (Denial of Service — Distributed)",
    "injection":  "T0836 (Modify Parameter via Injection)",
    "password":   "T1110 (Brute Force Credentials)",
    "xss":        "T1059 (Command and Scripting Interpreter)",
    "normal":     "N/A",
}

# Sampling targets per class (total ~500K, manageable in RAM)
SAMPLE_TARGET = {
    "normal":    80_000,
    "scanning":  100_000,
    "dos":       60_000,
    "ddos":      100_000,
    "injection": 40_000,
    "password":  60_000,
    "xss":       60_000,
    "backdoor":  30_000,
    "ransomware": 5_000,    # keep most — only 73K total
    "mitm":      1_052,     # keep ALL — extremely rare
}

# Numeric features to use (always present in network flows)
NUMERIC_FEATURES = [
    "duration", "src_bytes", "dst_bytes", "missed_bytes",
    "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes",
    "src_port", "dst_port",
    "dns_qclass", "dns_qtype", "dns_rcode",
    "http_request_body_len", "http_response_body_len", "http_status_code",
]

# Categorical features (will be one-hot encoded)
CAT_FEATURES = ["proto", "service", "conn_state"]

# Train/Val/Test split ratios (chronological)
TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20
TEST_RATIO  = 0.20

# Stage 1 parameters
STAGE1_TARGET_RECALL = 0.90
STAGE1_MAX_FPR       = 0.15

# Figures
FIG_DPI   = 150
PALETTE   = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12",
             "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
             "#e91e63", "#00bcd4"]

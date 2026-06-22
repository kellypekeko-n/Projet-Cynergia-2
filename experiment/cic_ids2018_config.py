"""
Configuration pour CIC-IDS2018 (Canadian Institute for Cybersecurity).
Dataset : https://www.unb.ca/cic/datasets/ids-2018.html
Trafic réseau — 15 classes d'attaques + normal.

Structure des fichiers :
  Processed Traffic Data for ML Algorithms/
    *.csv  — fichiers par jour (Friday-02-03-2018_TrafficForML_CICFlowMeter.csv, etc.)

Différence avec TON_IoT :
  - Features CICFlowMeter (flow-based) vs Zeek (TON_IoT)
  - Pas de features DNS/HTTP spécifiques
  - Colonne label = "Label" (pas "type")
  - Timestamps dans "Timestamp"
"""
import os

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ↓ Adapter ce chemin selon où tu as téléchargé CIC-IDS2018
CIC_DIR     = os.path.join(ROOT, "CIC_IDS2018_Datasets",
                           "Processed Traffic Data for ML Algorithms")

RESULTS_DIR  = os.path.join(ROOT, "experiment", "results_cic")
METRICS_DIR  = os.path.join(RESULTS_DIR, "metrics")
FIGURES_DIR  = os.path.join(RESULTS_DIR, "figures")
LATEX_DIR    = os.path.join(RESULTS_DIR, "latex")
CACHE_CSV    = os.path.join(METRICS_DIR, "cic_ids2018_sample.csv")

SEEDS    = [42, 123, 456, 789, 1024, 2024, 314, 999, 77, 555]
N_SPLITS = 5

LABEL_COL = "Label"
TIME_COL  = "Timestamp"

# Classes CIC-IDS2018 (normaliser les noms)
ALL_CLASSES = [
    "Benign", "DoS-GoldenEye", "DoS-Hulk", "DoS-Slowhttptest", "DoS-Slowloris",
    "DDoS-LOIC-HTTP", "BotNet", "FTP-BruteForce", "SSH-BruteForce",
    "Infiltration", "Web-BruteForce", "Web-XSS", "Web-Sql-Injection",
    "DDOS-HOIC", "Heartbleed",
]
NORMAL_CLASS   = "Benign"
STEALTHY_CLASSES = ["Infiltration", "BotNet", "Heartbleed", "FTP-BruteForce"]

# Mapping MITRE ATT&CK (CIC-IDS2018 → techniques)
MITRE_MAP = {
    "Benign":            "N/A",
    "DoS-GoldenEye":     "T1499 (Endpoint Denial of Service)",
    "DoS-Hulk":          "T1499 (Endpoint Denial of Service)",
    "DoS-Slowhttptest":  "T1499.001 (OS Exhaustion Flood)",
    "DoS-Slowloris":     "T1499.001 (OS Exhaustion Flood)",
    "DDoS-LOIC-HTTP":    "T1498 (Network DoS)",
    "DDOS-HOIC":         "T1498 (Network DoS)",
    "BotNet":            "T1071 (Application Layer Protocol)",
    "FTP-BruteForce":    "T1110 (Brute Force)",
    "SSH-BruteForce":    "T1110.001 (Password Guessing)",
    "Infiltration":      "T1078 (Valid Accounts)",
    "Web-BruteForce":    "T1110 (Brute Force)",
    "Web-XSS":           "T1059 (Command and Scripting Interpreter)",
    "Web-Sql-Injection": "T1190 (Exploit Public-Facing Application)",
    "Heartbleed":        "T1212 (Exploitation for Credential Access)",
}

# Features CICFlowMeter (présentes dans CIC-IDS2018)
NUMERIC_FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Bwd IAT Total", "Bwd IAT Mean",
    "Fwd PSH Flags", "Fwd URG Flags",
    "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s",
    "Packet Length Min", "Packet Length Max", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count",
    "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]
CAT_FEATURES = []  # CIC-IDS2018 n'a pas de features catégorielles natives

SAMPLE_TARGET = {
    "Benign":           100_000,
    "DoS-GoldenEye":     10_000,
    "DoS-Hulk":          50_000,
    "DoS-Slowhttptest":   5_000,
    "DoS-Slowloris":      5_000,
    "DDoS-LOIC-HTTP":    50_000,
    "DDOS-HOIC":         20_000,
    "BotNet":             5_000,
    "FTP-BruteForce":     5_000,
    "SSH-BruteForce":     5_000,
    "Infiltration":       2_000,  # garder tous — rare
    "Web-BruteForce":     5_000,
    "Web-XSS":            3_000,
    "Web-Sql-Injection":  3_000,
    "Heartbleed":           11,   # garder tous — seulement 11 samples
}

TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20
TEST_RATIO  = 0.20

STAGE1_TARGET_RECALL = 0.90
STAGE1_MAX_FPR       = 0.15

FIG_DPI = 150
PALETTE = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12",
           "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
           "#e91e63", "#00bcd4", "#ff5722", "#607d8b",
           "#795548", "#8bc34a", "#ff9800"]

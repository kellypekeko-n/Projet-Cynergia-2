"""
Experiment configuration — centralized constants.
All scripts import from here for reproducibility.
"""
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT, "ICS flow", "archive")
DATASET_CSV = os.path.join(DATASET_DIR, "Dataset.csv")
RESULTS_DIR = os.path.join(ROOT, "experiment", "results")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
LATEX_DIR   = os.path.join(RESULTS_DIR, "latex")
TABLES_DIR  = os.path.join(LATEX_DIR, "tables")
SECTIONS_DIR= os.path.join(LATEX_DIR, "sections")

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEEDS = [42, 123, 456, 789, 1024]
N_SPLITS = 5

# ─── Dataset ──────────────────────────────────────────────────────────────────
FEATURE_START = "duration"
FEATURE_END   = "rAckDelayAvg"
LABEL_COL     = "NST_M_Label"
TIME_COL      = "start"           # Unix timestamp float

STEALTHY_CLASSES = ["ip-scan", "port-scan", "replay"]
ALL_CLASSES      = ["Normal", "ddos", "ip-scan", "mitm", "port-scan", "replay"]

TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20
TEST_RATIO  = 0.20

# ─── Stage 1 ─────────────────────────────────────────────────────────────────
STAGE1_TARGET_RECALL = 0.90
STAGE1_MAX_FPR       = 0.15
THETA_PERCENTILES    = [90, 95, 99]   # percentile of normal scores → threshold

# ─── Stage 2 ─────────────────────────────────────────────────────────────────
OPTUNA_TRIALS = 20     # reduced for speed; increase to 50 for publication
CV_FOLDS      = 5

# ─── Figures ─────────────────────────────────────────────────────────────────
FIG_DPI    = 150
FIG_STYLE  = "seaborn-v0_8-whitegrid"
PALETTE    = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#9b59b6", "#1abc9c",
              "#e67e22", "#34495e"]

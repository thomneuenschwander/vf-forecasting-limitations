"""Every fixed choice of the protocol, in one place.

Section numbers refer to PROTOCOL.md.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Paths
RAW_DIR = ROOT / "data" / "raw"
ALLDATA_JSON = RAW_DIR / "alldata.json"
COORD_CSV = RAW_DIR / "Coord_242.csv"
SERIES_SUMMARY_CSV = RAW_DIR / "Series_Summary.csv"

OUT_DIR = ROOT / "outputs"
EXAMS_PARQUET = OUT_DIR / "exams.parquet"  # filtered cohort + fold
SAMPLES_PARQUET = OUT_DIR / "samples.parquet"  # one row per (history -> next exam)
GRID_PARQUET = OUT_DIR / "grid.parquet"  # geometry of the 54 locations
FILTER_LOG_CSV = OUT_DIR / "filter_log.csv"
PREDS_DIR = OUT_DIR / "preds"  # one parquet per model: sample_id, fold, pred (52 dB)

# 24-2 grid
N_LOCATIONS = 54
GRID_SHAPE = (8, 9)
BLIND_SPOT_INDEX = (25, 34)  # LocID 26 and 35 (Cluster == 0 in Coord_242.csv)
TESTABLE_INDEX = tuple(i for i in range(N_LOCATIONS) if i not in BLIND_SPOT_INDEX)
N_TESTABLE = len(TESTABLE_INDEX)  # 52
SENS_MIN, SENS_MAX = 0.0, 50.0  # range observed in UWHVF (dB)

# Cohort filters (PROTOCOL.md section 2), applied in this order
MIN_BASELINE_AGE_YEARS = 1.0  # F2
MIN_GAP_MONTHS = 3.0  # F3
MAX_GAP_MONTHS = 36.0  # F4
MIN_EXAMS_PER_SERIES = 3  # F5

# Samples (section 3): expanding window, starting with 2 exams of history
MIN_HISTORY = 2

# Cross-validation (section 3)
N_FOLDS = 5
SEED = 20260101

# Progression groups (section 5): MTD slope in dB/year
SLOPE_RAPID = -1.0
SLOPE_MODERATE = -0.5
PROGRESSION_ALPHA = 0.05

# Bootstrap (section 5)
N_BOOTSTRAP = 2_000

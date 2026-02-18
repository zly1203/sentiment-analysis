import os

# === Directory Structure ===
DIRS = {
    "raw": "./01_raw_inbox",
    "staging": "./02_staging",
    "review": "./03_manual_review",
    "gold": "./04_gold_standard",
    "test_lockbox": "./05_test_lockbox",
    "models": "./06_models",
    "workspace": "./07_daily_workspace",
    "reports": "./08_client_reports",
    "baseline": "./09_internal_baseline",
}

for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

# === Model Config ===
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 128
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
LABEL_OPTIONS = ["positive", "negative", "neutral"]

# === Inference Config ===
DEFAULT_CONFIDENCE_THRESHOLD = 0.85

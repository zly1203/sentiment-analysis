# Architecture & Operations Guide

## 1. System Overview

The pipeline has two phases that share the same directory structure but run independently:

| Phase | Components | Purpose |
|-------|-----------|---------|
| **Cold Start** | Notebook 00, Notebook 01, Simulate_Human, Notebook 02 | Clean data → GPT labeling → gold accumulation → first model |
| **Daily Ops** | app.py (or Notebook 03) | Production: model inference → human review → report + gold |

The two phases connect at **04_gold_standard/** — cold start writes initial gold data, daily ops adds new corrections, and Notebook 02 consumes all gold to retrain.

---

## 2. Cold-Start Flow

Run once (or for each new batch) when no model exists or when bootstrapping new data.

### Step-by-step

```
01_raw_inbox/cold_start/source.csv
        │
        ▼  Notebook 00: Clean & standardize → raw.csv
        │
01_raw_inbox/cold_start/raw.csv
        │
        ▼  Notebook 01: GPT-4o-mini labels each row
        │
   ┌────┴────┐
   │         │
 match     conflict
   │         │
   ▼         ▼
04_gold    03_review ──→ Simulate_Human ──→ 04_gold
(auto)                   (expert rules)    (corrected)
   │
   └──→ 09_baseline (full internal copy)
```

### Procedure

| Step | Action | Input | Output |
|------|--------|-------|--------|
| 1 | Place source CSV | — | `01_raw_inbox/cold_start/source.csv` |
| 2 | Run Notebook 00 | Source CSV | `01_raw_inbox/cold_start/raw.csv` (cleaned) |
| 3 | Run Notebook 01 | Raw CSV | `04_gold/...auto.csv` + `03_review/...review.csv` + `09_baseline/...rows.csv` |
| 4 | Run Simulate_Human | `03_review/...review.csv` | `04_gold/...corrected.csv` |
| 5 | Run Notebook 02 | All `04_gold/*.csv` | `06_models/v_{date}_*/` + `05_test_lockbox/final_test_set.csv` |

For multiple batches, repeat steps 1–4 for each batch, then run step 5 once. Notebook 02 merges all gold files.

### Key behaviors

- **Notebook 00** normalizes column names, auto-detects text column from aliases (`text`, `content`, `body`, `comment`, `review`, `tweet`), drops empty/duplicate rows, strips whitespace, and removes pandas index artifacts.
- **Notebook 01** tracks GPT API failures with a counter; if failures occur, a warning is printed with the count.
- **Simulate_Human** applies keyword-based rules (negative > ad > news > AI fallback) and writes only to gold, never to 07.
- **Notebook 02** is safe to re-run: it detects already-processed data by checking both the test lockbox and staging, so re-running with the same gold data produces identical output.

---

## 3. Training Flow

Notebook 02 runs in three phases (one cell each):

### Phase 1: Data Preparation

```
04_gold_standard/*.csv
        │
        ▼  Load all → deduplicate by text (keep last) → lowercase labels
        │
   ┌────┴────────────────────┐
   │                         │
   │  New data only          │  ALL gold minus test
   │  (not in test+staging)  │
   │         │               │
   │    10% append           │
   │    to lockbox           │
   │         │               │
   ▼         ▼               ▼
02_staging  05_test_lockbox  (training set written to staging)
```

**Test lockbox rules:**
- Created once with an initial 10% random split (`random_state=42`)
- On subsequent runs, only genuinely new data is split; 10% is appended to the lockbox
- "New" means not in test lockbox AND not in previous staging (prevents re-run inflation)
- Existing lockbox rows are never removed or reshuffled
- Training set = ALL gold data minus the full test lockbox (always trains on complete history)

**Empty guard:** If no new training data exists, Phase 1 raises `ValueError` to prevent training on zero rows.

### Phase 2: Tokenization

- Reads `02_staging/ready_for_training.csv` and `05_test_lockbox/final_test_set.csv`
- Applies `.str.lower()` before `LABEL_MAP` to prevent case-mismatch silent drops
- Tokenizes with `distilbert-base-uncased`, `max_length=128`

### Phase 3: Training & Save

- Fine-tunes DistilBERT (3 epochs, lr=2e-5, batch_size=8)
- Saves model + tokenizer to `06_models/v_{date}_{project}/`
- Writes `model_meta.json` with accuracy, sample counts, timestamps
- Cleans up `./results_temp/` intermediate checkpoints

---

## 4. Daily Operations Flow

### Via app.py (recommended)

```
User Upload (CSV/Excel)
        │
        ▼  Tab 1: Clean + AI Inference (batched, with progress bar)
        │
   ┌────┴────┐
   │         │
 auto-pass  review queue
 (≥ threshold)  (< threshold)
   │         │
   │    Both saved to disk immediately (crash recovery):
   │    07_workspace/_auto.csv + 03_review/_pending.csv
   │         │
   │    Tab 2: Human edits labels in data_editor
   │         │
   │    Commit detects changes by comparing against pre-edit snapshot
   │         │
   │    ┌────┴────┐
   │    │         │
   │  edited    unedited
   │    │         │
   │    ▼         ▼
   │  04_gold   03_review
   │  (human)   (pending writeback)
   │    │         │
   └────┴─────────┘
                │
                ▼
        08_client_reports
        (Report with label_source: auto/human/model_pending)
```

**Key app behaviors:**
- Threshold is locked after inference to prevent slider changes from losing edits
- Gold receives ONLY rows where `predicted_label` was actually changed (detected via snapshot comparison)
- Unedited review rows are written back to `03_manual_review/` as pending
- Tab 2 can load pending files from previous sessions (shown when no data is loaded)
- If all rows auto-pass, a "Generate Report" button is shown (no gold generated since no human input)
- Double-commit is prevented within a session via `committed` flag

### Via Notebook 03

Same flow, file-based:
- **Stage 1**: Inference → split to `07_workspace/_auto.csv` and `03_review/_review.csv`
- **Stage 2**: User manually reviews, saves as `*corrected*.csv` in `07_daily_workspace/`
  - Rows with `label` filled → gold (human-reviewed)
  - Rows with `label` empty → pending (written back to 03)
  - Full report with `label_source` to `08_client_reports/`

---

## 5. Directory Responsibilities

| Directory | Cold Start | Daily Ops | Written By |
|-----------|-----------|-----------|------------|
| `01_raw_inbox/` | Place raw data | Place new daily data | User |
| `02_staging/` | Training data snapshot | — | Notebook 02 |
| `03_manual_review/` | Review queue from NB01 | Review queue + pending writeback | NB01, NB03, app.py |
| `04_gold_standard/` | auto + corrected labels | Human corrections only | NB01, Simulate, NB03, app.py |
| `05_test_lockbox/` | Initial test set | Grows with new data | Notebook 02 |
| `06_models/` | Trained models | Read-only | Notebook 02 |
| `07_daily_workspace/` | — | Auto-pass results | NB03 Stage 1, app.py |
| `08_client_reports/` | — | Final delivery reports | NB03 Stage 2, app.py |
| `09_internal_baseline/` | Full GPT-labeled copy | — | Notebook 01 |

---

## 6. Design Decisions

### Why append-only test lockbox?

If the test set is reshuffled on each training run, model accuracy comparisons across versions are meaningless — you're evaluating on different data each time. The append-only approach guarantees that all models are evaluated on at least the same core set of test samples.

### Why lock threshold after inference?

Streamlit re-runs the entire script on every widget interaction. Without locking, moving the confidence slider while editing labels in Tab 2 would re-split the data and discard all unsaved edits.

### Why detect edits via snapshot comparison?

Previous design wrote ALL displayed rows to gold, polluting it with unreviewed model predictions. By storing a pre-edit snapshot of `(row_id, predicted_label)` and comparing at commit time, only rows the human actually changed are marked `human` and sent to gold.

### Why write pending rows back?

Without writeback, low-confidence rows that a human didn't review in a session would be lost. Writing them back to `03_manual_review/` as `_pending.csv` ensures they can be loaded and reviewed in a future session (via app.py's "Load Pending" or via Notebook 03).

### Why `label_source` in reports?

Client reports contain all rows regardless of review status. The `label_source` column (`auto`, `human`, `model_pending`) lets downstream consumers know the provenance and confidence of each label.

---

## 7. Retraining Trigger

The app's Knowledge Base tab (Tab 3) counts gold-standard rows created after the current model's `created_at` timestamp. When this count grows, it suggests retraining:

1. Run Notebook 02 (all three cells)
2. Click 🔄 in app sidebar to hot-reload the new model
3. Verify accuracy improved in sidebar display

---

## 8. Configuration

All shared settings live in `config.py`:

```python
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

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
LABEL_OPTIONS = ["positive", "negative", "neutral"]
DEFAULT_CONFIDENCE_THRESHOLD = 0.85
```

Notebook 03 uses a local `CONFIDENCE_THRESHOLD = 0.7` which can be adjusted per run. The app uses `DEFAULT_CONFIDENCE_THRESHOLD` from config as the slider default.

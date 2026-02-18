# Data Specification

CSV schemas, file naming conventions, and column reference for the SentiFlow pipeline.

---

## 1. File Naming Conventions

**Date:** `YYYYMMDD` (e.g. `20260213`)
**Time:** `HHMMSS` (e.g. `143022`)
**Project:** Lowercase identifier (e.g. `cold_start`, `batch_001`)

| Pattern | Example | Written By |
|---------|---------|------------|
| `{DATE}_{PROJECT}_auto.csv` | `20260213_cold_start_auto.csv` | NB01 → gold, NB03/app → workspace |
| `{DATE}_{PROJECT}_review.csv` | `20260213_cold_start_review.csv` | NB01, NB03 Stage 1 |
| `{DATE}_{PROJECT}_pending.csv` | `20260213_cold_start_pending.csv` | NB03 Stage 2, app.py |
| `{DATE}_{PROJECT}_corrected.csv` | `20260213_cold_start_corrected.csv` | NB03 Stage 2, Simulate_Human |
| `{DATE}_{PROJECT}_Final_Report.csv` | `20260213_cold_start_Final_Report.csv` | NB03 Stage 2 |
| `{DATE}_{PROJECT}_{TIME}_Report.csv` | `20260213_cold_start_143022_Report.csv` | app.py |
| `{DATE}_{PROJECT}_{TIME}_gold.csv` | `20260213_cold_start_143022_gold.csv` | app.py |
| `{DATE}_{PROJECT}_{ROWS}rows.csv` | `20260213_cold_start_1069rows.csv` | NB01 → baseline |
| `v_{DATE}_{PROJECT}/` | `v_20260213_cold_start/` | NB02 → model dir |
| `ready_for_training.csv` | — | NB02 → staging |
| `final_test_set.csv` | — | NB02 → test lockbox |

---

## 2. Schemas by Directory

### `01_raw_inbox/` — Raw Input

User-provided data. Minimum requirement: one text column.

| Column | Type | Required | Notes |
|--------|------|----------|-------|
| `text` | str | Yes | Aliases accepted: `content`, `body`, `comment`, `review`, `tweet` |
| `sentiment` | str | No | Pre-existing label; used by NB01 for conflict detection |
| *(other)* | any | No | Preserved through the pipeline |

### `02_staging/` — Training Data

Single file: `ready_for_training.csv`

| Column | Type | Values |
|--------|------|--------|
| `text` | str | — |
| `label` | str | `positive`, `negative`, `neutral` |

Deduplicated by `text` (keep last). Excludes test lockbox rows.

### `03_manual_review/` — Review Queue

**Review files** (`*_review.csv`):

| Column | Type | Notes |
|--------|------|-------|
| *(all original)* | — | Passed through from raw input |
| `predicted_label` | str | Model prediction (NB03/app); NB01 review files use `gpt_label` instead |
| `confidence` | float | Model confidence (NB03/app only) |

**Pending files** (`*_pending.csv`):

Same schema as review files. The `label` and `label_source` columns are stripped before writeback.

### `04_gold_standard/` — Approved Labels

**Standardized schema — all files must match:**

| Column | Type | Description |
|--------|------|-------------|
| `text` | str | The text content |
| `label` | str | Approved label (`positive`, `negative`, `neutral`) |
| `gold_origin` | str | Provenance tag (see below) |
| `created_at` | str | ISO 8601 timestamp |

**`gold_origin` values:**

| Value | Meaning | Origin |
|-------|---------|--------|
| `cold_start_auto` | GPT label matched existing sentiment | Notebook 01 |
| `cold_start_corrected` | Expert-rule or human correction | Simulate_Human |
| `daily_corrected` | Human correction in daily ops | NB03 Stage 2, app.py |

### `05_test_lockbox/` — Immutable Test Set

Single file: `final_test_set.csv`

| Column | Type |
|--------|------|
| `text` | str |
| `label` | str |

**Rules:** Created once (10% split). New data is appended on retraining runs. Existing rows are never removed or reshuffled.

### `06_models/` — Model Artifacts

```
v_{DATE}_{PROJECT}/
  ├── config.json
  ├── model.safetensors
  ├── tokenizer.json, tokenizer_config.json, special_tokens_map.json, vocab.txt
  └── model_meta.json
```

**`model_meta.json`:**

| Field | Type | Description |
|-------|------|-------------|
| `model_name` | str | Directory name |
| `accuracy` | float | Eval accuracy (4 decimal places) |
| `train_samples` | int | Training row count |
| `test_samples` | int | Test row count |
| `base_model` | str | `distilbert-base-uncased` |
| `max_length` | int | `128` |
| `created_at` | str | ISO 8601 timestamp |

**Selection rule:** Highest `accuracy`. Falls back to newest directory if no metadata exists.

### `07_daily_workspace/` — Daily Auto-Pass

| Column | Type | Notes |
|--------|------|-------|
| *(all original)* | — | From raw input |
| `predicted_label` | str | Model prediction |
| `confidence` | float | Score ≥ threshold |

Temporary. Consumed by Stage 2 / app commit.

### `08_client_reports/` — Final Delivery

| Column | Type | Description |
|--------|------|-------------|
| `text` | str | The text content |
| `label` | str | Final label (human correction or model prediction) |
| `label_source` | str | `auto`, `human`, or `model_pending` |
| `confidence` | float | Model confidence score |
| *(other)* | — | Varies by input |

**`label_source` values:**

| Value | Meaning |
|-------|---------|
| `auto` | High-confidence model prediction, no human review |
| `human` | Human-reviewed and corrected |
| `model_pending` | Low-confidence model prediction, not yet reviewed |

### `09_internal_baseline/` — Cold-Start Reference

| Column | Type | Description |
|--------|------|-------------|
| *(all original)* | — | From raw input |
| `global_uuid` | str | UUID4 row identifier |
| `batch_source` | str | `cold_start` |
| `gpt_label` | str | GPT-4o-mini prediction |
| `status` | str | `auto_pass` or `needs_review` |

---

## 3. Column Reference

| Column | Meaning | Where Used |
|--------|---------|------------|
| `text` | Text to classify (canonical name) | Everywhere |
| `content` | Alias for `text` | Raw input (auto-renamed) |
| `sentiment` | Pre-existing label from source | `01_raw_inbox` |
| `gpt_label` | GPT-4o-mini prediction | Notebook 01 only |
| `predicted_label` | DistilBERT model prediction | NB03, app.py |
| `confidence` | Model confidence score (0.0–1.0) | NB03, app.py |
| `label` | Final approved label | Gold standard, reports |
| `label_source` | Provenance: `auto` / `human` / `model_pending` | Reports |
| `gold_origin` | Gold provenance tag | Gold standard |
| `created_at` | ISO 8601 timestamp | Gold standard |
| `global_uuid` | Row UUID4 (NB01) | `01_raw_inbox`, `09_baseline` |
| `row_id` | Row UUID4 (app.py) | app.py session |
| `batch_source` | Batch identifier | Notebook 01 |
| `status` | `auto_pass` / `needs_review` | Notebook 01 |

---

## 4. Configuration Reference

From `config.py`:

| Setting | Value | Used By |
|---------|-------|---------|
| `MODEL_NAME` | `distilbert-base-uncased` | NB02 (training + tokenization) |
| `MAX_LENGTH` | `128` | All inference and training |
| `LABEL_MAP` | `{negative: 0, neutral: 1, positive: 2}` | NB02 (label encoding) |
| `LABEL_OPTIONS` | `[positive, negative, neutral]` | app.py (editor dropdown) |
| `DEFAULT_CONFIDENCE_THRESHOLD` | `0.85` | app.py (slider default) |

Notebook 03 uses a local `CONFIDENCE_THRESHOLD = 0.7` adjustable per run.

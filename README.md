# SentiFlow — Adaptive Sentiment Analysis Pipeline

A human-in-the-loop sentiment analysis system that combines GPT-4o-mini labeling, DistilBERT fine-tuning, and a Streamlit review interface. Designed for iterative model improvement through continuous gold-standard data accumulation.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key (for cold-start GPT labeling only)

## Setup

```bash
# Install dependencies
uv sync

# Configure environment
cp env.example .env
# Edit .env and set OPENAI_API_KEY=your_key_here
```

> `.env` contains secrets and is git-ignored.

## Quick Start

### Cold Start (first time, no model yet)

```bash
# 1. Place raw data
#    Put your source CSV in: 01_raw_inbox/cold_start/

# 2. Run notebooks in order:
#    Notebook 00 → Clean & standardize raw data into raw.csv
#    Notebook 01 → GPT labels + conflict detection
#    Simulate_Human.ipynb → Expert-rule labeling for review items
#    Notebook 02 → Train DistilBERT model

# 3. Launch the app
streamlit run app.py
```

### Daily Operations (model exists)

```bash
# Option A: Streamlit app (recommended)
streamlit run app.py
#   Tab 1: Upload CSV → AI inference → auto/review split
#   Tab 2: Fix low-confidence labels → Commit → Report + Gold

# Option B: Notebooks
#   Notebook 03 Stage 1: Predict & triage
#   Notebook 03 Stage 2: Merge & deliver
```

### Retraining

When enough new gold data accumulates (Tab 3 shows the count), run Notebook 02 to retrain. The app's 🔄 button hot-reloads the new model.

## Project Structure

```
├── config.py                 # Shared configuration (dirs, model params, thresholds)
├── app.py                    # Streamlit production UI
│
├── 00_data_cleaning.ipynb             # Data cleaning & standardization
├── 01_pipeline_setup_and_ingest.ipynb   # Cold-start: GPT labeling
├── 02_model_training_loop.ipynb         # Model training & evaluation
├── 03_daily_operations.ipynb            # Daily predict → review → deliver
├── Simulate_Human.ipynb                 # Expert-rule labeling (cold-start helper)
│
├── 01_raw_inbox/             # Raw data input
├── 02_staging/               # Training-ready data
├── 03_manual_review/         # Human review queue + pending items
├── 04_gold_standard/         # Approved labels (training fuel)
├── 05_test_lockbox/          # Immutable test set (append-only)
├── 06_models/                # Trained model versions + metadata
├── 07_daily_workspace/       # Daily auto-pass results
├── 08_client_reports/        # Final delivery reports
└── 09_internal_baseline/     # Cold-start reference data (internal)
```

## Documentation

| Document | Contents |
|----------|----------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design, data flow diagrams, operational procedures, design decisions |
| [DATA_SPEC.md](DATA_SPEC.md) | CSV schemas, file naming conventions, column reference |

## Key Design Decisions

- **Test lockbox**: Append-only test set that grows but never reshuffles, ensuring fair model comparison across versions.
- **Gold schema**: All gold-standard files use `text, label, gold_origin, created_at` regardless of origin.
- **Model selection**: Best model chosen by accuracy from `model_meta.json`, not by recency.
- **Partial review**: Human reviewers can label a subset; unreviewed rows fall back to model predictions for delivery and get written back as pending.
- **label_source tracking**: Every report row is tagged `auto`, `human`, or `model_pending` for full provenance.

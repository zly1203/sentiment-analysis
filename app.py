import streamlit as st
import pandas as pd
import glob
import os
import json
import datetime
import time
import uuid
import traceback

try:
    import graphviz
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False

from transformers import pipeline as hf_pipeline
from config import DIRS, MAX_LENGTH, DEFAULT_CONFIDENCE_THRESHOLD, LABEL_OPTIONS

# ================= 1. Page Config & Session State =================

st.set_page_config(
    page_title="SentiFlow | Adaptive Engine",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

TODAY = datetime.date.today().strftime("%Y%m%d")

# Session state defaults — initialized once per session
_DEFAULTS = {
    "full_scored_df": None,           # Scored DataFrame from inference or loaded pending
    "project_name": "batch_001",      # Current batch/project identifier
    "audit_log": {"raw": 0, "removed": 0, "clean": 0, "msg": ""},
    "locked_threshold": None,         # Threshold locked at inference time (slider-proof)
    "review_snapshot": None,          # Pre-edit labels for detecting human changes
    "committed": False,               # Prevents double-commit within a session
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ================= 2. Core Functions =================

@st.cache_resource
def load_ai_model():
    """Load best model by accuracy from model_meta.json, fall back to newest."""
    model_dirs = sorted(glob.glob(f"{DIRS['models']}/v_*"))
    if not model_dirs:
        return None, "No models found.", None

    best_dir = None
    best_acc = -1
    best_meta = None
    for d in model_dirs:
        meta_path = os.path.join(d, "model_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("accuracy", 0) > best_acc:
                best_acc = meta["accuracy"]
                best_dir = d
                best_meta = meta

    # Fallback: no model has metadata
    if best_dir is None:
        best_dir = model_dirs[-1]

    try:
        clf = hf_pipeline("text-classification", model=best_dir, tokenizer=best_dir, top_k=None)
        return clf, os.path.basename(best_dir), best_meta
    except Exception as e:
        return None, str(e), None


def clean_incoming_data(df):
    """Standardize columns, remove empty rows, assign UUIDs."""
    report = {"raw": len(df), "removed": 0, "clean": 0, "msg": ""}
    try:
        df.columns = df.columns.astype(str).str.strip().str.lower()
        rename_map = {
            "content": "text", "body": "text", "comment": "text",
            "review": "text", "tweet": "text",
        }
        df.rename(columns=rename_map, inplace=True)

        if "text" not in df.columns:
            return None, "Error: No 'text' column found.", report

        df["text"] = df["text"].astype(str)
        df_clean = df[df["text"].str.strip() != ""].copy()
        df_clean = df_clean[df_clean["text"].str.lower() != "nan"]

        report["clean"] = len(df_clean)
        report["removed"] = report["raw"] - report["clean"]
        report["msg"] = (
            f"Removed {report['removed']} rows (Empty/NaN)."
            if report["removed"] > 0
            else "Data is clean."
        )

        if "row_id" not in df_clean.columns:
            df_clean["row_id"] = [str(uuid.uuid4()) for _ in range(len(df_clean))]

        return df_clean, "Success", report
    except Exception as e:
        return None, f"Cleaning Error: {e}", report


def split_data_by_threshold(df, threshold):
    """Split scored DataFrame into auto-pass and review queues."""
    mask = df["confidence"] >= threshold
    return df[mask].copy(), df[~mask].copy()


def get_new_gold_count_since_model(meta):
    """Count gold-standard rows created after the current model was trained."""
    if not meta or "created_at" not in meta:
        return None
    model_time = datetime.datetime.fromisoformat(meta["created_at"])
    count = 0
    for f in glob.glob(f"{DIRS['gold']}/*.csv"):
        if datetime.datetime.fromtimestamp(os.path.getmtime(f)) > model_time:
            try:
                count += len(pd.read_csv(f))
            except Exception:
                pass
    return count


def build_report(df_auto, df_review_final):
    """Merge auto-pass and review data into a single report DataFrame.

    Returns a DataFrame with deterministic column order and label_source.
    """
    parts = []

    if not df_auto.empty:
        auto_part = df_auto.copy()
        auto_part.rename(columns={"predicted_label": "label"}, inplace=True)
        auto_part["label_source"] = "auto"
        parts.append(auto_part)

    if not df_review_final.empty:
        review_part = df_review_final.copy()
        review_part.rename(columns={"predicted_label": "label"}, inplace=True)
        parts.append(review_part)

    if not parts:
        return pd.DataFrame()

    df_report = pd.concat(parts, ignore_index=True)

    # Deterministic column order: key columns first, then alphabetical
    priority = ["text", "label", "label_source", "confidence"]
    ordered = [c for c in priority if c in df_report.columns]
    ordered += sorted(c for c in df_report.columns if c not in priority)
    return df_report[ordered]


def commit_core(df_review, edited_display, proj):
    """Shared commit logic: merge edits, detect changes, save gold + pending.

    Returns (human_rows, pending_rows, gold_count).
    """
    ts = datetime.datetime.now().strftime("%H%M%S")

    # 1. Merge edits into full review set (UUID-based)
    final_review = df_review.copy().set_index("row_id")
    final_review.update(edited_display.set_index("row_id"))
    final_review.reset_index(inplace=True)

    # 2. Detect which rows were human-edited
    snapshot = st.session_state.get("review_snapshot")
    if snapshot is not None and not snapshot.empty:
        snap_map = dict(zip(snapshot["row_id"], snapshot["predicted_label"]))
        final_review["label_source"] = final_review.apply(
            lambda r: (
                "human"
                if r["predicted_label"] != snap_map.get(r["row_id"], "")
                else "model_pending"
            ),
            axis=1,
        )
    else:
        final_review["label_source"] = "model_pending"

    # 3. Save gold (human-edited rows only)
    human_rows = final_review[final_review["label_source"] == "human"]
    gold_count = 0
    if not human_rows.empty:
        gold_path = f"{DIRS['gold']}/{TODAY}_{proj}_{ts}_gold.csv"
        gold_df = human_rows[["text", "predicted_label"]].copy()
        gold_df.rename(columns={"predicted_label": "label"}, inplace=True)
        gold_df["gold_origin"] = "daily_corrected"
        gold_df["created_at"] = datetime.datetime.now().isoformat()
        gold_df.to_csv(gold_path, index=False)
        gold_count = len(gold_df)

    # 4. Write unedited rows back to pending
    pending_rows = final_review[final_review["label_source"] == "model_pending"]
    if not pending_rows.empty:
        pending_save = pending_rows.drop(columns=["label_source"], errors="ignore")
        pending_save.to_csv(
            f"{DIRS['review']}/{TODAY}_{proj}_pending.csv", index=False,
        )

    return final_review, human_rows, pending_rows, gold_count


# ================= 3. Sidebar =================

with st.sidebar:
    st.title("SentiFlow")
    st.caption("Adaptive Sentiment Engine")

    st.markdown("---")
    st.markdown("### Model Status")

    col_txt, col_btn = st.columns([3, 1])
    with col_txt:
        st.write("AI Model:")
    with col_btn:
        if st.button("🔄", help="Refresh: reload the latest model."):
            st.cache_resource.clear()
            st.rerun()

    classifier, model_ver, model_meta = load_ai_model()
    if classifier:
        st.success("**ONLINE**")
        st.info(f"Serving: `{model_ver}`")
        if model_meta:
            st.caption(f"Accuracy: {model_meta['accuracy']:.2%}")
    else:
        st.error("OFFLINE")
        st.warning("No model available. Please contact the admin.")

    st.markdown("---")
    st.markdown("### Settings")
    conf_threshold = st.slider(
        "Confidence Threshold",
        0.50, 0.99, DEFAULT_CONFIDENCE_THRESHOLD, 0.01,
        help="Rows above this threshold are auto-classified. Lower = fewer rows to review.",
    )

    # Live preview of threshold impact (informational only)
    if st.session_state["full_scored_df"] is not None:
        locked = st.session_state["locked_threshold"]
        if locked is not None and abs(locked - conf_threshold) > 0.005:
            st.caption(f"Set to {locked:.2f} for current batch. Upload new data to use {conf_threshold:.2f}.")
        preview_threshold = locked if locked is not None else conf_threshold
        df_preview = st.session_state["full_scored_df"]
        n_auto = int((df_preview["confidence"] >= preview_threshold).sum())
        st.caption(f"High confidence: {n_auto} | To review: {len(df_preview) - n_auto}")

    st.markdown("---")
    st.markdown("### How It Works")
    st.markdown(
        """
    1. **Upload** your data.
    2. AI **analyzes** each row.
    3. You **review** uncertain ones.
    4. **Download** your report.
    """
    )


# ================= 4. Main Header =================

st.markdown("# SentiFlow")
st.markdown("### AI-Powered Sentiment Analysis with Human Review")

with st.expander("View System Architecture", expanded=False):
    if HAS_GRAPHVIZ:
        graph = graphviz.Digraph()
        graph.attr(rankdir="LR", size="12,5", dpi="72")
        graph.node("A", "Raw Data", shape="folder")
        graph.node("B", "AI Inference", shape="component", style="filled", fillcolor="#e1f5fe")
        graph.node("C", "Confidence\nCheck", shape="diamond", style="filled", fillcolor="#fff9c4")
        graph.node("D", "Auto-Pass", shape="folder", style="filled", fillcolor="#c8e6c9")
        graph.node("E", "Human Review", shape="note", style="filled", fillcolor="#ffccbc")
        graph.node("F", "Verified Data", shape="cylinder", style="filled", fillcolor="#fff59d")
        graph.node("G", "Model Update", shape="box3d")
        graph.edge("A", "B")
        graph.edge("B", "C")
        graph.edge("C", "D", label="High Conf")
        graph.edge("C", "E", label="Low Conf")
        graph.edge("E", "F", label="Corrected")
        graph.edge("F", "G", label="Learn", style="dashed")
        graph.edge("G", "B", label="Deploy", style="dashed")
        st.graphviz_chart(graph)
    else:
        st.info("Install `graphviz` to see the flow diagram.")


# ================= 5. Tabs =================

tab_ingest, tab_review, tab_knowledge = st.tabs(
    ["1️⃣ Upload & Analyze", "2️⃣ Review & Edit", "3️⃣ History"]
)

# --------------- TAB 1: Ingest & Analyze ---------------

with tab_ingest:
    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.subheader("Upload Data")
        proj = st.text_input("Batch Name", st.session_state["project_name"])
        uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

        if not classifier:
            st.warning("No model available. Please contact the admin.")
        elif uploaded:
            if st.button("⚡ Analyze", type="primary"):
                with st.status("⚙️ Analyzing...", expanded=True) as status:
                    try:
                        # Step 1: Clean data
                        st.write("1. Cleaning data...")
                        raw = (
                            pd.read_csv(uploaded)
                            if uploaded.name.endswith(".csv")
                            else pd.read_excel(uploaded)
                        )
                        clean_df, msg, report = clean_incoming_data(raw)

                        if clean_df is None:
                            status.update(label="Error", state="error")
                            st.error(msg)
                            st.stop()

                        if clean_df.empty:
                            status.update(label="Error", state="error")
                            st.error("All rows were empty or NaN. Nothing to process.")
                            st.stop()

                        st.session_state["audit_log"] = report

                        # Step 2: Batch inference with progress bar
                        st.write(f"2. Classifying {len(clean_df)} rows...")
                        texts = clean_df["text"].tolist()
                        batch_size = 64
                        all_preds = []
                        progress = st.progress(0, text="Processing...")

                        for i in range(0, len(texts), batch_size):
                            batch = texts[i : i + batch_size]
                            batch_preds = classifier(batch, truncation=True, max_length=MAX_LENGTH)
                            all_preds.extend(batch_preds)
                            done = min(i + batch_size, len(texts))
                            progress.progress(done / len(texts), text=f"{done}/{len(texts)} rows...")

                        progress.empty()

                        clean_df["predicted_label"] = [
                            max(p, key=lambda x: x["score"])["label"] for p in all_preds
                        ]
                        clean_df["confidence"] = [
                            max(p, key=lambda x: x["score"])["score"] for p in all_preds
                        ]

                        # Step 3: Lock threshold and store results
                        st.session_state["full_scored_df"] = clean_df
                        st.session_state["project_name"] = proj
                        st.session_state["locked_threshold"] = conf_threshold
                        st.session_state["committed"] = False

                        # Step 4: Auto-save both splits to disk (crash recovery)
                        df_auto_tmp, df_review_tmp = split_data_by_threshold(
                            clean_df, conf_threshold
                        )

                        if not df_auto_tmp.empty:
                            df_auto_tmp.to_csv(
                                f"{DIRS['workspace']}/{TODAY}_{proj}_auto.csv", index=False
                            )

                        if not df_review_tmp.empty:
                            df_review_tmp.to_csv(
                                f"{DIRS['review']}/{TODAY}_{proj}_pending.csv", index=False
                            )

                        # Step 5: Snapshot review labels for change detection at commit time
                        if not df_review_tmp.empty:
                            st.session_state["review_snapshot"] = df_review_tmp[
                                ["row_id", "predicted_label"]
                            ].copy()
                        else:
                            st.session_state["review_snapshot"] = pd.DataFrame(
                                columns=["row_id", "predicted_label"]
                            )

                        status.update(
                            label="✅ Analysis Complete!", state="complete", expanded=False
                        )
                    except Exception as e:
                        status.update(label="Error", state="error")
                        st.error(str(e))
                        st.code(traceback.format_exc())
                        st.stop()

    with col_r:
        if st.session_state["full_scored_df"] is not None:
            st.subheader("Analysis Summary")
            log = st.session_state["audit_log"]
            c1, c2, c3 = st.columns(3)
            c1.metric("1. Raw", log["raw"])
            c2.metric("2. Removed", log["removed"], delta_color="inverse")
            c3.metric("3. Final", log["clean"], delta="Ready")

            if log["removed"] > 0:
                st.warning(f"⚠️ {log['msg']}")

            st.divider()
            threshold = st.session_state["locked_threshold"] or conf_threshold
            df_full = st.session_state["full_scored_df"]
            df_auto_preview, df_review_preview = split_data_by_threshold(df_full, threshold)
            m1, m2 = st.columns(2)
            m1.metric("✅ High Confidence", len(df_auto_preview))
            m2.metric("⚠️ Needs Your Review", len(df_review_preview))


# --------------- TAB 2: Review & Optimize ---------------

with tab_review:
    st.header("Review & Edit Labels")

    # -- Load Pending: shown only when no data is loaded --
    if st.session_state["full_scored_df"] is None:
        pending_files = sorted(
            glob.glob(f"{DIRS['review']}/*.csv"), key=os.path.getmtime, reverse=True
        )
        if pending_files:
            with st.expander("📂 Continue previous review", expanded=True):
                options = [os.path.basename(f) for f in pending_files]
                selected = st.selectbox("Select a saved session:", options)

                if st.button("Load Selected File"):
                    selected_path = pending_files[options.index(selected)]
                    df_loaded = pd.read_csv(selected_path)
                    df_loaded = df_loaded.loc[:, ~df_loaded.columns.str.contains("^Unnamed")]

                    # Standardize text column name
                    if "content" in df_loaded.columns and "text" not in df_loaded.columns:
                        df_loaded.rename(columns={"content": "text"}, inplace=True)

                    if "row_id" not in df_loaded.columns:
                        df_loaded["row_id"] = [str(uuid.uuid4()) for _ in range(len(df_loaded))]
                    if "confidence" not in df_loaded.columns:
                        df_loaded["confidence"] = 0.0
                    if "predicted_label" not in df_loaded.columns:
                        df_loaded["predicted_label"] = "neutral"

                    # All loaded rows go to review (threshold above any real confidence)
                    st.session_state["full_scored_df"] = df_loaded
                    st.session_state["locked_threshold"] = 1.0
                    st.session_state["audit_log"] = {
                        "raw": len(df_loaded), "removed": 0,
                        "clean": len(df_loaded), "msg": "Loaded from pending file.",
                    }
                    st.session_state["review_snapshot"] = df_loaded[
                        ["row_id", "predicted_label"]
                    ].copy()
                    st.session_state["committed"] = False

                    # Extract project name from filename (e.g. "20260213_batch_001_pending.csv")
                    fname = os.path.basename(selected_path).replace(".csv", "")
                    parts = fname.split("_")
                    if len(parts) >= 2:
                        st.session_state["project_name"] = "_".join(parts[1:-1]) or parts[1]

                    st.rerun()
        else:
            st.info("👈 Upload your data in the first tab to get started.")

    # -- Main review UI --
    if st.session_state["full_scored_df"] is not None:
        threshold = st.session_state["locked_threshold"] or conf_threshold
        df_full = st.session_state["full_scored_df"]
        df_auto, df_review = split_data_by_threshold(df_full, threshold)

        # --- Case A: All rows auto-passed ---
        if df_review.empty:
            st.success("✨ All rows were classified with high confidence — no review needed!")

            if st.session_state["committed"]:
                st.info("Report already generated for this session.")
            elif st.button("📊 Generate Report", type="primary"):
                try:
                    proj = st.session_state["project_name"]
                    ts = datetime.datetime.now().strftime("%H%M%S")

                    df_report = build_report(df_auto, pd.DataFrame())
                    rep_path = f"{DIRS['reports']}/{TODAY}_{proj}_{ts}_Report.csv"
                    df_report.to_csv(rep_path, index=False)

                    st.session_state["committed"] = True
                    st.success(f"✅ Report saved: {os.path.basename(rep_path)}")
                    st.info(
                        f"All {len(df_report)} rows classified automatically."
                    )

                    with open(rep_path, "rb") as f:
                        st.download_button(
                            "📥 Download Report", f, file_name=os.path.basename(rep_path)
                        )
                except Exception as e:
                    st.error("Save failed")
                    st.code(traceback.format_exc())

        # --- Case B: Review queue has items ---
        else:
            col_filter, col_editor = st.columns([1, 4])

            with col_filter:
                st.markdown("### 🔍 Filter")
                st.caption(f"{len(df_review)} rows to review")
                labels = df_review["predicted_label"].unique().tolist()
                sel_labels = st.multiselect("Label:", labels, default=labels)
                df_display = df_review[df_review["predicted_label"].isin(sel_labels)]

                st.markdown("---")
                st.caption(
                    "Tip: Make sure all labels you want to show in the editor "
                    "are selected above before saving."
                )

            with col_editor:
                if df_display.empty:
                    st.warning("No rows match the current filter. Adjust the label filter.")
                else:
                    st.markdown(f"**Showing {len(df_display)} of {len(df_review)} rows**")

                    # Only show columns relevant to review
                    review_cols = ["row_id", "text", "predicted_label", "confidence"]
                    df_editor = df_display[review_cols].copy()

                    edited_display = st.data_editor(
                        df_editor,
                        column_config={
                            "row_id": st.column_config.TextColumn(
                                "ID", disabled=True, width="small"
                            ),
                            "text": st.column_config.TextColumn(
                                "Text", disabled=True, width="large"
                            ),
                            "predicted_label": st.column_config.SelectboxColumn(
                                "Correct Label", options=LABEL_OPTIONS, required=True
                            ),
                            "confidence": st.column_config.ProgressColumn(
                                "Conf", format="%.2f", min_value=0, max_value=1
                            ),
                        },
                        disabled=["text", "confidence", "row_id"],
                        use_container_width=True,
                        hide_index=True,
                        height=600,
                        key="review_editor",
                    )

                    # Pre-commit diff summary
                    snapshot = st.session_state.get("review_snapshot")
                    n_changed = 0
                    if snapshot is not None and not snapshot.empty:
                        snap_map = dict(
                            zip(snapshot["row_id"], snapshot["predicted_label"])
                        )
                        n_changed = sum(
                            1
                            for _, r in edited_display.iterrows()
                            if r["predicted_label"] != snap_map.get(r["row_id"], "")
                        )
                    st.caption(f"📝 {n_changed} label(s) changed")

                st.divider()

                if st.session_state["committed"]:
                    st.info(
                        "Report already generated. "
                        "Upload new data or load a saved session to continue."
                    )
                elif df_display.empty:
                    pass  # No buttons when filter shows nothing
                else:
                    btn_save, btn_report = st.columns(2)
                    with btn_save:
                        save_clicked = st.button(
                            "💾 Save Progress", use_container_width=True,
                            help="Save your corrections and come back later to continue.",
                        )
                    with btn_report:
                        report_clicked = st.button(
                            "📊 Generate Report", type="primary",
                            use_container_width=True,
                            help="Save corrections and generate the final report for download.",
                        )

                    if save_clicked:
                        try:
                            proj = st.session_state["project_name"]
                            final_review, human_rows, pending_rows, gold_count = (
                                commit_core(df_review, edited_display, proj)
                            )

                            if gold_count > 0:
                                st.success(
                                    f"✅ {gold_count} corrections saved."
                                )
                            else:
                                st.info("No changes detected — nothing to save.")

                            if not pending_rows.empty:
                                st.caption(
                                    f"📋 {len(pending_rows)} rows remaining for next session."
                                )

                                # Refresh session: keep only pending rows for continued editing
                                pending_clean = pending_rows.drop(
                                    columns=["label_source"], errors="ignore"
                                )
                                st.session_state["full_scored_df"] = pending_clean
                                st.session_state["locked_threshold"] = 1.0
                                st.session_state["audit_log"] = {
                                    "raw": len(pending_clean), "removed": 0,
                                    "clean": len(pending_clean),
                                    "msg": "Continued from save.",
                                }
                                st.session_state["review_snapshot"] = pending_clean[
                                    ["row_id", "predicted_label"]
                                ].copy()
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.success("All rows reviewed!")
                                st.session_state["committed"] = True

                        except Exception as e:
                            st.error("Save failed")
                            st.code(traceback.format_exc())

                    if report_clicked:
                        try:
                            proj = st.session_state["project_name"]
                            ts = datetime.datetime.now().strftime("%H%M%S")
                            final_review, human_rows, pending_rows, gold_count = (
                                commit_core(df_review, edited_display, proj)
                            )

                            # Build full report (auto + review)
                            df_report = build_report(df_auto, final_review)

                            # Integrity check
                            expected = st.session_state["audit_log"]["clean"]
                            if len(df_report) != expected:
                                st.error(
                                    f"Row count mismatch: expected {expected}, "
                                    f"got {len(df_report)}. Please retry."
                                )
                                st.stop()

                            # Save report
                            rep_path = f"{DIRS['reports']}/{TODAY}_{proj}_{ts}_Report.csv"
                            df_report.to_csv(rep_path, index=False)

                            st.session_state["committed"] = True

                            st.balloons()
                            st.success("Report generated!")

                            src = df_report["label_source"].value_counts().to_dict()
                            c1, c2, c3 = st.columns(3)
                            c1.metric("AI Classified", src.get("auto", 0))
                            c2.metric("You Corrected", src.get("human", 0))
                            c3.metric("Not Yet Reviewed", src.get("model_pending", 0))

                            if gold_count > 0:
                                st.info(
                                    f"✅ {gold_count} corrections saved."
                                )
                            if not pending_rows.empty:
                                st.caption(
                                    f"📋 {len(pending_rows)} unreviewed rows "
                                    "saved — you can continue later."
                                )

                            with open(rep_path, "rb") as f:
                                st.download_button(
                                    "📥 Download Report",
                                    f,
                                    file_name=os.path.basename(rep_path),
                                )
                        except Exception as e:
                            st.error("Save failed")
                            st.code(traceback.format_exc())


# --------------- TAB 3: Knowledge Base ---------------

with tab_knowledge:
    st.header("Review History")

    # Model status
    new_gold_count = get_new_gold_count_since_model(model_meta)
    if new_gold_count is not None:
        if new_gold_count > 0:
            st.info(
                f"📊 **{new_gold_count} new corrections** since the last model update."
            )
        else:
            st.success("Model is up to date.")

    # Correction history
    gold_files = sorted(glob.glob(f"{DIRS['gold']}/*.csv"), key=os.path.getmtime, reverse=True)
    if gold_files:
        data = []
        for f in gold_files:
            try:
                n = len(pd.read_csv(f))
            except Exception:
                n = "?"
            data.append({
                "File": os.path.basename(f),
                "Corrections": n,
                "Date": time.ctime(os.path.getmtime(f)),
            })
        st.dataframe(pd.DataFrame(data), use_container_width=True)
    else:
        st.info("No review history yet. Your corrections will appear here.")

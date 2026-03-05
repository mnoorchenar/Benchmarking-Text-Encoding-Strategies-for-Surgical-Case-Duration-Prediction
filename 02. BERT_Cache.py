# =============================================================================
# 02. BERT_Cache.py
#
# Computes and caches BERT embeddings for each active text column individually.
# Each column × model combination produces one .npy file.
# On rerun, existing cache files are automatically overwritten.
#
# POSSIBLE TASKS (8 total — which ones run depends on active text columns):
#   ClinicalBERT  on scheduled_procedure      (always runs)
#   ClinicalBERT  on procedure                (runs if procedure was selected)
#   ClinicalBERT  on operative_dx             (runs if operative_dx was selected)
#   ClinicalBERT  on most_responsible_dx      (runs if most_responsible_dx was selected)
#   Sentence-BERT on scheduled_procedure      (always runs)
#   Sentence-BERT on procedure                (runs if procedure was selected)
#   Sentence-BERT on operative_dx             (runs if operative_dx was selected)
#   Sentence-BERT on most_responsible_dx      (runs if most_responsible_dx was selected)
#
# Active columns are auto-detected from the 'text_config' metadata table
# written by 01. Pre-processing.py.  Each column is kept as-is — no columns
# are merged together.
# =============================================================================

import os
import sys
import sqlite3
import time
import numpy as np
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================
INPUT_DB    = './data/surgical_data.db'
INPUT_TABLE = 'Clean'
CACHE_DIR   = './data/bert_cache'
LOG_DIR     = './results'

# All possible text columns in fixed order.
ALL_TEXT_COLS = ['scheduled_procedure', 'procedure', 'operative_dx', 'most_responsible_dx']

# All possible tasks: id → (model, column_name, output_filename)
# Built as a flat list covering every column × model combination.
# The active subset is determined after auto-detection.
ALL_POSSIBLE_TASKS = {
    1: ('clinicalbert',  'scheduled_procedure',    'clinicalbert_scheduled_procedure.npy'),
    2: ('clinicalbert',  'procedure',              'clinicalbert_procedure.npy'),
    3: ('clinicalbert',  'operative_dx',           'clinicalbert_operative_dx.npy'),
    4: ('clinicalbert',  'most_responsible_dx',    'clinicalbert_most_responsible_dx.npy'),
    5: ('sentencebert',  'scheduled_procedure',    'sentencebert_scheduled_procedure.npy'),
    6: ('sentencebert',  'procedure',              'sentencebert_procedure.npy'),
    7: ('sentencebert',  'operative_dx',           'sentencebert_operative_dx.npy'),
    8: ('sentencebert',  'most_responsible_dx',    'sentencebert_most_responsible_dx.npy'),
}

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

# =============================================================================
# LOGGER
# =============================================================================

class _Tee:
    def __init__(self, log_path):
        self._terminal = sys.stdout
        self._log = open(log_path, "w", encoding="utf-8", buffering=1)
    def write(self, msg):
        self._terminal.write(msg)
        self._log.write(msg)
    def flush(self):
        self._terminal.flush()
        self._log.flush()
    def close(self):
        sys.stdout = self._terminal
        self._log.close()

def sep(title='', width=70, char='='):
    if title:
        print(f"\n{char*width}\n  {title}\n{char*width}")
    else:
        print(char * width)

# =============================================================================
# AUTO-DETECT ACTIVE TEXT COLUMNS
# Reads the 'text_config' metadata table written by 01. Pre-processing.py.
# Returns a list of column names that are active (e.g. ['scheduled_procedure']).
# Falls back to content-based detection if the metadata table is missing.
# =============================================================================

def detect_active_cols():
    active = ['scheduled_procedure']   # safe default

    try:
        with sqlite3.connect(INPUT_DB) as conn:
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

            if 'text_config' not in tables:
                print("  ⚠  'text_config' table not found — falling back to content-based detection...")
                return _detect_from_content(conn, active)

            row = conn.execute("SELECT active_text_cols FROM text_config").fetchone()
            active = [c.strip() for c in row[0].split(',') if c.strip()]
            print(f"  ✅ Detected from 'text_config' metadata table.")

    except Exception as e:
        print(f"  ⚠  Could not read text_config: {e} — falling back to content-based detection...")
        try:
            with sqlite3.connect(INPUT_DB) as conn:
                active = _detect_from_content(conn, active)
        except Exception as e2:
            print(f"  ⚠  Content detection also failed: {e2} — defaulting to scheduled_procedure only.")

    return active


def _detect_from_content(conn, default):
    """Check which known text columns exist in the DB and contain non-empty text."""
    try:
        existing = [r[1] for r in conn.execute(f"PRAGMA table_info({INPUT_TABLE})").fetchall()]
        active   = []
        for col in ALL_TEXT_COLS:
            if col not in existing:
                print(f"    {col}: not found in table — skipping")
                continue
            n = conn.execute(f"SELECT COUNT(*) FROM {INPUT_TABLE} WHERE [{col}] IS NOT NULL AND TRIM([{col}]) != ''").fetchone()[0]
            if n > 0:
                active.append(col)
                print(f"    {col}: {n:,} non-empty rows — included ✅")
            else:
                print(f"    {col}: column present but empty — skipping ❌")
        return active if active else default
    except Exception as e:
        print(f"  ⚠  Content detection error: {e}")
        return default

# =============================================================================
# DATA LOADING
# =============================================================================

def load_texts(col_name):
    with sqlite3.connect(INPUT_DB) as conn:
        df = pd.read_sql(f"SELECT [{col_name}] FROM {INPUT_TABLE}", conn)
    texts = df[col_name].astype(str).tolist()
    print(f"  Loaded {len(texts):,} texts from column '{col_name}'")
    return texts

# =============================================================================
# EMBEDDING FUNCTIONS
# =============================================================================

def compute_clinicalbert(texts, batch_size=32):
    import torch
    from transformers import AutoTokenizer, AutoModel

    print("  Loading ClinicalBERT model (emilyalsentzer/Bio_ClinicalBERT)...")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model     = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"  Device: {device}")

    embs      = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        with torch.no_grad():
            inp = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=64)
            inp = {k: v.to(device) for k, v in inp.items()}
            out = model(**inp)
            embs.append(out.last_hidden_state[:, 0, :].cpu().numpy())
        if (i // batch_size) % 50 == 0:
            print(f"    Batch {i//batch_size + 1}/{n_batches}  ({i:,}/{len(texts):,} texts)")

    return np.vstack(embs)


def compute_sentencebert(texts, batch_size=64):
    from sentence_transformers import SentenceTransformer

    print("  Loading Sentence-BERT model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("  Encoding...")
    return model.encode(texts, show_progress_bar=True, batch_size=batch_size)

# =============================================================================
# SINGLE TASK RUNNER
# =============================================================================

def run_task(task_id, tasks):
    method, col_name, out_filename = tasks[task_id]
    out_path = os.path.join(CACHE_DIR, out_filename)
    log_path = os.path.join(LOG_DIR, f"02_bert_task{task_id}_{method}_{col_name}.log")

    tee = _Tee(log_path)
    sys.stdout = tee

    sep(f"TASK {task_id} — {method.upper()} on '{col_name}'")
    print(f"  Output : {out_path}")
    print(f"  Log    : {log_path}")

    if os.path.exists(out_path):
        os.remove(out_path)
        print(f"  Removed existing cache file — will recompute fresh.")

    t0    = time.time()
    texts = load_texts(col_name)

    if method == 'clinicalbert':
        embeddings = compute_clinicalbert(texts)
    else:
        embeddings = compute_sentencebert(texts)

    elapsed = time.time() - t0
    print(f"\n  Embedding shape : {embeddings.shape}")
    print(f"  Elapsed         : {elapsed/60:.2f} minutes")
    print(f"  Saving to       : {out_path}")

    np.save(out_path, embeddings)
    print(f"  ✅ Saved successfully.")

    tee.close()

# =============================================================================
# INTERACTIVE MENU
# =============================================================================

def show_menu():
    sep("02. BERT_Cache.py")

    # ── Auto-detect active text columns ──────────────────────────────────────
    sep("AUTO-DETECTION  —  reading text column config from database", char='-')
    active_cols = detect_active_cols()

    print(f"\n  Active text columns : {active_cols}")
    for col in ALL_TEXT_COLS:
        status = '✅  active' if col in active_cols else '❌  not selected in preprocessing'
        print(f"    {col:<30} {status}")

    # ── Build TASKS for active columns only ───────────────────────────────────
    TASKS = {k: v for k, v in ALL_POSSIBLE_TASKS.items() if v[1] in active_cols}

    if not TASKS:
        print("\n  ❌ No active text columns detected — nothing to encode. Exiting.")
        return

    # ── Task status display ───────────────────────────────────────────────────
    sep("TASK MENU", char='-')
    print()
    print(f"  {'[ID]':<6} {'Model':<16} {'Column':<30} {'Cache file status'}")
    print(f"  {'-'*80}")
    for tid, (method, col_name, out_filename) in TASKS.items():
        out_path     = os.path.join(CACHE_DIR, out_filename)
        cache_status = "✅ exists — will be overwritten" if os.path.exists(out_path) else "❌ missing"
        print(f"  [{tid}]   {method:<16} {col_name:<30} {cache_status}")

    print()
    print("  [0]  Run ALL applicable tasks  [default]")
    print()
    raw = input("  Select tasks to run (e.g. 1,5 or press Enter for all): ").strip()

    # ── Task selection ────────────────────────────────────────────────────────
    if raw == '' or raw == '0':
        selected = list(TASKS.keys())
    else:
        selected = []
        for part in raw.split(','):
            part = part.strip()
            if not part.isdigit():
                print(f"  ⚠  Ignored unrecognized input: '{part}'")
                continue
            tid = int(part)
            if tid not in ALL_POSSIBLE_TASKS:
                print(f"  ⚠  Task {tid} does not exist — ignored.")
                continue
            if tid not in TASKS:
                _, col_name, _ = ALL_POSSIBLE_TASKS[tid]
                print(f"  ⚠  Task {tid} targets '{col_name}' which was not selected in preprocessing — running anyway as explicitly requested.")
                TASKS[tid] = ALL_POSSIBLE_TASKS[tid]
            selected.append(tid)

    if not selected:
        print("  No valid tasks selected — exiting.")
        return

    print(f"\n  Selected tasks: {selected}")
    print(f"  Running sequentially...\n")
    for tid in selected:
        run_task(tid, TASKS)

    sep("ALL SELECTED TASKS COMPLETE")
    for tid in selected:
        _, _, out_filename = TASKS[tid]
        out_path = os.path.join(CACHE_DIR, out_filename)
        if os.path.exists(out_path):
            arr = np.load(out_path)
            print(f"  Task {tid} — {out_filename:<45}  shape={arr.shape}")
        else:
            print(f"  Task {tid} — {out_filename:<45}  ⚠ file not found")

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    show_menu()
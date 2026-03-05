# =============================================================================
# LOGGER SETUP
# =============================================================================
import sys, os
os.makedirs("./results", exist_ok=True)

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

_tee = _Tee("./results/03_fold_encoding.log")
sys.stdout = _tee

# =============================================================================
# IMPORTS
# =============================================================================
import sqlite3, warnings, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================
INPUT_DB     = './data/surgical_data.db'
INPUT_TABLE  = 'Clean'
FOLD_TABLE   = 'fold_indices'
OUTPUT_DB    = './data/fold_encoded.db'

TARGET       = 'actual_casetime_minutes'
EXCLUDE_COLS = ['procedure_minutes', 'procedure_time', 'induction_time', 'emergence_time', 'scheduled_duration']
IMPUTE_COLS  = ['age_at_discharge', 'avg_BMI', 'anesthetic_type']
IMPUTE_TYPES = ['regression',       'regression', 'classification']

ALL_TEXT_COLS          = ['scheduled_procedure', 'procedure', 'operative_dx', 'most_responsible_dx']
TEXT_ENCODINGS         = ['label', 'tfidf', 'count']
DEFAULT_FEATURE_COUNTS = [10, 50, 100, 200]

N_SPLITS     = 5
RANDOM_STATE = 42

# =============================================================================
# LOG HELPERS
# =============================================================================

def sep(title='', width=70, char='='):
    if title:
        print(f"\n{char*width}\n  {title}\n{char*width}")
    else:
        print(char * width)

def print_matrix_summary(label, arr):
    print(f"    {label:<30} shape={arr.shape}  dtype={arr.dtype}  min={arr.min():.4f}  max={arr.max():.4f}  NaN={np.isnan(arr).sum()}")

# =============================================================================
# AUTO-DETECT ACTIVE TEXT COLUMNS
# =============================================================================

def detect_active_text_cols():
    active = ['scheduled_procedure']
    try:
        with sqlite3.connect(INPUT_DB) as conn:
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            if 'text_config' not in tables:
                print("  ⚠  'text_config' not found — falling back to content-based detection...")
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
    try:
        existing = [r[1] for r in conn.execute(f"PRAGMA table_info({INPUT_TABLE})").fetchall()]
        active = []
        for col in ALL_TEXT_COLS:
            if col not in existing:
                continue
            n = conn.execute(f"SELECT COUNT(*) FROM {INPUT_TABLE} WHERE [{col}] IS NOT NULL AND TRIM([{col}]) != ''").fetchone()[0]
            if n > 0:
                active.append(col)
                print(f"    {col}: {n:,} non-empty rows — included ✅")
            else:
                print(f"    {col}: empty — skipping ❌")
        return active if active else default
    except Exception as e:
        print(f"  ⚠  Content detection error: {e}")
        return default

# =============================================================================
# DATABASE HELPERS
# =============================================================================

def load_db(db, table):
    with sqlite3.connect(db) as conn:
        return pd.read_sql(f"SELECT * FROM {table}", conn)

def init_output_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS encoded_matrices (
            fold     INTEGER,
            split    TEXT,
            encoding TEXT,
            rows     INTEGER,
            cols     INTEGER,
            dtype    TEXT,
            data     BLOB
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS encoded_targets (
            fold  INTEGER,
            split TEXT,
            rows  INTEGER,
            dtype TEXT,
            data  BLOB
        )
    """)
    conn.commit()

def get_existing_encoding_keys():
    """Return a set of (fold, split, encoding) tuples already in the DB."""
    if not os.path.exists(OUTPUT_DB):
        return set()
    try:
        with sqlite3.connect(OUTPUT_DB) as conn:
            rows = conn.execute("SELECT fold, split, encoding FROM encoded_matrices").fetchall()
        return set(rows)
    except Exception:
        return set()

def get_existing_target_keys():
    """Return a set of (fold, split) tuples already in encoded_targets."""
    if not os.path.exists(OUTPUT_DB):
        return set()
    try:
        with sqlite3.connect(OUTPUT_DB) as conn:
            rows = conn.execute("SELECT fold, split FROM encoded_targets").fetchall()
        return set(rows)
    except Exception:
        return set()

def save_matrix(conn, fold, split, encoding_key, arr):
    conn.execute("""
        INSERT INTO encoded_matrices (fold, split, encoding, rows, cols, dtype, data)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (int(fold), split, encoding_key, arr.shape[0], arr.shape[1], str(arr.dtype), arr.tobytes()))

def save_target(conn, fold, split, arr):
    conn.execute("""
        INSERT INTO encoded_targets (fold, split, rows, dtype, data)
        VALUES (?, ?, ?, ?, ?)
    """, (int(fold), split, len(arr), str(arr.dtype), arr.tobytes()))

# =============================================================================
# FOLD-WISE IMPUTATION
# =============================================================================

def impute_fold(train_df, val_df, text_cols):
    train_df, val_df = train_df.copy(), val_df.copy()
    print(f"\n  Imputation (fit on training rows only):")

    for col, ptype in zip(IMPUTE_COLS, IMPUTE_TYPES):
        n_tr_nan = train_df[col].isna().sum()
        n_va_nan = val_df[col].isna().sum()
        print(f"    [{col}]  type={ptype}  train_NaN={n_tr_nan:,}  val_NaN={n_va_nan:,}")

        if n_tr_nan == 0 and n_va_nan == 0:
            print(f"      → No NaN found, skipping.")
            continue

        num_feats  = [c for c in train_df.columns if c not in [col, TARGET] + text_cols and pd.api.types.is_numeric_dtype(train_df[c])]
        pre        = Pipeline([('imp', SimpleImputer(strategy='median'))])
        tr_known   = train_df[train_df[col].notna()]
        tr_missing = train_df[train_df[col].isna()]
        va_missing = val_df[val_df[col].isna()]

        if len(tr_known) == 0:
            print(f"      → ⚠ No known values in training fold — skipping.")
            continue

        X_tr_k = pre.fit_transform(tr_known[num_feats])

        if ptype == 'classification':
            le    = LabelEncoder()
            y_trk = le.fit_transform(tr_known[col].astype(str))
            mdl   = XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss', verbosity=0)
            mdl.fit(X_tr_k, y_trk)
            def predict_col(X, _le=le, _mdl=mdl):
                return _le.inverse_transform(np.round(_mdl.predict(X)).astype(int))
        else:
            mdl = XGBRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
            mdl.fit(X_tr_k, tr_known[col].values)
            predict_col = mdl.predict

        if len(tr_missing) > 0:
            train_df.loc[train_df[col].isna(), col] = predict_col(pre.transform(tr_missing[num_feats]))
            print(f"      → Filled {len(tr_missing):,} train NaN")
        if len(va_missing) > 0:
            val_df.loc[val_df[col].isna(), col] = predict_col(pre.transform(va_missing[num_feats]))
            print(f"      → Filled {len(va_missing):,} val NaN")

    return train_df, val_df


def onehot_anesthetic(train_df, val_df, fold):
    train_df, val_df = train_df.copy(), val_df.copy()
    cats = sorted(train_df['anesthetic_type'].dropna().unique())
    print(f"  Anesthetic type categories from training fold {fold}: {sorted(cats)}")
    for cat in cats:
        col_name = f'anesthetic_type__{cat}'
        train_df[col_name] = (train_df['anesthetic_type'] == cat).astype(int)
        val_df[col_name]   = (val_df['anesthetic_type']   == cat).astype(int)
    train_df.drop(columns=['anesthetic_type'], inplace=True)
    val_df.drop(columns=['anesthetic_type'], inplace=True)
    return train_df, val_df

# =============================================================================
# TEXT ENCODING  (fit on train only, parameterised by N)
# =============================================================================

def encode_label(tr_texts, va_texts, text_cols, n):
    enc_tr, enc_va = [], []
    for col, tr_t, va_t in zip(text_cols, tr_texts, va_texts):
        top = pd.Series(tr_t).value_counts().nlargest(n - 1).index
        def _enc(texts, _top=top, _col=col):
            mapped = pd.Series(texts).where(pd.Series(texts).isin(_top), 'Other')
            return pd.get_dummies(mapped, prefix=f'{_col}_lbl').reindex(
                columns=[f'{_col}_lbl_{c}' for c in list(_top) + ['Other']], fill_value=0
            ).values
        enc_tr.append(_enc(tr_t))
        enc_va.append(_enc(va_t))
    return np.hstack(enc_tr).astype(np.float32), np.hstack(enc_va).astype(np.float32)

def encode_tfidf(tr_texts, va_texts, n):
    enc_tr, enc_va = [], []
    for tr_t, va_t in zip(tr_texts, va_texts):
        vec = TfidfVectorizer(max_features=n, stop_words='english', ngram_range=(1, 2))
        enc_tr.append(vec.fit_transform(tr_t).toarray())
        enc_va.append(vec.transform(va_t).toarray())
    return np.hstack(enc_tr).astype(np.float32), np.hstack(enc_va).astype(np.float32)

def encode_count(tr_texts, va_texts, n):
    enc_tr, enc_va = [], []
    for tr_t, va_t in zip(tr_texts, va_texts):
        vec = CountVectorizer(max_features=n, stop_words='english', ngram_range=(1, 2))
        enc_tr.append(vec.fit_transform(tr_t).toarray())
        enc_va.append(vec.transform(va_t).toarray())
    return np.hstack(enc_tr).astype(np.float32), np.hstack(enc_va).astype(np.float32)

def apply_text_encoding(train_df, val_df, encoding, text_cols, n):
    struct_cols = [c for c in train_df.columns if c not in text_cols + [TARGET]]
    S_tr = train_df[struct_cols].values.astype(np.float32)
    S_va = val_df[struct_cols].values.astype(np.float32)
    tr_texts = [train_df[c].astype(str).tolist() for c in text_cols]
    va_texts = [val_df[c].astype(str).tolist()   for c in text_cols]
    if encoding == 'label':
        T_tr, T_va = encode_label(tr_texts, va_texts, text_cols, n)
    elif encoding == 'tfidf':
        T_tr, T_va = encode_tfidf(tr_texts, va_texts, n)
    elif encoding == 'count':
        T_tr, T_va = encode_count(tr_texts, va_texts, n)
    else:
        raise ValueError(f"Unknown text encoding: {encoding}")
    return np.hstack([S_tr, T_tr]), np.hstack([S_va, T_va])

def apply_structured_only(train_df, val_df, text_cols):
    struct_cols = [c for c in train_df.columns if c not in text_cols + [TARGET]]
    return (train_df[struct_cols].values.astype(np.float32),
            val_df[struct_cols].values.astype(np.float32))

# =============================================================================
# FEATURE COUNT METADATA  (cumulative — merges old + new)
# =============================================================================

def load_existing_feature_counts():
    """Read feature counts previously stored in feature_config, if any."""
    try:
        with sqlite3.connect(INPUT_DB) as conn:
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            if 'feature_config' not in tables:
                return []
            row = conn.execute("SELECT feature_counts FROM feature_config").fetchone()
            return sorted(set(int(x.strip()) for x in row[0].split(',') if x.strip().isdigit()))
    except Exception:
        return []

def save_feature_counts(counts):
    """Persist the cumulative set of feature counts to feature_config."""
    meta = pd.DataFrame([{'feature_counts': ','.join(str(n) for n in sorted(counts))}])
    with sqlite3.connect(INPUT_DB) as conn:
        meta.to_sql('feature_config', conn, if_exists='replace', index=False)

# =============================================================================
# MAIN
# =============================================================================
sep("03. Fold_Encoding.py")

# --- Auto-detect active text columns -----------------------------------------
sep("AUTO-DETECTION  —  reading text column config from database")
TEXT_COLS = detect_active_text_cols()
print(f"\n  Active text columns : {TEXT_COLS}")
for col in ALL_TEXT_COLS:
    print(f"    {col:<30} {'✅  active' if col in TEXT_COLS else '❌  not selected'}")

# --- Existing state -----------------------------------------------------------
sep("EXISTING STATE")
existing_keys   = get_existing_encoding_keys()   # set of (fold, split, encoding_key)
existing_targets = get_existing_target_keys()    # set of (fold, split)
existing_counts = load_existing_feature_counts()

if existing_keys:
    existing_enc_names = sorted({k[2] for k in existing_keys})
    print(f"  Found existing encoded_matrices DB with {len(existing_keys):,} rows.")
    print(f"  Existing encoding keys : {existing_enc_names}")
    print(f"  Existing feature counts: {existing_counts}")
    print(f"  ℹ  Already-computed encoding keys will be SKIPPED.")
else:
    print(f"  No existing encoded_matrices found — fresh run.")

# --- Interactive feature count selection -------------------------------------
sep("FEATURE COUNT SELECTION")
print(f"""
  Classical text encodings (label, tfidf, count) are encoded once per N.
  Encoding key format: 'tfidf_n100', 'label_n50', etc.
  'only_structured' has no text features and is encoded once (no N suffix).

  Previously computed counts : {existing_counts if existing_counts else 'none'}
  Default new counts         : {DEFAULT_FEATURE_COUNTS}

  Only counts NOT already in the DB will be computed.
  Enter the full list you want (existing ones will be auto-skipped).
""")
raw_counts = input("  Enter feature counts (e.g. 10,50,100,200) or press Enter for default: ").strip()
if raw_counts:
    try:
        requested = sorted(set(int(x.strip()) for x in raw_counts.split(',') if x.strip().isdigit() and int(x.strip()) > 0))
        FEATURE_COUNTS = requested if requested else DEFAULT_FEATURE_COUNTS
    except Exception:
        FEATURE_COUNTS = DEFAULT_FEATURE_COUNTS
else:
    FEATURE_COUNTS = DEFAULT_FEATURE_COUNTS

# Split into new vs already-done
new_counts      = [n for n in FEATURE_COUNTS if not all((fold, split, f"{enc}_n{n}") in existing_keys for fold in range(N_SPLITS) for split in ['train', 'val'] for enc in TEXT_ENCODINGS)]
skip_counts     = [n for n in FEATURE_COUNTS if n not in new_counts]
all_counts      = sorted(set(existing_counts) | set(FEATURE_COUNTS))

print(f"\n  Requested : {FEATURE_COUNTS}")
print(f"  To skip   : {skip_counts}  (already fully encoded)")
print(f"  To compute: {new_counts}")
print(f"  Cumulative: {all_counts}")

# Update feature_config cumulatively
save_feature_counts(all_counts)
print(f"\n  Saved cumulative feature_config → {INPUT_DB}  counts={all_counts}")

# --- Load Clean table --------------------------------------------------------
sep("LOAD CLEAN TABLE")
df = load_db(INPUT_DB, INPUT_TABLE)
df = df[df[TARGET].notna()].copy().reset_index(drop=True)
df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns], inplace=True)
print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Residual NaN: {df[IMPUTE_COLS].isna().sum().to_dict()}")

# --- Load fold indices -------------------------------------------------------
sep("LOAD FOLD INDICES")
fold_df = load_db(INPUT_DB, FOLD_TABLE)
fold_indices = {}
for fold in range(N_SPLITS):
    train_idx = fold_df[(fold_df['fold'] == fold) & (fold_df['split'] == 'train')]['row_index'].values
    val_idx   = fold_df[(fold_df['fold'] == fold) & (fold_df['split'] == 'val')]['row_index'].values
    fold_indices[fold] = (train_idx, val_idx)
    print(f"  Fold {fold}: train={len(train_idx):,}  val={len(val_idx):,}")

# --- Initialize output DB (creates tables if new, leaves existing rows) ------
with sqlite3.connect(OUTPUT_DB) as conn:
    init_output_db(conn)

# --- Main loop --------------------------------------------------------------
sep("ENCODING LOOP  —  fold → impute → encode → save  (skipping existing)")

# Determine which folds actually need any work
folds_needing_work = set()
for fold in range(N_SPLITS):
    for split in ['train', 'val']:
        if (fold, split, 'only_structured') not in existing_keys:
            folds_needing_work.add(fold)
        for enc in TEXT_ENCODINGS:
            for N in new_counts:
                if (fold, split, f"{enc}_n{N}") not in existing_keys:
                    folds_needing_work.add(fold)

if not folds_needing_work:
    print(f"\n  ✅ Nothing to compute — all requested encoding keys already exist.")
else:
    for fold, (train_idx, val_idx) in fold_indices.items():
        if fold not in folds_needing_work:
            print(f"\n  Fold {fold}: all encoding keys already exist — skipping.")
            continue

        sep(f"FOLD {fold}  —  train={len(train_idx):,}  val={len(val_idx):,}", char='-')

        train_base = df.iloc[train_idx].copy().reset_index(drop=True)
        val_base   = df.iloc[val_idx].copy().reset_index(drop=True)
        train_base, val_base = impute_fold(train_base, val_base, TEXT_COLS)
        train_base, val_base = onehot_anesthetic(train_base, val_base, fold)

        y_train = train_base[TARGET].values.astype(np.float64)
        y_val   = val_base[TARGET].values.astype(np.float64)

        # Save targets only if not already saved
        with sqlite3.connect(OUTPUT_DB) as conn:
            if (fold, 'train') not in existing_targets:
                save_target(conn, fold, 'train', y_train)
                print(f"  Saved y_train ({len(y_train):,}) for fold {fold}")
            else:
                print(f"  y_train fold {fold}: already saved — skipping.")
            if (fold, 'val') not in existing_targets:
                save_target(conn, fold, 'val', y_val)
                print(f"  Saved y_val ({len(y_val):,}) for fold {fold}")
            else:
                print(f"  y_val fold {fold}: already saved — skipping.")
            conn.commit()

        # ── only_structured ──────────────────────────────────────────────────
        key = 'only_structured'
        if (fold, 'train', key) in existing_keys and (fold, 'val', key) in existing_keys:
            print(f"\n    {key}: already exists — skipping.")
        else:
            X_tr, X_va = apply_structured_only(train_base, val_base, TEXT_COLS)
            print_matrix_summary(f"train / {key}", X_tr)
            print_matrix_summary(f"val   / {key}", X_va)
            with sqlite3.connect(OUTPUT_DB) as conn:
                save_matrix(conn, fold, 'train', key, X_tr)
                save_matrix(conn, fold, 'val',   key, X_va)
                conn.commit()
            print(f"    Saved fold={fold} encoding={key}")

        # ── Text encodings (only new N values) ────────────────────────────────
        for encoding in TEXT_ENCODINGS:
            for N in new_counts:
                enc_key = f"{encoding}_n{N}"
                if (fold, 'train', enc_key) in existing_keys and (fold, 'val', enc_key) in existing_keys:
                    print(f"\n    {enc_key}: already exists — skipping.")
                    continue
                t0 = time.perf_counter()
                X_tr, X_va = apply_text_encoding(train_base, val_base, encoding, TEXT_COLS, N)
                elapsed = time.perf_counter() - t0
                print_matrix_summary(f"train / {enc_key}", X_tr)
                print_matrix_summary(f"val   / {enc_key}", X_va)
                print(f"      encoding time: {elapsed:.2f}s")
                with sqlite3.connect(OUTPUT_DB) as conn:
                    save_matrix(conn, fold, 'train', enc_key, X_tr)
                    save_matrix(conn, fold, 'val',   enc_key, X_va)
                    conn.commit()
                print(f"    Saved fold={fold} encoding={enc_key}")

# --- Summary ----------------------------------------------------------------
sep("SUMMARY")
with sqlite3.connect(OUTPUT_DB) as conn:
    rows = conn.execute("SELECT COUNT(*) FROM encoded_matrices").fetchone()[0]
    tgts = conn.execute("SELECT COUNT(*) FROM encoded_targets").fetchone()[0]
    print(f"  encoded_matrices table : {rows} rows")
    print(f"  encoded_targets  table : {tgts} rows")
    print(f"  Active text columns    : {TEXT_COLS}")
    print(f"  All feature counts     : {all_counts}")
    print(f"  Newly computed counts  : {new_counts}")
    print(f"  Skipped counts         : {skip_counts}")
    print()
    print(f"  {'Fold':<6} {'Split':<8} {'Encoding':<28} {'Rows':>8} {'Cols':>6}")
    print(f"  {'-'*60}")
    for row in conn.execute("SELECT fold, split, encoding, rows, cols FROM encoded_matrices ORDER BY fold, split, encoding"):
        print(f"  {row[0]:<6} {row[1]:<8} {row[2]:<28} {row[3]:>8,} {row[4]:>6}")

db_size_mb = os.path.getsize(OUTPUT_DB) / (1024 * 1024)
print(f"\n  Database size : {db_size_mb:.1f} MB")
print(f"  Saved to      : {OUTPUT_DB}")
print(f"  Log saved to  : ./results/03_fold_encoding.log")

_tee.close()
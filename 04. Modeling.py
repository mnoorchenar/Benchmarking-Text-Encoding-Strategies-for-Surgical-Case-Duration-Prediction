# =============================================================================
# LOGGER SETUP
# =============================================================================
import sys, os
os.makedirs("./results", exist_ok=True)

class _Tee:
    """Mirrors stdout to the main log file AND to an optional per-model log."""
    def __init__(self, log_path):
        self._terminal  = sys.stdout
        self._log       = open(log_path, "w", encoding="utf-8", buffering=1)
        self._model_log = None          # set via set_model_log()

    def set_model_log(self, fobj):
        """Pass an open file object (or None) to start/stop per-model capture."""
        self._model_log = fobj

    def write(self, msg):
        self._terminal.write(msg)
        self._log.write(msg)
        if self._model_log:
            self._model_log.write(msg)

    def flush(self):
        self._terminal.flush()
        self._log.flush()
        if self._model_log:
            self._model_log.flush()

    def close(self):
        sys.stdout = self._terminal
        self._log.close()
        # per-model log files are closed separately in the main body

_tee = _Tee("./results/04_modeling.log")
sys.stdout = _tee

# =============================================================================
# IMPORTS
# =============================================================================
import sqlite3, warnings, time
import numpy as np
import pandas as pd
import optuna
from scipy.stats import t
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.mixed_precision import set_global_policy

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# GPU SETUP  (MLP only)
# =============================================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        set_global_policy('mixed_float16')
        print(f"  ✅ GPU detected: {[g.name for g in gpus]}")
        print(f"  ✅ Mixed precision enabled (float16 compute / float32 weights)")
    except RuntimeError as e:
        print(f"  ⚠ GPU config error: {e}")
else:
    print("  ⚠ No GPU detected — MLP will run on CPU")

# =============================================================================
# CONFIG
# =============================================================================
ENCODED_DB  = './data/fold_encoded.db'
BERT_DIR    = './data/bert_cache'
RESULT_DB   = './data/result.db'
INPUT_DB    = './data/surgical_data.db'
INPUT_TABLE = 'Clean'

CLASSICAL_ENCODINGS    = ['label', 'tfidf', 'count']
BERT_METHODS           = ['clinicalbert', 'sentencebert']
ALL_TEXT_COLS          = ['scheduled_procedure', 'procedure', 'operative_dx', 'most_responsible_dx']
ALL_MODELS             = ['linear', 'ridge', 'lasso', 'randomforest', 'xgboost', 'mlp']
DEFAULT_FEATURE_COUNTS = [10, 50, 100, 200]

N_SPLITS     = 5
N_TRIALS     = 5
RANDOM_STATE = 42

# Populated after the skip/replace prompt (Stage 2)
REPLACE_EXISTING = False

# =============================================================================
# LOG HELPERS
# =============================================================================

def sep(title='', width=70, char='='):
    if title:
        print(f"\n{char*width}\n  {title}\n{char*width}")
    else:
        print(char * width)

def log_optuna(fold, encoding, n_features, model_name, best_value, best_params):
    print(f"      Optuna best MSE={best_value:.2f}  params={best_params}")

def log_fold_result(fold, encoding, n_features, model_name, metrics):
    print(f"      ✅ MAE={metrics['mae']:>7.2f}  SMAPE={metrics['smape']:>6.2f}%  R²={metrics['r2']:>6.4f}  RMSE={metrics['rmse']:>7.2f}")

# =============================================================================
# SUMMARY TABLES  (one per N, always including only_structured baseline)
# =============================================================================

def log_final_summary_tables():
    """Read ALL results from the DB and print one per-N table with ✅ best markers.
    only_structured (n_features=0) is included as a labelled baseline row in
    every table regardless of the current N so comparisons are always grounded.
    """
    sep("FINAL SUMMARY  —  all results in DB  (mean ± std across folds)")

    if not os.path.exists(RESULT_DB):
        print("  No results database found.")
        return

    with sqlite3.connect(RESULT_DB) as conn:
        try:
            df_all = pd.read_sql("SELECT * FROM metrics", conn)
        except Exception as e:
            print(f"  Could not read metrics table: {e}")
            return

    if df_all.empty:
        print("  No results found in DB.")
        return

    # Aggregate mean ± std across folds for every (encoding, n_features, model)
    df_agg = (
        df_all
        .groupby(['encoding', 'n_features', 'model'])[['mae', 'smape', 'r2', 'rmse']]
        .agg(['mean', 'std'])
    )
    df_agg.columns = [f"{m}_{s}" for m, s in df_agg.columns]
    df_agg = df_agg.reset_index()

    baseline = df_agg[df_agg['encoding'] == 'only_structured'].copy()

    # All distinct N values present in the DB
    all_n = sorted(df_all['n_features'].unique())

    # Metric metadata: display label, column key, "best" direction
    metric_meta = [
        ('MAE ↓',   'mae',   'min'),
        ('SMAPE ↓', 'smape', 'min'),
        ('R² ↑',    'r2',    'max'),
        ('RMSE ↓',  'rmse',  'min'),
    ]

    # Column widths (each metric cell = mean±std + optional ✅ marker)
    W_ENC   = 22
    W_N     = 6
    W_MODEL = 14
    W_CELL  = 18        # wide enough for "1234.56±123.45✅"

    def _header():
        h  = f"  {'Encoding':<{W_ENC}} {'N':>{W_N}} {'Model':<{W_MODEL}}"
        for label, _, _ in metric_meta:
            h += f"  {label:>{W_CELL}}"
        return h

    def _divider():
        return "  " + "-" * (W_ENC + 1 + W_N + 1 + W_MODEL + len(metric_meta) * (W_CELL + 2))

    def _row(enc, n_val, model, row_data, best_vals):
        n_str = 'BASE' if n_val == 0 else str(int(n_val))
        line  = f"  {enc:<{W_ENC}} {n_str:>{W_N}} {model:<{W_MODEL}}"
        for _, key, direction in metric_meta:
            mean_v = row_data[f'{key}_mean']
            std_v  = row_data[f'{key}_std']
            is_best = abs(mean_v - best_vals[key]) < 1e-9
            mark   = '✅' if is_best else '  '
            cell   = f"{mean_v:7.2f}±{std_v:5.2f}{mark}"
            line  += f"  {cell:>{W_CELL}}"
        return line

    for N in all_n:
        if N == 0:
            # Baseline is shown as a row inside every N-level table — no standalone table needed.
            continue
        text_part     = df_agg[df_agg['n_features'] == N].copy()
        baseline_copy = baseline.copy()   # n_features stays 0 → shown as BASE
        subset        = pd.concat([baseline_copy, text_part], ignore_index=True)
        title         = f"N = {N} text features  (+  only_structured baseline)"

        sep(f"SUMMARY TABLE — {title}")

        if subset.empty:
            print("  No rows.")
            continue

        # Best value per metric (across all rows in this table)
        best_vals = {}
        for _, key, direction in metric_meta:
            col = f'{key}_mean'
            best_vals[key] = subset[col].min() if direction == 'min' else subset[col].max()

        # Sort by MAE mean ascending
        subset_sorted = subset.sort_values('mae_mean')

        print(_header())
        print(_divider())
        for _, row in subset_sorted.iterrows():
            print(_row(row['encoding'], row['n_features'], row['model'], row, best_vals))

        # Best config per metric
        print()
        for label, key, direction in metric_meta:
            col = f'{key}_mean'
            idx = subset[col].idxmin() if direction == 'min' else subset[col].idxmax()
            b   = subset.loc[idx]
            n_s = 'BASE' if b['n_features'] == 0 else str(int(b['n_features']))
            print(f"  Best {label:<10}: encoding={b['encoding']}  N={n_s}  model={b['model']}  {key}={b[col]:.4f}")

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
        print(f"  ⚠  Could not read text_config: {e} — falling back to content detection...")
        try:
            with sqlite3.connect(INPUT_DB) as conn:
                active = _detect_from_content(conn, active)
        except Exception as e2:
            print(f"  ⚠  Content detection failed: {e2} — defaulting to scheduled_procedure only.")
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
        return active if active else default
    except Exception:
        return default

# =============================================================================
# AUTO-DETECT FEATURE COUNTS
# =============================================================================

def detect_feature_counts():
    try:
        with sqlite3.connect(INPUT_DB) as conn:
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            if 'feature_config' not in tables:
                print("  ⚠  'feature_config' not found — using default feature counts.")
                return DEFAULT_FEATURE_COUNTS
            row = conn.execute("SELECT feature_counts FROM feature_config").fetchone()
            counts = sorted(set(int(x.strip()) for x in row[0].split(',') if x.strip().isdigit()))
            print(f"  ✅ Detected from 'feature_config' metadata table.")
            return counts if counts else DEFAULT_FEATURE_COUNTS
    except Exception as e:
        print(f"  ⚠  Could not read feature_config: {e} — using default feature counts.")
        return DEFAULT_FEATURE_COUNTS

# =============================================================================
# EXISTING RESULT DETECTION
# Returns set of (fold, encoding, n_features, model) tuples already in DB.
# n_features is stored as int; only_structured uses 0.
# =============================================================================

def get_existing_results():
    if not os.path.exists(RESULT_DB):
        return set()
    try:
        with sqlite3.connect(RESULT_DB) as conn:
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            if 'metrics' not in tables:
                return set()
            rows = conn.execute("SELECT fold, encoding, n_features, model FROM metrics").fetchall()
        return set((int(r[0]), r[1], int(r[2]), r[3]) for r in rows)
    except Exception:
        return set()

def result_exists(existing, fold, encoding, n_features, model):
    """Returns True only when REPLACE_EXISTING is False AND the combo is in the DB."""
    if REPLACE_EXISTING:
        return False
    return (int(fold), encoding, int(n_features), model) in existing

# =============================================================================
# DATABASE HELPERS
# =============================================================================

def load_matrix(fold, split, encoding_key):
    with sqlite3.connect(ENCODED_DB) as conn:
        row = conn.execute("SELECT rows, cols, dtype, data FROM encoded_matrices WHERE fold=? AND split=? AND encoding=?", (fold, split, encoding_key)).fetchone()
    if row is None:
        raise ValueError(f"Matrix not found: fold={fold} split={split} encoding={encoding_key}")
    return np.frombuffer(row[3], dtype=row[2]).reshape(row[0], row[1]).copy()

def load_target(fold, split):
    with sqlite3.connect(ENCODED_DB) as conn:
        row = conn.execute("SELECT rows, dtype, data FROM encoded_targets WHERE fold=? AND split=?", (fold, split)).fetchone()
    if row is None:
        raise ValueError(f"Target not found: fold={fold} split={split}")
    return np.frombuffer(row[2], dtype=row[1]).copy()

def load_fold_indices():
    with sqlite3.connect(INPUT_DB) as conn:
        fold_df = pd.read_sql("SELECT * FROM fold_indices", conn)
    indices = {}
    for fold in range(N_SPLITS):
        train_idx = fold_df[(fold_df['fold'] == fold) & (fold_df['split'] == 'train')]['row_index'].values
        val_idx   = fold_df[(fold_df['fold'] == fold) & (fold_df['split'] == 'val')]['row_index'].values
        indices[fold] = (train_idx, val_idx)
    return indices

def save_db(df, table):
    with sqlite3.connect(RESULT_DB) as conn:
        df.to_sql(table, conn, if_exists='append', index=False)

def delete_combo(fold, encoding, n_features, model):
    """Delete all rows for this (fold, encoding, n_features, model) before re-saving."""
    with sqlite3.connect(RESULT_DB) as conn:
        for tbl in ['metrics', 'predictions', 'feature_importance', 'hyperparameter', 'timing']:
            cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (tbl,))
            if cur.fetchone():
                conn.execute(f"DELETE FROM {tbl} WHERE fold=? AND encoding=? AND n_features=? AND model=?", (int(fold), encoding, int(n_features), model))
        conn.commit()

def init_result_db():
    """Create result tables if they don't exist — does NOT wipe existing data."""
    with sqlite3.connect(RESULT_DB) as conn:
        for ddl in [
            "CREATE TABLE IF NOT EXISTS metrics (fold INTEGER, encoding TEXT, n_features INTEGER, model TEXT, mse REAL, rmse REAL, mae REAL, r2 REAL, mape REAL, smape REAL, mean_error REAL, std_error REAL, ci95_low REAL, ci95_high REAL)",
            "CREATE TABLE IF NOT EXISTS predictions (fold INTEGER, encoding TEXT, n_features INTEGER, model TEXT, actual TEXT, predicted TEXT)",
            "CREATE TABLE IF NOT EXISTS feature_importance (fold INTEGER, encoding TEXT, n_features INTEGER, model TEXT, importances TEXT)",
            "CREATE TABLE IF NOT EXISTS hyperparameter (fold INTEGER, encoding TEXT, n_features INTEGER, model TEXT, params TEXT)",
            "CREATE TABLE IF NOT EXISTS timing (fold INTEGER, encoding TEXT, n_features INTEGER, model TEXT, hpo_seconds REAL, fit_seconds REAL, predict_seconds REAL, total_seconds REAL)",
        ]:
            conn.execute(ddl)
        conn.commit()

# =============================================================================
# BERT PCA  (fold-wise, fit on training data only)
# =============================================================================

def bert_pca_features(bert_cache, active_text_cols, method, train_idx, val_idx, N):
    parts_tr, parts_va = [], []
    for col in active_text_cols:
        emb    = bert_cache[(method, col)]
        emb_tr = emb[train_idx].astype(np.float32)
        emb_va = emb[val_idx].astype(np.float32)
        n_components = min(N, emb_tr.shape[1], emb_tr.shape[0] - 1)
        pca          = PCA(n_components=n_components, random_state=RANDOM_STATE)
        emb_tr_pca   = pca.fit_transform(emb_tr)
        emb_va_pca   = pca.transform(emb_va)
        if emb_tr_pca.shape[1] < N:
            pad        = N - emb_tr_pca.shape[1]
            emb_tr_pca = np.hstack([emb_tr_pca, np.zeros((len(emb_tr_pca), pad), dtype=np.float32)])
            emb_va_pca = np.hstack([emb_va_pca, np.zeros((len(emb_va_pca), pad), dtype=np.float32)])
        var_explained = pca.explained_variance_ratio_.sum() * 100
        print(f"      PCA({N}) on {col}: {var_explained:.1f}% variance explained  (fit on {len(emb_tr):,} train rows)")
        parts_tr.append(emb_tr_pca)
        parts_va.append(emb_va_pca)
    return np.hstack(parts_tr).astype(np.float32), np.hstack(parts_va).astype(np.float32)

# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true, float), np.array(y_pred, float)
    errs  = y_true - y_pred
    mse   = mean_squared_error(y_true, y_pred)
    mae   = mean_absolute_error(y_true, y_pred)
    r2    = r2_score(y_true, y_pred)
    nz    = y_true != 0
    mape  = np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100 if nz.any() else np.nan
    denom = np.abs(y_true) + np.abs(y_pred)
    vld   = denom != 0
    smape = 100 * np.mean(2 * np.abs(y_true[vld] - y_pred[vld]) / denom[vld]) if vld.any() else np.nan
    mu, sd = np.mean(errs), np.std(errs)
    ci = t.interval(0.95, len(errs)-1, loc=mu, scale=sd/np.sqrt(len(errs)))
    return {'mse': mse, 'rmse': np.sqrt(mse), 'mae': mae, 'r2': r2, 'mape': mape, 'smape': smape, 'mean_error': mu, 'std_error': sd, 'ci95_low': ci[0], 'ci95_high': ci[1]}

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

def build_model(name, params=None):
    p = params or {}
    if name == 'linear':       return LinearRegression()
    if name == 'ridge':        return Ridge(alpha=p.get('alpha', 1.0))
    if name == 'lasso':        return Lasso(alpha=p.get('alpha', 1.0))
    if name == 'randomforest': return RandomForestRegressor(**p)
    if name == 'xgboost':      return xgb.XGBRegressor(**p)
    if name == 'mlp':
        mdl = Sequential([
            Input(shape=(p['input_dim'],)),
            Dense(p.get('units1', 64), activation='relu'),
            Dropout(p.get('dropout1', 0.0)),
            Dense(p.get('units2', 32), activation='relu'),
            Dropout(p.get('dropout2', 0.0)),
            Dense(1, dtype='float32'),
        ])
        mdl.compile(loss='mse', optimizer=Adam(p.get('lr', 1e-3)))
        return mdl
    raise ValueError(f"Unknown model: {name}")

def fit_predict(name, mdl, Xtr, Xtr_sc, Xva, Xva_sc, ytr, yva=None, final=False):
    if name == 'mlp':
        Xtr_f = Xtr_sc.astype(np.float32)
        ytr_f = ytr.astype(np.float32)
        if final and yva is not None:
            Xva_f    = Xva_sc.astype(np.float32)
            yva_f    = yva.astype(np.float32)
            train_ds = (tf.data.Dataset.from_tensor_slices((Xtr_f, ytr_f)).shuffle(buffer_size=min(10_000, len(ytr_f)), seed=RANDOM_STATE).batch(512).prefetch(tf.data.AUTOTUNE))
            val_ds   = (tf.data.Dataset.from_tensor_slices((Xva_f, yva_f)).batch(512).prefetch(tf.data.AUTOTUNE))
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0),
            ]
            mdl.fit(train_ds, validation_data=val_ds, epochs=200, callbacks=callbacks, verbose=0)
            return mdl.predict(val_ds, verbose=0).flatten()
        else:
            mdl.fit(Xtr_f, ytr_f, epochs=10, batch_size=512, verbose=0)
            return mdl.predict(Xva_sc.astype(np.float32), verbose=0).flatten()
    if name in ('randomforest', 'xgboost'):
        mdl.fit(Xtr, ytr)
        return mdl.predict(Xva)
    mdl.fit(Xtr_sc, ytr)
    return mdl.predict(Xva_sc)

class _KerasWrapper(BaseEstimator):
    def __init__(self, mdl): self.mdl = mdl
    def fit(self, X, y):
        self.mdl.fit(X.astype(np.float32), y.astype(np.float32), epochs=10, batch_size=512, verbose=0); return self
    def predict(self, X):
        return self.mdl.predict(X.astype(np.float32), verbose=0).flatten()

def make_objective(name, Xtr, Xtr_sc, Xva, Xva_sc, ytr, yva):
    def obj(trial):
        if name in ('ridge', 'lasso'):
            p = {'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True)}
        elif name == 'randomforest':
            p = {'n_estimators': trial.suggest_int('n_estimators', 100, 200), 'max_depth': trial.suggest_int('max_depth', 3, 10), 'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']), 'random_state': RANDOM_STATE, 'n_jobs': -1}
        elif name == 'xgboost':
            p = {'n_estimators': trial.suggest_int('n_estimators', 100, 200), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1), 'max_depth': trial.suggest_int('max_depth', 3, 6), 'random_state': RANDOM_STATE, 'n_jobs': -1}
        elif name == 'mlp':
            tf.keras.backend.clear_session()
            p = {'input_dim': Xtr_sc.shape[1], 'units1': trial.suggest_int('units1', 32, 128), 'units2': trial.suggest_int('units2', 16, 64), 'dropout1': trial.suggest_float('dropout1', 0.0, 0.3), 'dropout2': trial.suggest_float('dropout2', 0.0, 0.3), 'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True)}
        else:
            p = {}
        mdl   = build_model(name, p)
        preds = fit_predict(name, mdl, Xtr, Xtr_sc, Xva, Xva_sc, ytr, yva=None, final=False)
        return mean_squared_error(yva, preds)
    return obj

# =============================================================================
# SHARED SAVE HELPER
# =============================================================================

def save_run(fold, encoding, n_features, model_name, mdl, preds, y_val, X_va_sc, best_p, hpo_secs, fit_secs, pred_secs, total_secs):
    # If replacing, delete any existing rows for this combo before inserting
    if REPLACE_EXISTING:
        delete_combo(fold, encoding, n_features, model_name)

    metrics = compute_metrics(y_val, preds)
    metrics.update({'fold': fold, 'encoding': encoding, 'n_features': n_features, 'model': model_name})
    log_fold_result(fold, encoding, n_features, model_name, metrics)

    save_db(pd.DataFrame([metrics]), 'metrics')
    save_db(pd.DataFrame([{'fold': fold, 'encoding': encoding, 'n_features': n_features, 'model': model_name, 'actual': str(y_val.tolist()), 'predicted': str(preds.tolist())}]), 'predictions')
    save_db(pd.DataFrame([{'fold': fold, 'encoding': encoding, 'n_features': n_features, 'model': model_name, 'hpo_seconds': round(hpo_secs, 3), 'fit_seconds': round(fit_secs, 3), 'predict_seconds': round(pred_secs, 3), 'total_seconds': round(total_secs, 3)}]), 'timing')
    print(f"      ⏱  HPO={hpo_secs:.1f}s  fit={fit_secs:.1f}s  total={total_secs:.1f}s")

    imps = {}
    try:
        if model_name in ('randomforest', 'xgboost'):
            imps = {f'f{i}': float(v) for i, v in enumerate(mdl.feature_importances_)}
        elif model_name in ('linear', 'ridge', 'lasso'):
            imps = {f'f{i}': float(v) for i, v in enumerate(mdl.coef_)}
        elif model_name == 'mlp':
            res  = permutation_importance(_KerasWrapper(mdl), X_va_sc, y_val, n_repeats=1, random_state=RANDOM_STATE, scoring='neg_mean_squared_error')
            imps = {f'f{i}': float(v) for i, v in enumerate(res.importances_mean)}
    except Exception as e:
        print(f"      ⚠ Feature importance failed: {e}")

    save_db(pd.DataFrame([{'fold': fold, 'encoding': encoding, 'n_features': n_features, 'model': model_name, 'importances': str(imps)}]), 'feature_importance')
    save_db(pd.DataFrame([{'fold': fold, 'encoding': encoding, 'n_features': n_features, 'model': model_name, 'params': str({k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) for k, v in best_p.items()})}]), 'hyperparameter')
    return metrics

# =============================================================================
# STAGE 1 — STARTUP
# =============================================================================
sep("04. Modeling.py")
init_result_db()   # create tables if absent — never wipes existing results

# =============================================================================
# STAGE 2 — EXISTING RESULT DETECTION + SKIP / REPLACE PROMPT
# =============================================================================
sep("EXISTING RESULT CHECK")

existing_results = get_existing_results()

if existing_results:
    print(f"\n  Found {len(existing_results):,} existing result combos in {RESULT_DB}.")
    print()
    print(f"  [S]  Skip existing combos and run only new ones  ← DEFAULT")
    print(f"  [R]  Replace existing combos (re-run and overwrite)")
    print()
    choice = input("  Your choice (S / R, or press Enter to skip): ").strip().upper()
    if choice == 'R':
        REPLACE_EXISTING = True
        print(f"  → Replace mode: existing results will be overwritten.")
    else:
        REPLACE_EXISTING = False
        print(f"  → Skip mode: existing results will be preserved.")
else:
    print(f"  No existing results found — fresh run.")
    REPLACE_EXISTING = False

# =============================================================================
# STAGE 3 — AUTO-DETECTION
# =============================================================================
sep("AUTO-DETECTION  —  reading config from database")
ACTIVE_TEXT_COLS = detect_active_text_cols()
print(f"\n  Active text columns : {ACTIVE_TEXT_COLS}")

FEATURE_COUNTS = detect_feature_counts()
print(f"  Feature counts      : {FEATURE_COUNTS}")

# =============================================================================
# STAGE 4 — MODEL SELECTION
# =============================================================================
sep("MODEL SELECTION")
model_emojis = {
    'linear':       '📏 Linear Regression',
    'ridge':        '🏔️  Ridge Regression',
    'lasso':        '🪢 Lasso Regression',
    'randomforest': '🌲 Random Forest',
    'xgboost':      '⚡ XGBoost',
    'mlp':          '🧠 Neural Network (MLP)',
}
print(f"  [0]  🚀 Run ALL models  [default]")
for i, m in enumerate(ALL_MODELS, 1):
    print(f"  [{i}]  {model_emojis[m]}")
print()
user_input = input("  🎯 Select models by number (e.g. 1,3,5 or press Enter for all): ").strip()

if user_input == '' or user_input == '0':
    MODEL_LIST = ALL_MODELS
else:
    MODEL_LIST = []
    for part in user_input.split(','):
        part = part.strip()
        if part.isdigit() and 1 <= int(part) <= len(ALL_MODELS):
            MODEL_LIST.append(ALL_MODELS[int(part) - 1])
        else:
            print(f"  ⚠ Ignored unrecognized input: '{part}'")
    MODEL_LIST = list(dict.fromkeys(MODEL_LIST))
    if not MODEL_LIST:
        print("  No valid models selected — exiting.")
        _tee.close()
        sys.exit(0)

print(f"  Running models: {MODEL_LIST}")

# =============================================================================
# STAGE 5 — OPEN PER-MODEL LOG FILES
# One .log file per selected model, written alongside the main log.
# All output that appears on stdout while that model is running is mirrored
# to its dedicated log via _Tee.set_model_log().
# =============================================================================
sep("PER-MODEL LOG FILES")
model_log_files = {}
for m in MODEL_LIST:
    log_path = f"./results/04_modeling_{m}.log"
    model_log_files[m] = open(log_path, "w", encoding="utf-8", buffering=1)
    print(f"  Opened: {log_path}")

# =============================================================================
# STAGE 6 — VERIFY INPUTS
# =============================================================================
sep("VERIFY INPUTS")

if not os.path.exists(ENCODED_DB):
    print(f"  ❌ Missing: {ENCODED_DB}  →  Run 03. Fold_Encoding.py first.")
    _tee.close()
    sys.exit(1)
print(f"  ✅ {ENCODED_DB} found")

bert_cache     = {}
BERT_ENCODINGS = []
for method in BERT_METHODS:
    method_ok = True
    for col in ACTIVE_TEXT_COLS:
        fpath = os.path.join(BERT_DIR, f"{method}_{col}.npy")
        if not os.path.exists(fpath):
            print(f"  ⚠  Missing: {fpath}  — {method} encoding will be skipped.")
            method_ok = False
            break
        arr = np.load(fpath)
        bert_cache[(method, col)] = arr
        print(f"  ✅ Loaded {method}_{col}.npy  shape={arr.shape}")
    if method_ok:
        BERT_ENCODINGS.append(method)
        print(f"  ✅ {method}: all {len(ACTIVE_TEXT_COLS)} file(s) found — PCA encoding enabled.")
    else:
        print(f"  ❌ {method}: skipped.")

fold_indices = load_fold_indices()
print(f"  ✅ Fold indices loaded")

# =============================================================================
# STAGE 7 — ENCODING MATRIX VERIFICATION
# =============================================================================
sep("ENCODING MATRIX SHAPES  (from fold_encoded.db)")
with sqlite3.connect(ENCODED_DB) as conn:
    db_rows = conn.execute("SELECT fold, split, encoding, rows, cols FROM encoded_matrices ORDER BY fold, split, encoding").fetchall()
print(f"\n  {'Fold':<6} {'Split':<8} {'Encoding':<28} {'Rows':>8} {'Cols':>6}")
print(f"  {'-'*60}")
for r in db_rows:
    print(f"  {r[0]:<6} {r[1]:<8} {r[2]:<28} {r[3]:>8,} {r[4]:>6}")

# =============================================================================
# STAGE 8 — MAIN LOOP  fold → N → encoding → model
# =============================================================================
sep("MAIN LOOP  —  fold → N → encoding → model")
all_metrics   = []
skipped_count = 0

for fold in range(N_SPLITS):
    train_idx, val_idx = fold_indices[fold]
    sep(f"FOLD {fold}  —  train={len(train_idx):,}  val={len(val_idx):,}", char='-')

    y_train = load_target(fold, 'train')
    y_val   = load_target(fold, 'val')
    print(f"  y_train: mean={y_train.mean():.1f}  std={y_train.std():.1f}")
    print(f"  y_val:   mean={y_val.mean():.1f}  std={y_val.std():.1f}")

    X_struct_tr = load_matrix(fold, 'train', 'only_structured')
    X_struct_va = load_matrix(fold, 'val',   'only_structured')

    # =========================================================================
    # A) only_structured  (n_features = 0, no N dependency)
    #    This is the mandatory BASELINE — always run regardless of N.
    # =========================================================================
    sep(f"  FOLD {fold} — only_structured  (Baseline)", char='-')
    scaler_s  = MinMaxScaler()
    X_tr_sc_s = scaler_s.fit_transform(X_struct_tr)
    X_va_sc_s = scaler_s.transform(X_struct_va)

    for model_name in MODEL_LIST:
        _tee.set_model_log(model_log_files[model_name])

        if result_exists(existing_results, fold, 'only_structured', 0, model_name):
            print(f"    ⏭  only_structured / {model_name}: already done — skipping.")
            skipped_count += 1
            _tee.set_model_log(None)
            continue

        print(f"\n    🔧 only_structured / {model_name}")
        t_total = time.perf_counter()
        try:
            best_p, hpo_secs = {}, 0.0
            if model_name != 'linear':
                t0 = time.perf_counter()
                study = optuna.create_study(direction='minimize')
                study.optimize(make_objective(model_name, X_struct_tr, X_tr_sc_s, X_struct_va, X_va_sc_s, y_train, y_val), n_trials=N_TRIALS)
                hpo_secs = time.perf_counter() - t0
                best_p   = study.best_trial.params
                if model_name == 'mlp':
                    best_p['input_dim'] = X_tr_sc_s.shape[1]
                    tf.keras.backend.clear_session()
                log_optuna(fold, 'only_structured', 0, model_name, study.best_trial.value, best_p)

            mdl   = build_model(model_name, best_p)
            t0    = time.perf_counter()
            preds = fit_predict(model_name, mdl, X_struct_tr, X_tr_sc_s, X_struct_va, X_va_sc_s, y_train, yva=y_val, final=True)
            fit_secs   = time.perf_counter() - t0
            total_secs = time.perf_counter() - t_total

            m = save_run(fold, 'only_structured', 0, model_name, mdl, preds, y_val, X_va_sc_s, best_p, hpo_secs, fit_secs, 0.0, total_secs)
            all_metrics.append(m)
        except Exception as e:
            print(f"      🚫 Error: {e}")

        _tee.set_model_log(None)

    # =========================================================================
    # B) Text encodings — loop over N × (classical + BERT)
    # =========================================================================
    for N in FEATURE_COUNTS:
        print(f"\n  ── N = {N} ──────────────────────────────────────────────")

        # --- Classical -------------------------------------------------------
        for encoding in CLASSICAL_ENCODINGS:
            enc_key = f"{encoding}_n{N}"
            try:
                X_tr = load_matrix(fold, 'train', enc_key)
                X_va = load_matrix(fold, 'val',   enc_key)
            except ValueError as e:
                print(f"    ⚠ {e} — skipping.")
                continue

            scaler  = MinMaxScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_va_sc = scaler.transform(X_va)

            for model_name in MODEL_LIST:
                _tee.set_model_log(model_log_files[model_name])

                if result_exists(existing_results, fold, encoding, N, model_name):
                    print(f"    ⏭  {enc_key} / {model_name}: already done — skipping.")
                    skipped_count += 1
                    _tee.set_model_log(None)
                    continue

                print(f"\n    🔧 {enc_key} / {model_name}")
                t_total = time.perf_counter()
                try:
                    best_p, hpo_secs = {}, 0.0
                    if model_name != 'linear':
                        t0 = time.perf_counter()
                        study = optuna.create_study(direction='minimize')
                        study.optimize(make_objective(model_name, X_tr, X_tr_sc, X_va, X_va_sc, y_train, y_val), n_trials=N_TRIALS)
                        hpo_secs = time.perf_counter() - t0
                        best_p   = study.best_trial.params
                        if model_name == 'mlp':
                            best_p['input_dim'] = X_tr_sc.shape[1]
                            tf.keras.backend.clear_session()
                        log_optuna(fold, encoding, N, model_name, study.best_trial.value, best_p)

                    mdl   = build_model(model_name, best_p)
                    t0    = time.perf_counter()
                    preds = fit_predict(model_name, mdl, X_tr, X_tr_sc, X_va, X_va_sc, y_train, yva=y_val, final=True)
                    fit_secs   = time.perf_counter() - t0
                    total_secs = time.perf_counter() - t_total

                    m = save_run(fold, encoding, N, model_name, mdl, preds, y_val, X_va_sc, best_p, hpo_secs, fit_secs, 0.0, total_secs)
                    all_metrics.append(m)
                except Exception as e:
                    print(f"      🚫 Error: {e}")

                _tee.set_model_log(None)

        # --- BERT (PCA fold-wise) --------------------------------------------
        for method in BERT_ENCODINGS:
            all_done = all(result_exists(existing_results, fold, method, N, m) for m in MODEL_LIST)
            if all_done:
                print(f"    ⏭  {method} N={N}: all models already done — skipping PCA.")
                skipped_count += len(MODEL_LIST)
                continue

            print(f"\n  Encoding: {method.upper()}  N={N}  (PCA, fit on train only)")
            t_pca = time.perf_counter()
            X_bert_tr, X_bert_va = bert_pca_features(bert_cache, ACTIVE_TEXT_COLS, method, train_idx, val_idx, N)
            pca_secs = time.perf_counter() - t_pca

            X_tr = np.hstack([X_struct_tr, X_bert_tr])
            X_va = np.hstack([X_struct_va, X_bert_va])
            print(f"    PCA time: {pca_secs:.2f}s  train={X_tr.shape}  val={X_va.shape}")

            scaler  = MinMaxScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_va_sc = scaler.transform(X_va)

            for model_name in MODEL_LIST:
                _tee.set_model_log(model_log_files[model_name])

                if result_exists(existing_results, fold, method, N, model_name):
                    print(f"    ⏭  {method} N={N} / {model_name}: already done — skipping.")
                    skipped_count += 1
                    _tee.set_model_log(None)
                    continue

                print(f"\n    🔧 {method} N={N} / {model_name}")
                t_total = time.perf_counter()
                try:
                    best_p, hpo_secs = {}, 0.0
                    if model_name != 'linear':
                        t0 = time.perf_counter()
                        study = optuna.create_study(direction='minimize')
                        study.optimize(make_objective(model_name, X_tr, X_tr_sc, X_va, X_va_sc, y_train, y_val), n_trials=N_TRIALS)
                        hpo_secs = time.perf_counter() - t0
                        best_p   = study.best_trial.params
                        if model_name == 'mlp':
                            best_p['input_dim'] = X_tr_sc.shape[1]
                            tf.keras.backend.clear_session()
                        log_optuna(fold, method, N, model_name, study.best_trial.value, best_p)

                    mdl   = build_model(model_name, best_p)
                    t0    = time.perf_counter()
                    preds = fit_predict(model_name, mdl, X_tr, X_tr_sc, X_va, X_va_sc, y_train, yva=y_val, final=True)
                    fit_secs   = time.perf_counter() - t0
                    total_secs = time.perf_counter() - t_total

                    m = save_run(fold, method, N, model_name, mdl, preds, y_val, X_va_sc, best_p, hpo_secs, fit_secs, pca_secs, total_secs)
                    all_metrics.append(m)
                except Exception as e:
                    print(f"      🚫 Error: {e}")

                _tee.set_model_log(None)

# =============================================================================
# STAGE 9 — CLOSE PER-MODEL LOG FILES
# =============================================================================
_tee.set_model_log(None)
for m, fobj in model_log_files.items():
    fobj.close()
    print(f"  Closed: ./results/04_modeling_{m}.log")

# =============================================================================
# STAGE 10 — FINAL SUMMARY TABLES
# =============================================================================
sep("RUN SUMMARY")
print(f"  New results computed : {len(all_metrics)}")
print(f"  Combinations skipped : {skipped_count}  (already in DB)")
print(f"  Replace mode active  : {REPLACE_EXISTING}")
if all_metrics:
    print(f"  ℹ  Summary tables below include ALL results stored in {RESULT_DB},")
    print(f"     not only the {len(all_metrics)} new results from this session.")
else:
    print(f"  ℹ  No new results were computed this session.")
    print(f"     Summary tables are generated from all existing results in {RESULT_DB}.")

# Always print summary tables — reads directly from the DB so the output
# is complete regardless of whether any new results were computed this run.
log_final_summary_tables()

sep("FILES WRITTEN")
print(f"  Main log   : ./results/04_modeling.log")
for m in MODEL_LIST:
    print(f"  Model log  : ./results/04_modeling_{m}.log")
print(f"  Result DB  : {RESULT_DB}")

_tee.close()
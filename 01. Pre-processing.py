# =============================================================================
# LOGGER SETUP
# =============================================================================
import sys

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

import os
os.makedirs("./results", exist_ok=True)
_tee = _Tee("./results/01_preprocessing.log")
sys.stdout = _tee

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import sqlite3
import re
from datetime import datetime

# =============================================================================
# CONSTANTS
# =============================================================================
RAW_CSV      = './data/casetime.csv'
OUTPUT_DB    = './data/surgical_data.db'
OUTPUT_TABLE = 'Clean'

TARGET = 'actual_casetime_minutes'

TIME_COLS = [
    'procedure_minutes', 'actual_casetime_minutes',
    'procedure_time', 'induction_time', 'emergence_time', 'scheduled_duration'
]

# All four raw text columns that may appear in the dataset.
ALL_TEXT_COLS = ['scheduled_procedure', 'procedure', 'operative_dx', 'most_responsible_dx']

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def sep(title='', width=70, char='='):
    if title:
        print(f"\n{char*width}")
        print(f"  {title}")
        print(f"{char*width}")
    else:
        print(char * width)

def save_to_db(df, db_file, table_name):
    with sqlite3.connect(db_file) as conn:
        df.to_sql(table_name, conn, if_exists='replace', index=False)
    print(f"  Saved '{table_name}' → {db_file}   shape={df.shape}")

def clean_missing(df):
    pat = re.compile(r'^\s*(nan|none|null|na|n/a|missing|unknown|\?|-|)\s*$', re.IGNORECASE)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].apply(lambda x: np.nan if isinstance(x, str) and pat.match(x) else x)
    for col in df.select_dtypes(include='datetime64').columns:
        orig = df[col].copy()
        mask = df[col].astype(str).apply(lambda x: bool(pat.match(x)))
        df[col] = orig
        df.loc[mask, col] = pd.NaT
    return df

def normalize_strings(df):
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip().str.lower().replace(r'^\s*$', np.nan, regex=True)
    return df

def handle_rare(df, col, thresh_pct=0.5, drop=False, label='Other'):
    freq = df[col].value_counts(normalize=True) * 100
    rare = freq[freq < thresh_pct].index
    if drop:
        return df[~df[col].isin(rare)].copy()
    df = df.copy()
    df[col] = df[col].where(~df[col].isin(rare), label)
    return df

def onehot(df, cols, drop_first=False):
    df = pd.get_dummies(df, columns=cols, prefix_sep='__', dummy_na=False, drop_first=drop_first)
    for c in df.columns:
        if df[c].dtype == bool or (df[c].dropna().isin([0, 1]).all() and not df[c].isna().any()):
            df[c] = df[c].astype(int)
    return df

def print_missing_summary(df, label=''):
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print(f"  No missing values{' — ' + label if label else ''}.")
        return
    total = len(df)
    print(f"\n  Missing value summary{' — ' + label if label else ''}:")
    print(f"  {'Column':<45} {'Missing':>8} {'%':>8}")
    print(f"  {'-'*63}")
    for col, cnt in missing.items():
        print(f"  {col:<45} {cnt:>8,} {cnt/total*100:>7.2f}%")

def print_freq_table(series, label='', top_n=20):
    vc = series.value_counts(dropna=False)
    total = len(series)
    print(f"\n  Frequency table — {label or series.name}  (top {top_n} of {len(vc)} unique values):")
    print(f"  {'Value':<45} {'Count':>8} {'%':>8}")
    print(f"  {'-'*63}")
    for val, cnt in vc.head(top_n).items():
        print(f"  {str(val):<45} {cnt:>8,} {cnt/total*100:>7.2f}%")

def print_numeric_summary(df, cols, label=''):
    print(f"\n  Numeric summary{' — ' + label if label else ''}:")
    print(f"  {'Column':<35} {'N':>8} {'Mean':>9} {'SD':>9} {'Min':>9} {'P25':>9} {'Median':>9} {'P75':>9} {'P90':>9} {'P95':>9} {'Max':>9} {'NaN':>6}")
    print(f"  {'-'*122}")
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        print(f"  {col:<35} {len(s):>8,} {s.mean():>9.2f} {s.std():>9.2f} {s.min():>9.2f} {s.quantile(0.25):>9.2f} {s.median():>9.2f} {s.quantile(0.75):>9.2f} {s.quantile(0.90):>9.2f} {s.quantile(0.95):>9.2f} {s.max():>9.2f} {df[col].isna().sum():>6,}")

def print_shape_log(df, label):
    print(f"  Shape after {label}: {df.shape[0]:,} rows × {df.shape[1]} columns")

# =============================================================================
# STAGE 0 — TEXT FEATURE SELECTION  (interactive)
# =============================================================================
sep("STAGE 0 — TEXT FEATURE SELECTION  (interactive)")

print("""
  Each text column is kept as-is — no columns are merged together.
  By default only 'scheduled_procedure' is included (pre-operative).

  ┌─────────────────────────────────────────────────────────────────┐
  │  Procedure text columns                                         │
  │    [ALWAYS]  scheduled_procedure   pre-operative   ✅ on        │
  │    [1]       procedure             post-operative  ❌ off       │
  ├─────────────────────────────────────────────────────────────────┤
  │  Diagnosis text columns                                         │
  │    [2]       operative_dx          pre-operative*  ⚠️  off      │
  │    [3]       most_responsible_dx   post-operative  ❌ off       │
  └─────────────────────────────────────────────────────────────────┘
  * operative_dx is pre-operative but may be less specific than the
    post-operative final coding.

  Enter numbers to add (e.g.  2  or  1,2,3) or press Enter for default:
""")

_raw_sel = input("  Your selection: ").strip()
_sel     = {p.strip() for p in _raw_sel.split(',') if p.strip().isdigit()} if _raw_sel else set()

USE_PROCEDURE_POST = '1' in _sel   # procedure            (post-operative)
USE_OPERATIVE_DX   = '2' in _sel   # operative_dx         (pre-operative, less updated)
USE_MOST_RESP_DX   = '3' in _sel   # most_responsible_dx  (post-operative)

# Build the list of text columns to keep — order is fixed for reproducibility.
ACTIVE_TEXT_COLS = ['scheduled_procedure']
if USE_PROCEDURE_POST: ACTIVE_TEXT_COLS.append('procedure')
if USE_OPERATIVE_DX:   ACTIVE_TEXT_COLS.append('operative_dx')
if USE_MOST_RESP_DX:   ACTIVE_TEXT_COLS.append('most_responsible_dx')

print(f"\n  ── Active text column configuration ──")
for col in ALL_TEXT_COLS:
    status = '✅  included' if col in ACTIVE_TEXT_COLS else '❌  excluded'
    tag    = '(pre-operative)' if col in ('scheduled_procedure', 'operative_dx') else '(post-operative)'
    always = '  [always]' if col == 'scheduled_procedure' else ''
    print(f"  {col:<30} {status}  {tag}{always}")

# =============================================================================
# STAGE 1 — LOAD & INITIAL CLEANING
# =============================================================================
sep("STAGE 1 — LOAD & INITIAL CLEANING")

df = pd.read_csv(RAW_CSV)
df.columns = df.columns.str.strip()
n_init = len(df)
print(f"  Loaded raw CSV: {n_init:,} rows × {df.shape[1]} columns")
print(f"  Columns: {df.columns.tolist()}")

df = normalize_strings(df)
df = clean_missing(df)
print_shape_log(df, "normalize + clean_missing")
print_missing_summary(df, "after initial cleaning")

# =============================================================================
# STAGE 2 — DATETIME FEATURES
# =============================================================================
sep("STAGE 2 — DATETIME FEATURES")

dt_all = [c for c in df.columns if any(x in c.lower() for x in ['dttm', 'date', 'time', 'minute'])]
before = len(df)
df.dropna(subset=dt_all, inplace=True)
print(f"  Removed {before - len(df):,} rows with missing datetime/time/minute values")
print_shape_log(df, "dropping missing datetimes")

parse_cols = [c for c in df.columns if any(x in c.lower() for x in ['dttm', 'date'])]
for c in parse_cols:
    df[c] = pd.to_datetime(df[c], errors='coerce')

df['procedure_time']     = (df['procedure_stop_dttm']  - df['procedure_start_dttm']).dt.total_seconds() / 60
df['induction_time']     = (df['procedure_start_dttm'] - df['OR_entered_dttm']).dt.total_seconds() / 60
df['emergence_time']     = (df['OR_left_dttm']         - df['procedure_stop_dttm']).dt.total_seconds() / 60
df['scheduled_duration'] = (df['scheduled_end_dttm']   - df['scheduled_start_dttm']).dt.total_seconds() / 60
df['scheduled_start_hour'] = df['scheduled_start_dttm'].dt.hour
df['or_entry_hour']        = df['OR_entered_dttm'].dt.hour
df['month_of_year']        = df['scheduled_start_dttm'].dt.month
df['day_of_week']          = df['scheduled_start_dttm'].dt.dayofweek

df.drop(columns=parse_cols, inplace=True)

print("\n  Derived duration columns:")
print("    procedure_time       = procedure_start_dttm → procedure_stop_dttm  (incision to closure)")
print("    induction_time       = OR_entered_dttm      → procedure_start_dttm (room-in to incision)")
print("    emergence_time       = procedure_stop_dttm  → OR_left_dttm         (closure to room-out)")
print("    scheduled_duration   = scheduled_start_dttm → scheduled_end_dttm  (scheduler estimate)")
print("    actual_casetime_minutes = raw Cerner field  (room-in to room-out)")
print("\n  NOTE: actual_casetime_minutes is the prediction TARGET.")
print("        It corresponds to the full OR occupancy window used by schedulers.")
print_shape_log(df, "datetime derivation")

# =============================================================================
# STAGE 3 — IMPLAUSIBLE DURATION FILTER  (0 – 2,880 min / 48 h, rule-based)
# =============================================================================
sep("STAGE 3 — IMPLAUSIBLE DURATION FILTER  (0 – 2,880 min / 48 h, rule-based)")

print("\n  Threshold justification — rows outside [0, 2880] min before filtering:")
print(f"  {'Column':<35} {'N':>8} {'Below 0':>10} {'Above 2880':>12} {'Total out':>12} {'99.9th pct':>12} {'99.99th pct':>13}")
print(f"  {'-'*104}")
for col in TIME_COLS:
    if col not in df.columns:
        continue
    s = df[col].dropna()
    n_b = (s < 0).sum()
    n_a = (s > 2880).sum()
    print(f"  {col:<35} {len(s):>8,} {n_b:>10,} {n_a:>12,} {n_b+n_a:>12,} {s.quantile(0.999):>12.1f} {s.quantile(0.9999):>13.1f}")

print()
for col in TIME_COLS:
    if col not in df.columns:
        continue
    before = len(df)
    df = df[df[col].between(0, 2880)]
    removed = before - len(df)
    if removed:
        print(f"  {col}: removed {removed:,} rows outside [0, 2880] min")

print_shape_log(df, "implausible duration filter")

# =============================================================================
# STAGE 4 — TARGET DISTRIBUTION  (actual_casetime_minutes, post-filter)
# =============================================================================
sep("STAGE 4 — TARGET DISTRIBUTION  (actual_casetime_minutes, post-filter)")

print("\n  Definition: room-in to room-out elapsed time (Cerner field 'actual_casetime_minutes')")
print("  This matches the scheduler's estimate field 'scheduled_duration' in definition.")
print_numeric_summary(df, ['actual_casetime_minutes', 'scheduled_duration', 'procedure_time', 'induction_time', 'emergence_time'])

s = df['actual_casetime_minutes'].dropna()
print(f"\n  actual_casetime_minutes — detailed percentiles:")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]:
    print(f"    {p:5.1f}th pct : {s.quantile(p/100):>8.1f} min")

# =============================================================================
# STAGE 5 — CLAMP INVALID AGE / BMI  (rule-based bounds)
# =============================================================================
sep("STAGE 5 — CLAMP INVALID AGE / BMI  (rule-based bounds)")

inv_age = ~df['age_at_discharge'].between(18, 130)
df.loc[inv_age, 'age_at_discharge'] = np.nan
print(f"  Clamped {inv_age.sum():,} age values outside [18, 130] → NaN  (will be imputed fold-wise)")

inv_bmi = ~df['avg_BMI'].between(5, 200)
df.loc[inv_bmi, 'avg_BMI'] = np.nan
print(f"  Clamped {inv_bmi.sum():,} BMI values outside [5, 200]   → NaN  (will be imputed fold-wise)")

print_numeric_summary(df, ['age_at_discharge', 'avg_BMI'], "after clamping")

# =============================================================================
# STAGE 6 — DROP IDENTIFIER COLUMNS
# =============================================================================
sep("STAGE 6 — DROP IDENTIFIER COLUMNS")
drop_ids = [c for c in ['case_id', 'patient_id', 'avg_wt_enct', 'avg_ht_enct', 'week_day'] if c in df.columns]
df.drop(columns=drop_ids, inplace=True)
print(f"  Dropped: {drop_ids}")
print_shape_log(df, "dropping identifiers")

# =============================================================================
# STAGE 7 — MISSINGNESS
# =============================================================================
sep("STAGE 7 — MISSINGNESS")
print_missing_summary(df, "before dropping non-imputable")

# Text columns are NOT in must_have — NaN in text cols is filled with '' in
# Stage 9.  Only structured columns that cannot be meaningfully imputed are
# required to be non-null here.
must_have = ['ASA_score', 'sex', 'surg_encounter_type', 'case_service']
before = len(df)
df.dropna(subset=must_have, inplace=True)
print(f"\n  Dropped {before - len(df):,} rows missing in {must_have}")
print(f"  NOTE: age_at_discharge, avg_BMI, anesthetic_type NaN rows KEPT — will be imputed fold-wise in 02. Modeling.py")
print(f"  NOTE: text column NaN rows KEPT — filled with empty string in Stage 9.")
print_missing_summary(df, "after dropping non-imputable")
print_shape_log(df, "dropping non-imputable missing rows")

# =============================================================================
# STAGE 8 — CATEGORICAL CLEANING
# =============================================================================
sep("STAGE 8 — CATEGORICAL CLEANING")

# --- ASA Score ---
print("\n  [ASA Score]")
print_freq_table(df['ASA_score'], "ASA_score before cleanup")
df['ASA_score'] = df['ASA_score'].apply(
    lambda x: int(m.group(1)) if (m := re.match(r'^([1-5])(?:e)?$', str(x).strip().lower())) else np.nan
)
df.dropna(subset=['ASA_score'], inplace=True)
print_freq_table(df['ASA_score'], "ASA_score after cleanup")
print_shape_log(df, "ASA cleanup")

# --- OR Trip Sequence ---
print("\n  [OR Trip Sequence]")
print_freq_table(df['OR_trip_sequence'], "OR_trip_sequence before binarize")
df['OR_trip_sequence'] = (df['OR_trip_sequence'] == 1).astype(int)
print_freq_table(df['OR_trip_sequence'], "OR_trip_sequence after binarize (1=first case)")

# --- Scheduling Flags ---
for col, val in [
    ('first_scheduled_case_of_day_status', 'first scheduled case of day'),
    ('last_scheduled_case_of_day_status',  'last scheduled case of day'),
    ('primary_procedure_status',           'primary procedure')
]:
    print(f"\n  [{col}]")
    print_freq_table(df[col], f"{col} before binarize")
    df[col] = (df[col] == val).astype(int)
    print_freq_table(df[col], f"{col} after binarize")

# --- Sex ---
print("\n  [Sex]")
print_freq_table(df['sex'], "sex before cleanup")
df = handle_rare(df, 'sex', thresh_pct=0.5, drop=True)
df['sex'] = (df['sex'] == 'male').astype(int)
print_freq_table(df['sex'], "sex after binarize (1=male)")
print_shape_log(df, "sex rare drop")

# --- Surgical Encounter ---
print("\n  [Surgical Encounter Type]")
print_freq_table(df['surg_encounter_type'], "surg_encounter_type before mapping")
df['surgery_encounter_inpatient'] = np.where(
    df['surg_encounter_type'].str.lower().isin(['same day admission', 'one day stay']), 0,
    np.where(df['surg_encounter_type'].str.lower() == 'inpatient', 1, np.nan)
)
df.drop(columns=['surg_encounter_type'], inplace=True)
print_freq_table(df['surgery_encounter_inpatient'], "surgery_encounter_inpatient after mapping (0=outpatient, 1=inpatient)")

# --- Surgical Location ---
print("\n  [Surgical Location]")
print_freq_table(df['surgical_location'], "surgical_location before mapping")

def simplify_loc(loc):
    loc = str(loc).strip().lower()
    if loc.startswith('vh or'):   return 'VH_OR'
    if loc.startswith('uh or'):   return 'UH_OR'
    if loc.startswith('vsc or'):  return 'VSC_OR'
    if loc.startswith('zzvh ob'): return 'OB_VH'
    if 'anesthesia' in loc:       return 'Anesthesia'
    if any(x in loc for x in ['pacu', 'pmdu', 'phase', 'recovery']): return 'Recovery'
    if any(x in loc for x in ['tee', 'pain']): return 'Procedure_Room'
    if 'alternate' in loc:        return 'Alternate_OR'
    return 'Other'

df['surgical_location'] = df['surgical_location'].apply(simplify_loc)
before = len(df)
df = handle_rare(df, 'surgical_location', thresh_pct=0.1, drop=True)
print(f"  Dropped {before - len(df):,} rows with rare surgical_location (<0.1%)")
print_freq_table(df['surgical_location'], "surgical_location after mapping + rare drop")
print_shape_log(df, "surgical_location cleanup")

# --- Case Service ---
print("\n  [Case Service]")
print_freq_table(df['case_service'], "case_service before mapping")
svc_map = {
    'orthopedic surgery': 'Orthopedic', 'general surgery': 'General Surgery',
    'obstetrics/gynecology': 'OB/GYN', 'otolaryngology': 'ENT (Otolaryngology)',
    'urology': 'Urology', 'plastic surgery': 'Plastic Surgery',
    'neurosurgery': 'Neurosurgery', 'cardiac surgery': 'Cardiac Surgery',
    'vascular surgery': 'Vascular Surgery', 'thoracic surgery': 'Thoracic Surgery',
    'dental surgery': 'Dental Surgery', 'ophthalmology': 'Ophthalmology',
    'lrcp surg': 'Surgical Oncology', 'cardiology surg': 'Cardiac Surgery',
    'medicine surg': np.nan, 'unknown case service': np.nan, 'anesthesia surg': np.nan
}
df['case_service'] = df['case_service'].str.lower().map(svc_map)
before = len(df)
df.dropna(subset=['case_service'], inplace=True)
print(f"  Dropped {before - len(df):,} rows with unmapped case_service")
print_freq_table(df['case_service'], "case_service after mapping")
print_shape_log(df, "case_service cleanup")

# --- Anesthetic Type ---
print("\n  [Anesthetic Type]")
print_freq_table(df['anesthetic_type'], "anesthetic_type before mapping")
df['anesthetic_type'] = df['anesthetic_type'].str.replace(r'^general/|/general', '', regex=True).str.strip()
anesthesia_map = {
    "general": "General", "general/epidural": "Combined", "general/regional": "Combined",
    "general/spinal": "Combined", "general/spinal opioid": "Combined",
    "general/axillary": "Combined", "general rectal": "General", "general endo": "General",
    "general/home regional": "Combined", "spinal block": "Neuraxial",
    "epidural block": "Neuraxial", "combined spinal/epidural": "Neuraxial",
    "lumbar epidural block": "Neuraxial", "brachial plexus block": "Regional",
    "supraclavicular block": "Regional", "interscalene block": "Regional",
    "infraclavicular block": "Regional", "intercostal brachial": "Regional",
    "sciatic catheter block": "Regional", "paravertebral nerve block": "Regional",
    "transverse abdominus plane block": "Regional", "lumbar plexus block": "Regional",
    "cervical plexus block": "Regional", "ilioinguinal block": "Regional",
    "axillary block": "Regional", "femoral block": "Regional",
    "popliteal block": "Regional", "saphenous knee block": "Regional",
    "saphenous elbow block": "Regional", "suprascapular block": "Regional",
    "home regional": "Regional", "regional": "Regional",
    "regional/home regional": "Regional", "local": "Local",
    "local with standby": "Local", "local/sedation": "Local",
    "local - monitored anesthesia care": "Local", "facial block": "Local",
    "o'brien block": "Local", "peribulbar and retrobulbar block": "Local",
    "ankle block": "Local", "caudal block": "Local", "bier block": "Local",
    "local neurolept": "Sedation", "iv sedation": "Sedation",
    "neurolept": "Sedation", "iv regional": "Sedation",
    "no anesthesia given": np.nan, "system": np.nan, "other": np.nan
}
df['anesthetic_type'] = df['anesthetic_type'].map(anesthesia_map)
print_freq_table(df['anesthetic_type'], "anesthetic_type after mapping (NaN rows KEPT for fold-wise imputation)")
print_shape_log(df, "anesthetic_type mapping")

# =============================================================================
# STAGE 9 — TEXT COLUMN SELECTION
#
# Each text column is kept as its own independent column — no columns are
# merged together.  Columns not selected by the user are dropped.  NaN in
# kept text columns is filled with an empty string so BERT / TF-IDF / count
# encoders receive a valid (though uninformative) input without errors.
# =============================================================================
sep("STAGE 9 — TEXT COLUMN SELECTION")

# Drop columns not selected by the user.
_drop_text = [c for c in ALL_TEXT_COLS if c not in ACTIVE_TEXT_COLS and c in df.columns]
df.drop(columns=_drop_text, inplace=True)
print(f"  Dropped text columns : {_drop_text}")

# Fill NaN in kept text columns with empty string.
for c in ACTIVE_TEXT_COLS:
    if c in df.columns:
        n_nan = df[c].isna().sum()
        df[c] = df[c].fillna('').astype(str)
        if n_nan:
            print(f"  Filled {n_nan:,} NaN → '' in '{c}'")

print(f"\n  Active text columns  : {ACTIVE_TEXT_COLS}")
print(f"\n  Sample text values (first 3 rows):")
for c in ACTIVE_TEXT_COLS:
    for i in range(min(3, len(df))):
        print(f"    Row {i}  {c}: {df[c].iloc[i][:120]}")

print_shape_log(df, "text column selection")

# =============================================================================
# STAGE 10 — ONE-HOT ENCODING  (case_service + surgical_location)
# =============================================================================
sep("STAGE 10 — ONE-HOT ENCODING  (case_service + surgical_location)")
print("  NOTE: anesthetic_type intentionally excluded here.")
print("        It contains NaN and will be imputed fold-wise, then one-hot encoded inside each fold.")

before_cols = df.shape[1]
df = onehot(df, ['case_service', 'surgical_location'], drop_first=False)
new_cols = [c for c in df.columns if c.startswith('case_service__') or c.startswith('surgical_location__')]
print(f"\n  Added {df.shape[1] - before_cols} one-hot columns:")
for c in new_cols:
    print(f"    {c}  (sum={df[c].sum():,})")
print_shape_log(df, "one-hot encoding")

# =============================================================================
# FINAL DATASET SUMMARY
# =============================================================================
sep("FINAL DATASET SUMMARY")

df.reset_index(drop=True, inplace=True)

print(f"\n  Total rows       : {len(df):,}")
print(f"  Total columns    : {df.shape[1]}")
print(f"  Rows removed     : {n_init - len(df):,}  ({(n_init - len(df))/n_init*100:.2f}% of original)")

print(f"\n  Active text columns: {ACTIVE_TEXT_COLS}")

print(f"\n  Column inventory:")
text_c   = [c for c in df.columns if c in ALL_TEXT_COLS]
struct_c = [c for c in df.columns if c not in text_c + [TARGET] and '__' not in c]
onehot_c = [c for c in df.columns if '__' in c]
print(f"    Target          : {TARGET}")
print(f"    Text columns    : {text_c}")
print(f"    Structured cols : {len(struct_c)}  → {struct_c}")
print(f"    One-hot cols    : {len(onehot_c)}")

print_missing_summary(df, "final Clean table")
print("\n  Columns with NaN (expected — fold-wise imputation targets):")
for col in ['age_at_discharge', 'avg_BMI', 'anesthetic_type']:
    n = df[col].isna().sum() if col in df.columns else 'N/A'
    print(f"    {col}: {n} NaN")

print_numeric_summary(df, ['actual_casetime_minutes', 'age_at_discharge', 'avg_BMI', 'ASA_score', 'OR_team_size'] + [c for c in ['scheduled_start_hour', 'or_entry_hour', 'month_of_year', 'day_of_week'] if c in df.columns], "final structured features")

# =============================================================================
# SAVE CLEAN TABLE
# =============================================================================
sep("SAVE")
save_to_db(df, OUTPUT_DB, OUTPUT_TABLE)

# Save text column configuration as a metadata table so downstream scripts
# (02. BERT_Cache.py etc.) can detect which columns are active without user input.
_text_config = pd.DataFrame([{'active_text_cols': ','.join(ACTIVE_TEXT_COLS)}])
with sqlite3.connect(OUTPUT_DB) as conn:
    _text_config.to_sql('text_config', conn, if_exists='replace', index=False)
print(f"  Saved 'text_config' metadata → {OUTPUT_DB}  active={ACTIVE_TEXT_COLS}")

# =============================================================================
# FOLD INDEX GENERATION
# =============================================================================
sep("FOLD INDEX GENERATION")

from sklearn.model_selection import KFold

N_SPLITS     = 5
RANDOM_STATE = 42

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

fold_rows = []
for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
    for idx in train_idx:
        fold_rows.append({'fold': int(fold), 'split': 'train', 'row_index': int(idx)})
    for idx in val_idx:
        fold_rows.append({'fold': int(fold), 'split': 'val', 'row_index': int(idx)})

fold_df = pd.DataFrame(fold_rows)

with sqlite3.connect(OUTPUT_DB) as conn:
    fold_df.to_sql('fold_indices', conn, if_exists='replace', index=False)

print(f"  KFold: n_splits={N_SPLITS}  shuffle=True  random_state={RANDOM_STATE}")
print(f"  Total rows in fold_indices table: {len(fold_df):,}")
print()
print(f"  {'Fold':<6} {'Train rows':>12} {'Val rows':>12}")
print(f"  {'-'*32}")
for fold in range(N_SPLITS):
    n_train = len(fold_df[(fold_df['fold'] == fold) & (fold_df['split'] == 'train')])
    n_val   = len(fold_df[(fold_df['fold'] == fold) & (fold_df['split'] == 'val')])
    print(f"  {fold:<6} {n_train:>12,} {n_val:>12,}")

print(f"\n  Saved table 'fold_indices' → {OUTPUT_DB}")
print(f"\n  Pipeline complete. Log saved to ./results/01_preprocessing.log")

# =============================================================================
# CLOSE LOGGER
# =============================================================================
_tee.close()
"""
================================================================================
Innovative Network-Host Statistical Information Fusion Methodology
================================================================================

This framework implements a novel multi-modal fusion pipeline that integrates
per-second network traffic flow telemetry with host-level process statistics
for robust multi-class intrusion detection. The methodology is grounded on three
foundational pillars:

1. TEMPORAL AGGREGATION: Raw flow records (NetFlow/Argus format) and host
   process metrics (ATOP format) are aggregated into unified per-second
   feature vectors. This temporal alignment ensures that network behavioral
   anomalies and host resource anomalies are synchronized at the same
   granularity.

2. MULTI-DIMENSIONAL FEATURE ENGINEERING:
   - Network Traffic (E/A/B/C/D Taxonomy):
       E) Volumetric & ratio features (packet/byte counts, directional ratios)
       A) Statistical distribution features (mean, max, min, std, p95)
       B) Endpoint diversity features (unique IPs, ports, Shannon entropy)
       C) Protocol-state categorical features (TCP states, protocols)
       D) Flag-level signature features (per-flag occurrence counts)
   - Host Statistics (5-Layer Taxonomy):
       Layer 1) Aggregate resource consumption (CPU, MEM, I/O totals)
       Layer 2) Distribution of continuous metrics (mean, max, std, p95)
       Layer 3) Process diversity & entropy (unique commands, CPU numbers, policies)
       Layer 4) Process state composition (running, sleeping, zombie ratios)
       Layer 5) Anomalous growth & priority indicators (VGROW, RGROW, NICE, PRI)

3. OR-FUSION LABELING STRATEGY: A second-level label is assigned via logical
   disjunction (OR) of per-second attack indicators from both modalities.
   If either the host statistics or the network traffic signals an attack
   within a given second, the unified second-level label is marked as the
   corresponding attack class; otherwise, it is labeled as benign (normal).

This design enables the model to detect attacks that manifest exclusively in
one modality while maintaining high precision through cross-modal validation.
================================================================================
"""

import pandas as pd
import numpy as np
import os
import csv
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

# ==============================================================================
# Global Configuration
# ==============================================================================
STATISTICS_DIR = "statistics/data_process"
TRAFFIC_DIR    = "traffic/data_process"
OUTPUT_DIR     = "merge/data_process"

EPS = 1e-12

# Labeling thresholds: a second is considered attack if at least one record
# in that second exhibits anomalous behavior in either modality.
STAT_ATTACK_MIN_RECORDS  = 1
TRAF_ATTACK_MIN_RECORDS  = 1

FILE_PAIRS = [
    ("dataTheft_label_atop.csv",  "label_traffic_dataTheft.csv",  "dataTheft"),
    ("diskWipe_label_atop.csv",   "label_traffic_diskWipe.csv",   "diskWipe"),
    ("dos_label_atop.csv",        "label_traffic_dos.csv",        "dos"),
    ("mining_label_atop.csv",     "label_traffic_mining.csv",     "mining"),
    ("mirai_label_atop.csv",      "label_traffic_mirai.csv",      "mirai"),
    ("ransomware_label_atop.csv", "label_traffic_ransomware.csv", "ransomware"),
    ("rootkit_label_atop.csv",    "label_traffic_rootkit.csv",    "rootkit"),
]

KNOWN_STATES = ['S0', 'S1', 'S2', 'S3', 'SF', 'REJ',
                'RSTO', 'RSTR', 'RSTOS0', 'RSTRH',
                'SH', 'SHR', 'OTH', 'EST', 'CLO']
KNOWN_PROTOS = ['tcp', 'udp', 'icmp', 'arp', 'igmp', 'ipv6-icmp']
KNOWN_FLAGS  = ['e', 's', 'd', 'f', 'r', 'S', 'A', 'a', 'F', 'R']

NEW_DATA     = os.path.join(OUTPUT_DIR, "merged_ALL_multiclass_new.csv")
TRAIN_OUTPUT = os.path.join(OUTPUT_DIR, "train.csv")
TEST_OUTPUT  = os.path.join(OUTPUT_DIR, "test.csv")
TEST_SIZE    = 0.3
RANDOM_STATE = 42
LABEL_COL    = 'Label'

# Auxiliary columns used only during intermediate processing; excluded from
# the final feature set to prevent information leakage.
AUX_COLS = [
    'stat_n_attack_records',
    'traf_n_attack_records',
    'traf_n_attack_records.1',
    'has_traffic',
    'attack_file',
]

# ==============================================================================
# Utility Functions
# ==============================================================================
def fix_column_names(header):
    if len(header) == 1 and ',' in header[0]:
        return header[0].split(',')
    elif len(header) == 1 and '\t' in header[0]:
        return header[0].split('\t')
    return header

def safe_read_csv(filepath):
    for encoding in ['utf-8-sig', 'utf-8', 'latin1']:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                header = fix_column_names(next(csv.reader(f)))
            sep = '\t' if any('\t' in c for c in header) else ','
            df = pd.read_csv(filepath, sep=sep, encoding=encoding,
                             header=0, names=header, engine='python',
                             on_bad_lines='warn')
            return df
        except Exception:
            continue
    raise ValueError(f"Unable to read file: {filepath}")

def shannon_entropy(values):
    if len(values) == 0:
        return 0.0
    vc = pd.Series(values).value_counts(normalize=True)
    return float(-np.sum(vc.values * np.log2(vc.values + EPS)))

def safe_div_series(a: pd.Series, b: pd.Series) -> pd.Series:
    result = a.copy().astype(float)
    mask = b > 0
    result[mask]  = a[mask] / b[mask]
    result[~mask] = 0.0
    return result

def to_unix_seconds(series):
    s = pd.to_numeric(series, errors='coerce')
    if s.dropna().mean() > 2e10:
        s = s / 1000
    return s

def attack_int(series):
    """Normalize attack labels: attack=1, benign=0. Compatible with both numeric
    and string representations (e.g., 'normal' vs. attack class names)."""
    raw = series.astype(str).str.strip()
    vals = raw.unique().tolist()
    is_num = all(v.replace('.', '', 1).lstrip('-').isdigit()
                 for v in vals if v not in ('nan', ''))
    if is_num:
        return (raw.astype(float).astype(int) != 0).astype(int)
    return (raw.str.lower() != 'normal').astype(int)

# ==============================================================================
# Network Traffic Aggregation (Features + Per-Second Label)
# ==============================================================================
def aggregate_traffic_features(df, attack_type):
    df = df.copy()
    for col in ['TotPkts', 'TotBytes', 'SrcBytes', 'DstBytes',
                'SrcPkts', 'DstPkts', 'Dur', 'Mean', 'StdDev',
                'Sum', 'Min', 'Max', 'Rate', 'SrcRate', 'DstRate', 'Seq']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    for col in ['State', 'Proto', 'Flgs']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df['is_attack'] = attack_int(df['Label'])
    df['time_sec']  = to_unix_seconds(df['StartTime']).astype(np.int64)
    grouped = df.groupby('time_sec', sort=True)
    parts   = []

    # Category E: Volumetric and ratio features
    e = pd.DataFrame(index=grouped.size().index)
    e['flow_count']       = grouped.size()
    e['total_TotPkts']    = grouped['TotPkts'].sum()
    e['total_TotBytes']   = grouped['TotBytes'].sum()
    e['total_SrcBytes']   = grouped['SrcBytes'].sum()
    e['total_DstBytes']   = grouped['DstBytes'].sum()
    e['total_SrcPkts']    = grouped['SrcPkts'].sum()
    e['total_DstPkts']    = grouped['DstPkts'].sum()
    e['src_bytes_ratio']  = safe_div_series(e['total_SrcBytes'], e['total_TotBytes'])
    e['dst_bytes_ratio']  = safe_div_series(e['total_DstBytes'], e['total_TotBytes'])
    e['src_pkts_ratio']   = safe_div_series(e['total_SrcPkts'],  e['total_TotPkts'])
    e['short_flow_ratio'] = (df['Dur'] < 0.1).astype(int).groupby(df['time_sec']).mean()
    e['large_flow_ratio'] = (df['TotBytes'] > 10000).astype(int).groupby(df['time_sec']).mean()
    e['asym_flow_ratio']  = (df['SrcBytes'] > df['DstBytes'] * 5).astype(int).groupby(df['time_sec']).mean()
    parts.append(e)

    # Category A: Statistical distribution features (moments and quantiles)
    a = pd.DataFrame(index=e.index)
    for col in ['Mean', 'StdDev', 'Min', 'Max', 'Sum',
                'Rate', 'SrcRate', 'DstRate', 'Dur', 'Seq']:
        if col not in df.columns:
            continue
        g = grouped[col]
        a[f'{col}_mean'] = g.mean()
        a[f'{col}_max']  = g.max()
        a[f'{col}_min']  = g.min()
        a[f'{col}_std']  = g.std().fillna(0)
        a[f'{col}_p95']  = g.quantile(0.95)
    parts.append(a)

    # Category B: Endpoint diversity and entropy features
    b = pd.DataFrame(index=e.index)
    for col, alias in [('SrcAddr', 'src_ip'), ('DstAddr', 'dst_ip'),
                       ('Sport',   'sport'),  ('Dport',   'dport')]:
        if col not in df.columns:
            continue
        b[f'n_unique_{alias}'] = grouped[col].nunique()
        b[f'entropy_{alias}']  = grouped[col].apply(shannon_entropy)
    parts.append(b)

    # Category C: Categorical state and protocol count features
    c = pd.DataFrame(index=e.index)
    if 'State' in df.columns:
        for s in KNOWN_STATES:
            c[f'state_{s.lower()}_cnt'] = (
                (df['State'].str.upper() == s.upper()).astype(int)
                .groupby(df['time_sec']).sum())
        known = df['State'].str.upper().isin([x.upper() for x in KNOWN_STATES])
        c['state_other_cnt'] = (~known).astype(int).groupby(df['time_sec']).sum()
    if 'Proto' in df.columns:
        for p in KNOWN_PROTOS:
            c[f'proto_{p.replace("-","_")}_cnt'] = (
                (df['Proto'].str.lower() == p).astype(int)
                .groupby(df['time_sec']).sum())
        known_p = df['Proto'].str.lower().isin(KNOWN_PROTOS)
        c['proto_other_cnt'] = (~known_p).astype(int).groupby(df['time_sec']).sum()
    parts.append(c)

    # Category D: TCP flag occurrence features
    d = pd.DataFrame(index=e.index)
    if 'Flgs' in df.columns:
        flgs = df['Flgs'].fillna('').astype(str).str.strip()
        for ch in KNOWN_FLAGS:
            d[f'flg_{ch}_cnt'] = (
                flgs.str.count(re.escape(ch)).groupby(df['time_sec']).sum())
    parts.append(d)

    # Per-second attack flow count determines the traffic-side label
    traf_attack_per_sec = df.groupby('time_sec')['is_attack'].sum()
    traf_label = traf_attack_per_sec.apply(
        lambda n: attack_type if n >= TRAF_ATTACK_MIN_RECORDS else 'normal'
    )

    feat = pd.concat(parts, axis=1).fillna(0)
    feat.index.name = 'time_sec'
    feat = feat.reset_index()
    feat['traf_label']            = feat['time_sec'].map(traf_label)
    feat['traf_n_attack_records'] = feat['time_sec'].map(traf_attack_per_sec).fillna(0).astype(int)
    return feat

# ==============================================================================
# Host Statistics Aggregation (Features + Per-Second Label)
# ==============================================================================
def aggregate_statistics_with_label(df, attack_type):
    df = df.copy()
    for col in ['CPU', 'MEM', 'RDDSK', 'WRDSK', 'WCANCL', 'DSK',
                'MINFLT', 'MAJFLT', 'VSTEXT', 'VSIZE', 'RSIZE',
                'VGROW', 'RGROW', 'TRUN', 'TSLPI', 'TSLPU',
                'NICE', 'PRI', 'RTPR', 'CPUNR', 'EXC']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    for col in ['CMD', 'S', 'ST', 'POLI']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df['is_attack'] = attack_int(df['Label'])
    df['time_sec']  = to_unix_seconds(df['TIMESTAMP']).astype(np.int64)
    grouped = df.groupby('time_sec', sort=True)
    parts   = []

    # Layer 1: Aggregate resource consumption totals
    g1 = pd.DataFrame(index=grouped.size().index)
    g1['proc_count']   = grouped.size()
    g1['total_CPU']    = grouped['CPU'].sum()
    g1['total_MEM']    = grouped['MEM'].sum()
    g1['total_RDDSK']  = grouped['RDDSK'].sum()
    g1['total_WRDSK']  = grouped['WRDSK'].sum()
    g1['total_DSK']    = grouped['DSK'].sum()
    g1['total_VGROW']  = grouped['VGROW'].sum()
    g1['total_RGROW']  = grouped['RGROW'].sum()
    g1['total_MAJFLT'] = grouped['MAJFLT'].sum()
    g1['total_MINFLT'] = grouped['MINFLT'].sum()
    g1['total_TRUN']   = grouped['TRUN'].sum()
    g1['total_TSLPI']  = grouped['TSLPI'].sum()
    g1['total_TSLPU']  = grouped['TSLPU'].sum()
    parts.append(g1)

    # Layer 2: Statistical distribution of continuous process metrics
    g2 = pd.DataFrame(index=g1.index)
    for col in ['CPU', 'MEM', 'RDDSK', 'WRDSK', 'VSIZE', 'RSIZE', 'EXC']:
        if col not in df.columns:
            continue
        g = grouped[col]
        g2[f'{col}_mean'] = g.mean()
        g2[f'{col}_max']  = g.max()
        g2[f'{col}_std']  = g.std().fillna(0)
        g2[f'{col}_p95']  = g.quantile(0.95)
    parts.append(g2)

    # Layer 3: Process diversity and entropy features
    g3 = pd.DataFrame(index=g1.index)
    if 'CMD' in df.columns:
        g3['n_unique_CMD'] = grouped['CMD'].nunique()
        g3['entropy_CMD']  = grouped['CMD'].apply(shannon_entropy)
    if 'CPUNR' in df.columns:
        g3['n_unique_CPUNR'] = grouped['CPUNR'].nunique()
    if 'POLI' in df.columns:
        g3['n_unique_POLI']  = grouped['POLI'].nunique()
    parts.append(g3)

    # Layer 4: Process state composition and anomaly ratios
    g4 = pd.DataFrame(index=g1.index)
    if 'S' in df.columns:
        for state, name in [('R', 'n_proc_running'), ('S', 'n_proc_sleeping'),
                            ('D', 'n_proc_uninterrupt'), ('Z', 'n_proc_zombie'),
                            ('T', 'n_proc_stopped')]:
            g4[name] = (df['S'] == state).astype(int).groupby(df['time_sec']).sum()
        g4['zombie_ratio']      = safe_div_series(g4['n_proc_zombie'],      g1['proc_count'])
        g4['running_ratio']     = safe_div_series(g4['n_proc_running'],     g1['proc_count'])
        g4['uninterrupt_ratio'] = safe_div_series(g4['n_proc_uninterrupt'], g1['proc_count'])
    parts.append(g4)

    # Layer 5: Growth and priority anomaly indicators
    g5 = pd.DataFrame(index=g1.index)
    if 'VGROW' in df.columns:
        g5['VGROW_max']   = grouped['VGROW'].max()
        g5['RGROW_max']   = grouped['RGROW'].max()
        g5['n_vgrow_pos'] = (df['VGROW'] > 0).astype(int).groupby(df['time_sec']).sum()
        g5['n_rgrow_pos'] = (df['RGROW'] > 0).astype(int).groupby(df['time_sec']).sum()
    if 'NICE' in df.columns:
        g5['NICE_mean'] = grouped['NICE'].mean()
        g5['NICE_min']  = grouped['NICE'].min()
    if 'PRI' in df.columns:
        g5['PRI_mean']  = grouped['PRI'].mean()
        g5['PRI_max']   = grouped['PRI'].max()
    if 'RTPR' in df.columns:
        g5['n_rt_proc'] = (df['RTPR'] > 0).astype(int).groupby(df['time_sec']).sum()
    parts.append(g5)

    # Per-second attack process count determines the host-side label
    stat_attack_per_sec = df.groupby('time_sec')['is_attack'].sum()
    stat_label = stat_attack_per_sec.apply(
        lambda n: attack_type if n >= STAT_ATTACK_MIN_RECORDS else 'normal'
    )

    feat = pd.concat(parts, axis=1).fillna(0)
    feat.index.name = 'time_sec'
    feat = feat.reset_index()
    feat['stat_label']            = feat['time_sec'].map(stat_label)
    feat['stat_n_attack_records'] = feat['time_sec'].map(stat_attack_per_sec).fillna(0).astype(int)
    return feat

# ==============================================================================
# Fusion Labeling Strategy
# ==============================================================================
def or_fuse_label(stat_label, traf_label, attack_type):
    s_atk = str(stat_label).strip().lower() != 'normal'
    if pd.isna(traf_label):
        t_atk = False
    else:
        t_atk = str(traf_label).strip().lower() != 'normal'
    return attack_type if (s_atk or t_atk) else 'normal'

# ==============================================================================
# Single-Pair Fusion Pipeline
# ==============================================================================
def fuse_one_pair(stat_path, traf_path, attack_type):
    stat_df = safe_read_csv(stat_path)
    traf_df = safe_read_csv(traf_path)

    stat_agg = aggregate_statistics_with_label(stat_df, attack_type)
    traf_agg = aggregate_traffic_features(traf_df, attack_type)

    # Prefix columns to maintain modality traceability
    stat_keep = {'time_sec', 'stat_label', 'stat_n_attack_records'}
    traf_keep = {'time_sec', 'traf_label', 'traf_n_attack_records'}
    stat_feat = stat_agg.rename(columns={c: f'stat_{c}' for c in stat_agg.columns if c not in stat_keep})
    traf_feat = traf_agg.rename(columns={c: f'traf_{c}' for c in traf_agg.columns if c not in traf_keep})

    # Left join on host statistics timeline to retain all benign seconds
    merged = pd.merge(stat_feat, traf_feat, on='time_sec', how='left')

    # Impute missing traffic features for seconds without network activity
    traf_num_cols = [c for c in merged.columns
                     if c.startswith('traf_') and c not in ('traf_label', 'traf_n_attack_records')]
    merged[traf_num_cols] = merged[traf_num_cols].fillna(0)
    merged['traf_n_attack_records'] = merged['traf_n_attack_records'].fillna(0).astype(int)

    # Traffic presence indicator
    merged['has_traffic'] = (merged.get('traf_flow_count', pd.Series(0, index=merged.index)) > 0).astype(int)

    # OR-fusion of per-second labels
    merged['Label'] = merged.apply(
        lambda r: or_fuse_label(r['stat_label'], r['traf_label'], attack_type),
        axis=1
    )

    # Drop intermediate label columns and enforce canonical column ordering
    merged = merged.drop(columns=['stat_label', 'traf_label'])
    stat_cols = [c for c in merged.columns
                 if c.startswith('stat_') and c != 'stat_n_attack_records']
    traf_cols = [c for c in merged.columns
                 if c.startswith('traf_')]
    order = (['time_sec'] + stat_cols + traf_cols +
             ['has_traffic', 'stat_n_attack_records', 'traf_n_attack_records', 'Label'])
    merged = merged[order].sort_values('time_sec').reset_index(drop=True)
    return merged

# ==============================================================================
# Main Execution Pipeline
# ==============================================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_merged = []
    success, failed = 0, []

    for stat_fname, traf_fname, attack_type in FILE_PAIRS:
        stat_path = os.path.join(STATISTICS_DIR, stat_fname)
        traf_path = os.path.join(TRAFFIC_DIR,    traf_fname)

        if not os.path.exists(stat_path) or not os.path.exists(traf_path):
            failed.append(attack_type)
            continue
        try:
            m = fuse_one_pair(stat_path, traf_path, attack_type)
            m['attack_file'] = attack_type
            all_merged.append(m)
            success += 1
        except Exception:
            failed.append(attack_type)

    if not all_merged:
        raise RuntimeError("No attack types were successfully fused. Pipeline aborted.")

    combined = pd.concat(all_merged, ignore_index=True).fillna(0)

    df = combined.copy()

    actual_aux = [c for c in AUX_COLS if c in df.columns]
    df = df.drop(columns=actual_aux)

    if 'time_sec' in df.columns:
        df = df.drop(columns=['time_sec'])

    # Canonical column ordering: stat features -> traf features -> others -> Label
    stat_cols  = [c for c in df.columns if c.startswith('stat_')]
    traf_cols  = [c for c in df.columns if c.startswith('traf_')]
    other_cols = [c for c in df.columns
                  if c != LABEL_COL
                  and not c.startswith('stat_')
                  and not c.startswith('traf_')]

    new_col_order = stat_cols + traf_cols + other_cols + [LABEL_COL]
    df = df[new_col_order]

    df.to_csv(NEW_DATA, index=False)

    feature_cols = [c for c in df.columns if c != LABEL_COL]

    X = df[feature_cols]
    y = df[LABEL_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    train_perm = np.random.RandomState(RANDOM_STATE).permutation(len(X_train))
    test_perm  = np.random.RandomState(RANDOM_STATE + 1).permutation(len(X_test))

    train_df = pd.concat([
        X_train.iloc[train_perm].reset_index(drop=True),
        y_train.iloc[train_perm].reset_index(drop=True)
    ], axis=1)

    test_df = pd.concat([
        X_test.iloc[test_perm].reset_index(drop=True),
        y_test.iloc[test_perm].reset_index(drop=True)
    ], axis=1)

    train_df.to_csv(TRAIN_OUTPUT, index=False)
    test_df.to_csv(TEST_OUTPUT,   index=False)

    print("Files saved successfully:")
    print(f"  {NEW_DATA}")
    print(f"  {TRAIN_OUTPUT}")
    print(f"  {TEST_OUTPUT}")
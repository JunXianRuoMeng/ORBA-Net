"""
Innovative Network-Physical Data Fusion Framework
=================================================

This module implements a novel second-level aggregation and fusion pipeline
for heterogeneous cyber-physical data streams. The core methodology addresses
the fundamental challenge of temporal misalignment between high-frequency
network packet traces and low-frequency physical sensor readings in industrial
control systems.


1. Multi-Dimensional Network Traffic Aggregation:
   Raw network packets are aggregated into per-second feature vectors
   through four complementary perspectives:
   - Temporal dynamics (packet rates, inter-arrival statistics, burstiness)
   - Payload statistics (size distributions, quantiles, skewness, kurtosis)
   - Communication diversity (entropy of endpoints, port diversity)
   - Protocol semantics (Modbus function code frequencies, read/write ratios)
   - TCP behavioral patterns (flag bit decomposition, SYN/ACK ratios)

2. Cross-Modal Label Fusion:
   Labels from both modalities are fused via a priority-based majority-voting
   mechanism that prioritizes physical-layer anomaly indicators, reflecting
   the physical-grounded nature of industrial attacks.
"""
import pandas as pd
import numpy as np
import io
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

EPS = 1e-12

KNOWN_MODBUS_FN = [
    'Read Coils', 'Read Discrete Inputs', 'Read Holding Registers',
    'Read Input Registers', 'Write Single Coil', 'Write Single Register',
    'Write Multiple Coils', 'Write Multiple Registers',
    'Read Exception Status', 'Diagnostics', 'Read FIFO Queue',
    'Encapsulated Interface Transport',
    'Read Coils Request', 'Read Coils Response',
    'Read Discrete Inputs Request', 'Read Discrete Inputs Response',
    'Read Holding Registers Request', 'Read Holding Registers Response',
    'Read Input Registers Request', 'Read Input Registers Response',
    'Write Single Coil Request', 'Write Single Coil Response',
    'Write Single Register Request', 'Write Single Register Response',
    'Write Multiple Coils Request', 'Write Multiple Coils Response',
    'Write Multiple Registers Request', 'Write Multiple Registers Response',
]


def detect_encoding(filepath):
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb18030', 'latin1', 'cp1252']
    for enc in encodings:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                f.read(4096)
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return 'latin1'


def safe_read_csv(filepath, **kwargs):
    """
    Robust CSV loader with automatic encoding detection and null-byte sanitization.
    Handles UTF-16/32 BOMs, mixed encodings, and embedded NUL characters that
    commonly occur in industrial network capture exports.
    """
    with open(filepath, 'rb') as f:
        raw = f.read()
    meta = {'encoding': None, 'nul_bytes_removed': 0, 'original_size': len(raw)}
    if raw.startswith(b'\xff\xfe\x00\x00') or raw.startswith(b'\x00\x00\xfe\xff'):
        text = raw.decode('utf-32')
        meta['encoding'] = 'utf-32'
    elif raw.startswith(b'\xff\xfe') or raw.startswith(b'\xfe\xff'):
        text = raw.decode('utf-16')
        meta['encoding'] = 'utf-16'
    elif raw.startswith(b'\xef\xbb\xbf'):
        text = raw[3:].decode('utf-8', errors='replace')
        meta['encoding'] = 'utf-8-sig'
    else:
        nul_ratio = raw.count(b'\x00') / max(len(raw), 1)
        if nul_ratio > 0.2:
            for enc in ['utf-16-le', 'utf-16-be']:
                try:
                    text = raw.decode(enc)
                    meta['encoding'] = enc + ' (no BOM, heuristic)'
                    break
                except UnicodeDecodeError:
                    continue
            else:
                text = raw.decode('latin1', errors='replace')
                meta['encoding'] = 'latin1 (fallback)'
        else:
            for enc in ['utf-8', 'gbk', 'gb18030', 'cp1252', 'latin1']:
                try:
                    text = raw.decode(enc)
                    meta['encoding'] = enc
                    break
                except UnicodeDecodeError:
                    continue
            else:
                text = raw.decode('latin1', errors='replace')
                meta['encoding'] = 'latin1 (fallback)'
    if '\x00' in text:
        meta['nul_bytes_removed'] = text.count('\x00')
        text = text.replace('\x00', '')
    df = pd.read_csv(io.StringIO(text), **kwargs)
    return df, meta


def parse_net_time(s):
    if pd.isna(s):
        return pd.NaT
    s = str(s).strip()
    formats = [
        '%Y-%m-%d %H:%M:%S.%f', '%Y/%m/%d %H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S',
        '%d/%m/%Y %H:%M:%S.%f', '%d/%m/%Y %H:%M:%S',
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return pd.NaT


def parse_phy_time(s):
    if pd.isna(s):
        return pd.NaT
    s = str(s).strip()
    if '.' in s:
        s = s.split('.')[0]
    formats = [
        '%d/%m/%Y %H:%M:%S', '%d-%m-%Y %H:%M:%S',
        '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S',
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return pd.NaT


def shannon_entropy(values):
    if len(values) == 0:
        return 0.0
    vc = pd.Series(values).value_counts(normalize=True)
    return float(-np.sum(vc.values * np.log2(vc.values + EPS)))


def safe_div(a, b):
    return np.where(b > 0, a / np.where(b > 0, b, 1), 0.0)


def load_and_normalize(net_path, phy_path, log):
    """
    Loads network and physical datasets with robust encoding handling,
    strips whitespace from string fields, and normalizes timestamps to
    second-level granularity for subsequent temporal alignment.
    """
    net_df, net_meta = safe_read_csv(
        net_path, sep=None, engine='python', skipinitialspace=True,
        dtype={'flags': str},
    )
    net_df.columns = net_df.columns.str.strip()
    for c in net_df.select_dtypes(include='object').columns:
        net_df[c] = net_df[c].astype(str).str.strip()
    log.append(f"[Network] Encoding: {net_meta['encoding']}, "
               f"NUL bytes removed: {net_meta['nul_bytes_removed']}, "
               f"Original size: {net_meta['original_size']} bytes")
    log.append(f"[Network] Raw rows: {len(net_df)}")
    log.append(f"[Network] Columns: {list(net_df.columns)}")
    net_df['time_full'] = pd.to_datetime(
        net_df['Time'].apply(parse_net_time), errors='coerce'
    )
    n_bad = net_df['time_full'].isna().sum()
    if n_bad > 0:
        log.append(f"[Network] Time parse failures: {n_bad} rows (dropped)")
        net_df = net_df.dropna(subset=['time_full']).reset_index(drop=True)
    net_df['time_sec'] = net_df['time_full'].dt.floor('s')
    log.append(f"[Network] Valid rows: {len(net_df)}")
    log.append(f"[Network] Time range: {net_df['time_sec'].min()} ~ {net_df['time_sec'].max()}")

    phy_df, phy_meta = safe_read_csv(
        phy_path, sep=None, engine='python', skipinitialspace=True,
    )
    phy_df.columns = phy_df.columns.str.strip()
    for c in phy_df.select_dtypes(include='object').columns:
        phy_df[c] = phy_df[c].astype(str).str.strip()
    log.append(f"[Physical] Encoding: {phy_meta['encoding']}, "
               f"NUL bytes removed: {phy_meta['nul_bytes_removed']}, "
               f"Original size: {phy_meta['original_size']} bytes")
    log.append(f"[Physical] Raw rows: {len(phy_df)}")
    log.append(f"[Physical] Columns: {len(phy_df.columns)}")
    phy_df['time_sec'] = pd.to_datetime(
        phy_df['Time'].apply(parse_phy_time), errors='coerce'
    )
    n_bad = phy_df['time_sec'].isna().sum()
    if n_bad > 0:
        log.append(f"[Physical] Time parse failures: {n_bad} rows (dropped)")
        phy_df = phy_df.dropna(subset=['time_sec']).reset_index(drop=True)
    log.append(f"[Physical] Valid rows: {len(phy_df)}")
    log.append(f"[Physical] Time range: {phy_df['time_sec'].min()} ~ {phy_df['time_sec'].max()}")
    return net_df, phy_df


def restrict_to_common_window(net_df, phy_df, log):
    """
    Computes the temporal intersection between network and physical datasets.
    Only records within the common second-level window are retained, ensuring
    strict temporal alignment for downstream fusion.
    """
    net_secs = set(net_df['time_sec'].unique())
    phy_secs = set(phy_df['time_sec'].unique())
    common_secs = net_secs & phy_secs
    only_net = net_secs - phy_secs
    only_phy = phy_secs - net_secs
    log.append(f"Network-exclusive seconds: {len(only_net)} (discarded)")
    log.append(f"Physical-exclusive seconds: {len(only_phy)} (discarded)")
    log.append(f"Common seconds: {len(common_secs)}")
    if only_net:
        sample = sorted(only_net)[:5]
        log.append(f"  Network-exclusive samples: {[str(s) for s in sample]}")
    if only_phy:
        sample = sorted(only_phy)[:5]
        log.append(f"  Physical-exclusive samples: {[str(s) for s in sample]}")
    net_df = net_df[net_df['time_sec'].isin(common_secs)].reset_index(drop=True)
    phy_df = phy_df[phy_df['time_sec'].isin(common_secs)].reset_index(drop=True)
    log.append(f"After restriction: {len(net_df)} network packets")
    log.append(f"After restriction: {len(phy_df)} physical records")
    return net_df, phy_df, sorted(common_secs)


def aggregate_network(net_df, log):
    """
    Aggregates raw network packet traces into per-second feature vectors through
    a multi-perspective feature engineering pipeline:

    E-group: Temporal dynamics -- packet rates, inter-arrival time statistics,
             burstiness index, request/response ratios.
    A-group: Payload statistics -- size distribution moments (mean, std,
             quantiles, skewness, kurtosis).
    B-group: Communication diversity -- entropy and uniqueness of endpoints
             (IP, MAC, ports).
    C-group: Protocol semantics -- Modbus function code frequencies,
             read/write/diagnostic ratios.
    D-group: TCP behavioral patterns -- flag bit decomposition, SYN/ACK ratios.
    """
    for col in ['size', 'sport', 'dport', 'n_pkt_src', 'n_pkt_dst']:
        if col in net_df.columns:
            net_df[col] = pd.to_numeric(net_df[col], errors='coerce')
    grouped = net_df.groupby('time_sec', sort=True)
    parts = []

    # E-group: Temporal dynamics and rate features
    e_feat = pd.DataFrame(index=grouped.size().index)
    e_feat['pkt_count'] = grouped.size()
    e_feat['bytes_per_sec'] = grouped['size'].sum()
    def iat_stats(group):
        if len(group) < 2:
            return pd.Series({'iat_mean': 0.0, 'iat_std': 0.0,
                              'iat_min': 0.0, 'iat_max': 0.0, 'burstiness': 0.0})
        times = group['time_full'].sort_values()
        iats = times.diff().dt.total_seconds().dropna()
        m, s = iats.mean(), iats.std()
        burst = (s - m) / (s + m) if (s + m) > 0 else 0.0
        return pd.Series({
            'iat_mean': m, 'iat_std': s if not np.isnan(s) else 0.0,
            'iat_min': iats.min(), 'iat_max': iats.max(),
            'burstiness': burst if not np.isnan(burst) else 0.0
        })
    iat_df = grouped.apply(iat_stats)
    e_feat = e_feat.join(iat_df)
    e_feat['pkt_max_per_src_ip'] = net_df.groupby(['time_sec', 'ip_s']).size().groupby('time_sec').max()
    e_feat['pkt_max_per_dst_ip'] = net_df.groupby(['time_sec', 'ip_d']).size().groupby('time_sec').max()
    is_request = (net_df['dport'] == 502).astype(int)
    is_response = (net_df['sport'] == 502).astype(int)
    e_feat['n_request'] = is_request.groupby(net_df['time_sec']).sum()
    e_feat['n_response'] = is_response.groupby(net_df['time_sec']).sum()
    e_feat['request_response_ratio'] = safe_div(e_feat['n_request'].values, e_feat['n_response'].values)
    parts.append(e_feat)

    # A-group: Payload size distribution statistics
    a_feat = pd.DataFrame(index=e_feat.index)
    sizes = grouped['size']
    a_feat['size_mean'] = sizes.mean()
    a_feat['size_std'] = sizes.std().fillna(0)
    a_feat['size_min'] = sizes.min()
    a_feat['size_max'] = sizes.max()
    a_feat['size_p25'] = sizes.quantile(0.25)
    a_feat['size_p50'] = sizes.quantile(0.50)
    a_feat['size_p75'] = sizes.quantile(0.75)
    a_feat['size_p95'] = sizes.quantile(0.95)
    a_feat['size_skew'] = sizes.skew().fillna(0)
    a_feat['size_kurt'] = sizes.apply(lambda x: x.kurt() if len(x) > 3 else 0).fillna(0)
    a_feat['size_sum'] = sizes.sum()
    parts.append(a_feat)

    # B-group: Endpoint communication diversity via entropy
    b_feat = pd.DataFrame(index=e_feat.index)
    for col, alias in [('ip_s', 'src_ip'), ('ip_d', 'dst_ip'),
                       ('mac_s', 'src_mac'), ('mac_d', 'dst_mac'),
                       ('sport', 'sport'), ('dport', 'dport')]:
        if col not in net_df.columns:
            continue
        b_feat[f'n_unique_{alias}'] = grouped[col].nunique()
        b_feat[f'entropy_{alias}'] = grouped[col].apply(shannon_entropy)
    parts.append(b_feat)

    # C-group: Modbus function code semantic analysis
    c_feat = pd.DataFrame(index=e_feat.index, dtype=float)
    if 'modbus_fn' in net_df.columns:
        fn_norm = net_df['modbus_fn'].fillna('N/A').astype(str).str.strip()
        for fn in KNOWN_MODBUS_FN:
            col = f'fn_{fn.lower().replace(" ", "_")}_count'
            c_feat[col] = (fn_norm == fn).groupby(net_df['time_sec']).sum()
        is_known = fn_norm.isin(KNOWN_MODBUS_FN) | (fn_norm == 'N/A')
        c_feat['fn_other_count'] = (~is_known).astype(int).groupby(net_df['time_sec']).sum()
        is_write = fn_norm.str.contains('Write', case=False, na=False).astype(int)
        is_read = fn_norm.str.contains('Read', case=False, na=False).astype(int)
        is_diagnostic = fn_norm.str.contains('Diagnostic|Exception', case=False, na=False, regex=True).astype(int)
        c_feat['write_count'] = is_write.groupby(net_df['time_sec']).sum()
        c_feat['read_count'] = is_read.groupby(net_df['time_sec']).sum()
        c_feat['diagnostic_count'] = is_diagnostic.groupby(net_df['time_sec']).sum()
        c_feat['write_ratio'] = safe_div(c_feat['write_count'].values, e_feat['pkt_count'].values)
        c_feat['read_ratio'] = safe_div(c_feat['read_count'].values, e_feat['pkt_count'].values)
    parts.append(c_feat)

    # D-group: TCP flag bit decomposition
    d_feat = pd.DataFrame(index=e_feat.index, dtype=float)
    if 'flags' in net_df.columns:
        flag_str = net_df['flags'].fillna('0').astype(str).str.strip().str.strip("'\"")
        flag_str = flag_str.str.replace(r'\.0+$', '', regex=True)
        flag_str = flag_str.str.replace(r'[^01]', '', regex=True)
        flag_str = flag_str.replace('', '0')
        max_len = int(flag_str.str.len().max())
        flag_padded = flag_str.str.zfill(max_len)
        if max_len == 5:
            bit_names = ['n_ack', 'n_psh', 'n_rst', 'n_syn', 'n_fin']
        elif max_len == 6:
            bit_names = ['n_urg', 'n_ack', 'n_psh', 'n_rst', 'n_syn', 'n_fin']
        elif max_len == 8:
            bit_names = ['n_cwr', 'n_ece', 'n_urg', 'n_ack',
                         'n_psh', 'n_rst', 'n_syn', 'n_fin']
        else:
            bit_names = [f'n_flag_bit_{i}' for i in range(max_len)]
        log.append(f"      Flags bit-length: {max_len}, Mapping: {bit_names}")
        for i, name in enumerate(bit_names):
            bits = pd.to_numeric(flag_padded.str[i], errors='coerce').fillna(0)
            d_feat[name] = bits.groupby(net_df['time_sec']).sum()
        if 'n_syn' in d_feat and 'n_ack' in d_feat:
            d_feat['syn_to_ack_ratio'] = safe_div(d_feat['n_syn'].values, d_feat['n_ack'].values)
    parts.append(d_feat)

    feat = pd.concat(parts, axis=1).fillna(0)
    feat.index.name = 'time_sec'
    feat = feat.reset_index()
    log.append(f"  Network aggregation complete, feature dimension: {feat.shape[1] - 1} (excluding time_sec)")
    log.append(f"  Total seconds: {len(feat)}")
    return feat


def normalize_physical(phy_df, log):
    """
    Normalizes physical sensor and actuator states. Boolean columns (pumps,
    valves) are mapped to binary integers; numerical columns (tank levels,
    flow sensors) are coerced to float with NaN imputation.
    """
    phy_df = phy_df.copy()
    bool_cols = [c for c in phy_df.columns if c.startswith('Pump_') or c.startswith('Valv_')]
    for col in bool_cols:
        phy_df[col] = phy_df[col].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0})
        phy_df[col] = phy_df[col].fillna(0).astype(int)
    log.append(f"  Boolean columns converted: {len(bool_cols)} (Pump/Valv)")
    num_cols = [c for c in phy_df.columns if c.startswith('Tank_') or c.startswith('Flow_sensor_')]
    for col in num_cols:
        phy_df[col] = pd.to_numeric(phy_df[col], errors='coerce').fillna(0)
    return phy_df


def fuse_labels(net_df, phy_df, log):
    """
    Fuses labels from both modalities via a priority-based majority vote.
    Physical-layer anomaly labels take precedence over network-layer labels,
    reflecting the physical-grounded nature of industrial cyber-attacks.
    """
    phy_label_col = 'Label' if 'Label' in phy_df.columns else 'label'
    phy_lbl = phy_df[['time_sec', phy_label_col]].copy()
    phy_lbl.columns = ['time_sec', 'phy_label']
    phy_lbl['phy_label'] = phy_lbl['phy_label'].astype(str).str.strip()

    net_label_col = 'label' if 'label' in net_df.columns else 'Label'
    net_lbl_raw = net_df[['time_sec', net_label_col]].copy()
    net_lbl_raw.columns = ['time_sec', 'net_label']
    net_lbl_raw['net_label'] = net_lbl_raw['net_label'].astype(str).str.strip()

    def majority_attack(labels):
        non_normal = labels[labels.str.lower() != 'normal']
        if len(non_normal) == 0:
            return 'normal'
        return non_normal.value_counts().idxmax()

    net_lbl = net_lbl_raw.groupby('time_sec')['net_label'].apply(majority_attack).reset_index()

    merged = pd.merge(phy_lbl, net_lbl, on='time_sec', how='outer')
    merged['phy_label'] = merged['phy_label'].fillna('normal')
    merged['net_label'] = merged['net_label'].fillna('normal')

    def fuse(row):
        if row['phy_label'].lower() != 'normal':
            return row['phy_label']
        if row['net_label'].lower() != 'normal':
            return row['net_label']
        return 'normal'

    merged['Label'] = merged.apply(fuse, axis=1)
    log.append(f"  Physical label distribution: {dict(merged['phy_label'].value_counts())}")
    log.append(f"  Network label distribution (per-second majority): {dict(merged['net_label'].value_counts())}")
    log.append(f"  Fused label distribution: {dict(merged['Label'].value_counts())}")
    return merged[['time_sec', 'phy_label', 'net_label', 'Label']]


def merge_all(phy_df, net_feat, label_df, log):
    """
    Concatenates normalized physical features with aggregated network features
    and fused labels to produce the final multimodal dataset.
    """
    drop_time_cols = ['Time']
    phy_clean = phy_df.drop(columns=[c for c in drop_time_cols if c in phy_df.columns])
    binary_label_cols = [c for c in phy_clean.columns
                         if c.lower().replace(' ', '') in ('label_n', 'lable_n')]
    if binary_label_cols:
        phy_clean = phy_clean.drop(columns=binary_label_cols)
        log.append(f"  Dropped binary label columns: {binary_label_cols}")
    multi_label_cols = [c for c in phy_clean.columns
                        if c.lower() in ('label', 'lable')]
    if multi_label_cols:
        phy_clean = phy_clean.drop(columns=multi_label_cols)
    fused = pd.merge(phy_clean, net_feat, on='time_sec', how='inner')
    fused = pd.merge(fused, label_df[['time_sec', 'Label']], on='time_sec', how='inner')
    fused = fused.rename(columns={'time_sec': 'Time'})
    cols = ['Time'] + [c for c in fused.columns if c not in ['Time', 'Label']] + ['Label']
    fused = fused[cols]
    fused = fused.sort_values('Time').reset_index(drop=True)
    log.append(f"  Final dataset shape: {fused.shape}")
    return fused


def fuse_one_pair(net_path, phy_path):
    """
    End-to-end fusion pipeline for a single network-physical file pair.
    """
    log = []
    net_df, phy_df = load_and_normalize(net_path, phy_path, log)
    net_df, phy_df, common_secs = restrict_to_common_window(net_df, phy_df, log)
    if len(common_secs) == 0:
        raise ValueError(f"No common time window for {net_path} and {phy_path}")
    net_feat = aggregate_network(net_df, log)
    phy_df = normalize_physical(phy_df, log)
    label_df = fuse_labels(net_df, phy_df, log)
    fused = merge_all(phy_df, net_feat, label_df, log)
    return fused


def fix_label_nomal(input_path, output_path):
    """
    Corrects the 'nomal' -> 'normal' typo that occurs in certain dataset exports.
    """
    df = pd.read_csv(input_path)
    label_col = df.columns[-1]
    df[label_col] = df[label_col].replace('nomal', 'normal')
    df.to_csv(output_path, index=False)
    print(f"Label corrected: {input_path} -> {output_path} ('nomal' -> 'normal')")


if __name__ == '__main__':
    net_dir = Path('Network dataset')
    phy_dir = Path('Physical dataset')

    pairs = [
        (net_dir / 'attack_1.csv', phy_dir / 'phy_att_1.csv'),
        (net_dir / 'attack_2.csv', phy_dir / 'phy_att_2.csv'),
        (net_dir / 'attack_3.csv', phy_dir / 'phy_att_3.csv'),
        (net_dir / 'attack_4.csv', phy_dir / 'phy_att_4.csv'),
    ]

    dfs = []
    for net_f, phy_f in pairs:
        fused_df = fuse_one_pair(str(net_f), str(phy_f))
        dfs.append(fused_df)

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv('combined_dataset.csv', index=False, encoding='utf-8')

    train_df, test_df = train_test_split(
        combined,
        test_size=0.3,
        random_state=42,
        stratify=combined['Label']
    )
    train_df.to_csv('train1.csv', index=False, encoding='utf-8')
    test_df.to_csv('test1.csv', index=False, encoding='utf-8')

    fix_label_nomal('train1.csv', 'train.csv')
    fix_label_nomal('test1.csv', 'test.csv')

    print('combined_dataset.csv, train.csv, test.csv generated successfully.')
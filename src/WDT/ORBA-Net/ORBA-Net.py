import os
import math
import time
import pickle
import random
import unicodedata

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset, Subset

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

torch.backends.cudnn.benchmark = True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    train_df = train_df.iloc[:, 1:]
    test_df  = test_df.iloc[:, 1:]

    network_train  = train_df.iloc[:, 40:118].values
    physical_train = train_df.iloc[:, 0:40].values
    labels_train   = train_df['Label'].values

    network_test   = test_df.iloc[:, 40:118].values
    physical_test  = test_df.iloc[:, 0:40].values
    labels_test    = test_df['Label'].values

    imputer_net = SimpleImputer(strategy='mean')
    imputer_phy = SimpleImputer(strategy='mean')
    network_train  = imputer_net.fit_transform(network_train)
    network_test   = imputer_net.transform(network_test)
    physical_train = imputer_phy.fit_transform(physical_train)
    physical_test  = imputer_phy.transform(physical_test)

    scaler_net = StandardScaler()
    scaler_phy = StandardScaler()
    network_train  = scaler_net.fit_transform(network_train)
    network_test   = scaler_net.transform(network_test)
    physical_train = scaler_phy.fit_transform(physical_train)
    physical_test  = scaler_phy.transform(physical_test)

    label_encoder = LabelEncoder()
    labels_train  = label_encoder.fit_transform(labels_train)
    labels_test   = label_encoder.transform(labels_test)

    return {
        'network_train':  torch.tensor(network_train,  dtype=torch.float32),
        'physical_train': torch.tensor(physical_train, dtype=torch.float32),
        'labels_train':   torch.tensor(labels_train,   dtype=torch.long),
        'network_test':   torch.tensor(network_test,   dtype=torch.float32),
        'physical_test':  torch.tensor(physical_test,  dtype=torch.float32),
        'labels_test':    torch.tensor(labels_test,    dtype=torch.long),
        'num_classes':    len(label_encoder.classes_),
        'class_names':    [str(c) for c in label_encoder.classes_],
    }


class FeatureTokenizer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_tokens, dropout=0.2):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim  = embed_dim
        self.proj = nn.Linear(input_dim, num_tokens * embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B = x.size(0)
        x = self.proj(x).view(B, self.num_tokens, self.embed_dim)
        return self.drop(self.act(self.norm(x)))


class FFN(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d * 2, d))

    def forward(self, x):
        return self.net(x)


class SelfAttnBlock(nn.Module):
    def __init__(self, d, heads, dropout, attn_drop):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn  = nn.MultiheadAttention(d, heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ffn   = FFN(d, dropout)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class ORSD(nn.Module):
    """Orthogonal Residual-based Shared-unique Decoupling module.

    Disentangles dual-stream features into shared and unique components
    via cross-modal prediction residuals, with orthogonal regularization
    to enforce component independence.
    """
    def __init__(self, d: int, hid: int, dropout: float = 0.1):
        super().__init__()
        self.d     = d
        self.scale = math.sqrt(d)

        self.p2n = nn.Sequential(nn.Linear(d, hid), nn.GELU(), nn.Linear(hid, d))
        self.n2p = nn.Sequential(nn.Linear(d, hid), nn.GELU(), nn.Linear(hid, d))

        self.shared_proj_p = nn.Sequential(nn.Linear(d, d), nn.Tanh())
        self.shared_proj_n = nn.Sequential(nn.Linear(d, d), nn.Tanh())
        self.unique_gate_p = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())
        self.unique_gate_n = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())

        self.W_qp = nn.Linear(d, d, bias=False)
        self.W_kn = nn.Linear(d, d, bias=False)
        self.W_vn = nn.Linear(d, d, bias=False)
        self.W_qn = nn.Linear(d, d, bias=False)
        self.W_kp = nn.Linear(d, d, bias=False)
        self.W_vp = nn.Linear(d, d, bias=False)

        self.adap_gate = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )

        self.fuse = nn.Sequential(
            nn.Linear(4 * d, 2 * d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d, d), nn.LayerNorm(d)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, hp: torch.Tensor, hn: torch.Tensor):
        pred_n = self.p2n(hp)
        pred_p = self.n2p(hn)
        r_p = hp - pred_p
        r_n = hn - pred_n

        r_p_shared = self.shared_proj_p(r_p)
        r_n_shared = self.shared_proj_n(r_n)
        r_p_unique = r_p * self.unique_gate_p(r_p)
        r_n_unique = r_n * self.unique_gate_n(r_n)

        q_p = self.W_qp(r_p_shared)
        k_n = self.W_kn(r_n_shared)
        v_n = self.W_vn(r_n_shared)
        attn_pn = torch.sigmoid((q_p * k_n).sum(dim=-1, keepdim=True) / self.scale)
        cross_p = attn_pn * v_n + (1 - attn_pn) * r_p_shared

        q_n = self.W_qn(r_n_shared)
        k_p = self.W_kp(r_p_shared)
        v_p = self.W_vp(r_p_shared)
        attn_np = torch.sigmoid((q_n * k_p).sum(dim=-1, keepdim=True) / self.scale)
        cross_n = attn_np * v_p + (1 - attn_np) * r_n_shared

        norm_up   = r_p_unique.norm(dim=-1, keepdim=True)
        norm_un   = r_n_unique.norm(dim=-1, keepdim=True)
        norm_feats = torch.cat([norm_up, norm_un], dim=-1)
        adapt_w   = self.adap_gate(norm_feats)
        cross_p   = adapt_w * cross_p
        cross_n   = adapt_w * cross_n

        coupling = self.fuse(
            torch.cat([cross_p, cross_n, r_p_unique, r_n_unique], dim=-1)
        )
        coupling = self.drop(coupling)

        mse     = F.mse_loss(pred_n, hn.detach()) + F.mse_loss(pred_p, hp.detach())
        ortho_p = (r_p_shared * r_p_unique).sum(dim=-1).pow(2).mean()
        ortho_n = (r_n_shared * r_n_unique).sum(dim=-1).pow(2).mean()
        ortho_loss = ortho_p + ortho_n

        return coupling, mse, ortho_loss


class BCDAM(nn.Module):
    """Bilinear Cross-modal Dynamic Attention Module.

    Performs tensor-product interaction between dual-stream representations
    and dynamically gates the fused output via learnable dimension-wise weights.
    """
    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.d = d

        self.proj_p = nn.Linear(d, d)
        self.proj_n = nn.Linear(d, d)

        self.W_bip = nn.Linear(d, d, bias=False)
        self.W_bin = nn.Linear(d, d, bias=False)

        self.dim_gate = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(),
            nn.Linear(d * 2, d * 2)
        )

        self.interact_gate = nn.Sequential(
            nn.Linear(d, d), nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, hp: torch.Tensor, hn: torch.Tensor):
        z_interact = self.W_bip(hp) * self.W_bin(hn)

        ab_logit = self.dim_gate(z_interact)
        ab       = F.softmax(ab_logit.view(-1, self.d, 2), dim=-1)
        alpha    = ab[:, :, 0]
        beta     = ab[:, :, 1]

        up    = self.proj_p(hp)
        un    = self.proj_n(hn)
        gamma = self.interact_gate(z_interact)
        u     = alpha * up + beta * un + gamma * z_interact

        return self.norm(self.drop(u))


class ORBANet(nn.Module):
    """ORBA-Net: Orthogonal Residual-based Bilinear Attention Network.

    A dual-stream multimodal fusion architecture that integrates physical
    and network modality features through ORSD decoupling and BCDAM gating
    for robust intrusion detection in industrial cyber-physical systems.
    """
    def __init__(self, net_dim, phy_dim, num_classes, hp):
        super().__init__()
        self.use_coupling = hp['use_coupling']
        self.use_gate     = hp['use_gate']
        d = hp['embed_dim']

        self.phy_tok = FeatureTokenizer(phy_dim, d, 6, 0.2)
        self.net_tok = FeatureTokenizer(net_dim, d, 6, 0.2)
        self.phy_type = nn.Parameter(torch.zeros(1, 1, d))
        self.net_type = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.phy_type, std=0.02)
        nn.init.normal_(self.net_type, std=0.02)

        self.phy_blocks = nn.ModuleList([
            SelfAttnBlock(d, 8, 0.2, hp['attn_drop'])
            for _ in range(hp['num_layers'])])
        self.net_blocks = nn.ModuleList([
            SelfAttnBlock(d, 8, 0.2, hp['attn_drop'])
            for _ in range(hp['num_layers'])])
        self.phy_final = nn.LayerNorm(d)
        self.net_final = nn.LayerNorm(d)

        if self.use_coupling:
            self.coupling = ORSD(d, hp['couple_hid'], dropout=0.2)

        if self.use_gate:
            self.bcdam = BCDAM(d, dropout=0.2)
        else:
            self.proj_p = nn.Linear(d, d)
            self.proj_n = nn.Linear(d, d)

        fused_dim = 2 * d
        self.head = nn.Sequential(
            nn.Linear(fused_dim, d), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(d, num_classes))

    def forward(self, net_input, phy_input):
        device = net_input.device
        B = net_input.size(0)
        d = self.phy_tok.embed_dim

        phy = self.phy_tok(phy_input) + self.phy_type
        net = self.net_tok(net_input) + self.net_type
        for blk in self.phy_blocks:
            phy = blk(phy)
        for blk in self.net_blocks:
            net = blk(net)
        hp = self.phy_final(phy).mean(dim=1)
        hn = self.net_final(net).mean(dim=1)

        mse = torch.zeros((), device=device)
        ortho_loss = torch.zeros((), device=device)
        if self.use_coupling:
            coupling, mse, ortho_loss = self.coupling(hp, hn)
        else:
            coupling = torch.zeros(B, d, device=device)

        if self.use_gate:
            u = self.bcdam(hp, hn)
        else:
            u = 0.5 * (self.proj_p(hp) + self.proj_n(hn))

        fused  = torch.cat([u, coupling], dim=-1)
        logits = self.head(fused)

        return {'logits': logits, 'mse': mse, 'ortho_loss': ortho_loss}


def class_balanced_weights(labels_np, num_classes, beta=0.999):
    counts = np.bincount(labels_np, minlength=num_classes).astype(float)
    eff    = 1.0 - np.power(beta, counts)
    w      = (1.0 - beta) / np.maximum(eff, 1e-12)
    return w / w.sum() * num_classes


def get_warmup_cosine_schedule(optimizer, warmup_epochs, total_epochs, min_factor=0.01):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, device, num_classes, well_idx, mask=None):
    model.eval()
    preds, ys = [], []
    for net_input, phy_input, labels in loader:
        net_input = net_input.to(device)
        phy_input = phy_input.to(device)
        if mask == 'phy':
            phy_input = torch.zeros_like(phy_input)
        elif mask == 'net':
            net_input = torch.zeros_like(net_input)
        logits = model(net_input, phy_input)['logits']
        preds.append(logits.argmax(1).cpu())
        ys.append(labels)
    preds = torch.cat(preds).numpy()
    ys    = torch.cat(ys).numpy()
    per_class = f1_score(ys, preds, average=None,
                         labels=list(range(num_classes)), zero_division=0)
    return {
        'macro_well': float(per_class[well_idx].mean()),
        'macro_all':  float(per_class.mean()),
        'weighted':   float(f1_score(ys, preds, average='weighted', zero_division=0)),
        'per_class':  per_class,
        '_preds':     preds,
        '_labels':    ys,
    }


def train_and_evaluate(data, seed, device, batch_size=128):
    set_seed(seed)
    num_classes  = data['num_classes']
    class_names  = data['class_names']

    test_counts = np.bincount(data['labels_test'].numpy(), minlength=num_classes)
    well_idx    = np.where(test_counts >= 50)[0]
    if len(well_idx) == 0:
        well_idx = np.arange(num_classes)

    labels_train_np = data['labels_train'].numpy()
    all_idx = np.arange(len(labels_train_np))
    tr_idx, va_idx = train_test_split(
        all_idx, test_size=0.15, stratify=labels_train_np, random_state=seed)

    class_w      = class_balanced_weights(labels_train_np[tr_idx], num_classes)
    class_weights = torch.tensor(class_w, dtype=torch.float32, device=device)

    full_train = TensorDataset(
        data['network_train'], data['physical_train'], data['labels_train'])
    test_ds    = TensorDataset(
        data['network_test'],  data['physical_test'],  data['labels_test'])
    train_ds   = Subset(full_train, tr_idx.tolist())
    val_ds     = Subset(full_train, va_idx.tolist())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,  batch_size=256, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False)

    hp = {
        'embed_dim':     64,
        'num_layers':    2,
        'attn_drop':     0.01,
        'couple_hid':    128,
        'lambda_couple': 0.05,
        'lambda_ortho':  0.2,
        'num_epochs':    150,
        'use_coupling':  True,
        'use_gate':      True,
        'is_main':       True,
    }

    def build_model():
        return ORBANet(
            net_dim=data['network_train'].size(1),
            phy_dim=data['physical_train'].size(1),
            num_classes=num_classes,
            hp=hp).to(device)

    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    warmup    = max(3, hp['num_epochs'] // 10)
    scheduler = get_warmup_cosine_schedule(optimizer, warmup, hp['num_epochs'])
    use_amp   = (device.type == 'cuda')
    scaler    = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_val   = -1.0
    best_state = None

    epoch_records = []

    for epoch in range(hp['num_epochs']):
        model.train()
        t0 = time.time()
        run_loss = 0.0
        run_mse_ep   = 0.0
        run_ortho_ep = 0.0
        for net_input, phy_input, labels in train_loader:
            net_input = net_input.to(device, non_blocking=True)
            phy_input = phy_input.to(device, non_blocking=True)
            labels    = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                out  = model(net_input, phy_input)
                loss = F.cross_entropy(out['logits'], labels, weight=class_weights)
                if hp['use_coupling']:
                    loss = loss + hp['lambda_couple'] * out['mse'] \
                                + hp['lambda_ortho']  * out['ortho_loss']
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item()
            if hp['use_coupling']:
                run_mse_ep   += out['mse'].item()
                run_ortho_ep += out['ortho_loss'].item()
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device, num_classes, well_idx)
        vm = val_metrics['weighted']

        _n_bat     = max(1, len(train_loader))
        _avg_total = run_loss    / _n_bat
        _avg_mse   = run_mse_ep  / _n_bat
        _avg_ortho = run_ortho_ep / _n_bat
        if hp['use_coupling']:
            _avg_cls = (_avg_total
                        - hp['lambda_couple'] * _avg_mse
                        - hp['lambda_ortho']  * _avg_ortho)
        else:
            _avg_cls = _avg_total
        epoch_records.append({
            'epoch':        epoch + 1,
            'L_total':      round(_avg_total, 8),
            'L_cls':        round(_avg_cls,   8),
            'L_mse':        round(_avg_mse,   8),
            'L_ortho':      round(_avg_ortho, 8),
            'val_weighted': round(float(vm),  8),
            'lr':           round(scheduler.get_last_lr()[0], 8),
        })

        if vm > best_val:
            best_val   = vm
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        if (epoch + 1) % 20 == 0 or epoch == hp['num_epochs'] - 1:
            nb = max(1, len(train_loader))
            print(f"    Ep[{epoch+1:3d}/{hp['num_epochs']}] {time.time()-t0:4.1f}s | "
                  f"loss={run_loss/nb:.4f} | "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

    eval_model = build_model()
    if best_state is not None:
        eval_model.load_state_dict(best_state)

    clean = evaluate(eval_model, test_loader, device, num_classes, well_idx)
    d_phy = evaluate(eval_model, test_loader, device, num_classes, well_idx, mask='phy')
    d_net = evaluate(eval_model, test_loader, device, num_classes, well_idx, mask='net')

    print(classification_report(
        clean['_labels'], clean['_preds'],
        target_names=class_names, zero_division=0, digits=4))

    _fig_dir = './Fig_data'
    os.makedirs(_fig_dir, exist_ok=True)

    _csv_path = os.path.join(_fig_dir, 'training_metrics.csv')
    pd.DataFrame(epoch_records).to_csv(_csv_path, index=False, encoding='utf-8')
    print(f"  [Saved] Training metrics ({len(epoch_records)} epochs) -> {_csv_path}")

    _model_path = os.path.join(_fig_dir, 'full_model.pt')
    torch.save({
        'best_state':    best_state,
        'hp':            hp,
        'net_dim':       data['network_train'].size(1),
        'phy_dim':       data['physical_train'].size(1),
        'num_classes':   data['num_classes'],
        'class_names':   data['class_names'],
        'network_test':  data['network_test'],
        'physical_test': data['physical_test'],
        'labels_test':   data['labels_test'],
    }, _model_path)
    print(f"  [Saved] Model checkpoint -> {_model_path}")

    _total_params = sum(p.numel() for p in eval_model.parameters())
    _param_M      = _total_params / 1e6

    eval_model.eval()
    eval_model.to(device)
    with torch.no_grad():
        for _wn, _wp, _ in test_loader:
            _ = eval_model(_wn.to(device), _wp.to(device))['logits']
    if device.type == 'cuda':
        torch.cuda.synchronize()
    _t0 = time.perf_counter()
    with torch.no_grad():
        for _tn, _tp, _ in test_loader:
            _ = eval_model(_tn.to(device), _tp.to(device))['logits']
    if device.type == 'cuda':
        torch.cuda.synchronize()
    _t1           = time.perf_counter()
    _n_test       = len(data['labels_test'])
    _infer_tot_s  = _t1 - _t0
    _infer_per_s  = _infer_tot_s / _n_test
    _infer_per_us = _infer_per_s * 1e6

    _perf_path = os.path.join(_fig_dir, 'model_performance.csv')
    pd.DataFrame([
        {'metric': 'params_total',        'value': f'{_total_params}'},
        {'metric': 'params_M',            'value': f'{_param_M:.6f}'},
        {'metric': 'infer_total_s',       'value': f'{_infer_tot_s:.6f}'},
        {'metric': 'infer_per_sample_s',  'value': f'{_infer_per_s:.6f}'},
        {'metric': 'infer_per_sample_us', 'value': f'{_infer_per_us:.6f}'},
        {'metric': 'n_test_samples',      'value': f'{_n_test}'},
    ]).to_csv(_perf_path, index=False, encoding='utf-8')
    print(f"  [Saved] Parameters={_param_M:.6f} M  "
          f"Inference/sample={_infer_per_us:.6f} us -> {_perf_path}")

    result = {
        'weighted':          clean['weighted'],
        'macro_well':        clean['macro_well'],
        'macro_all':         clean['macro_all'],
        'per_class':         clean['per_class'],
        'drop_phy_weighted': d_phy['weighted'],
        'drop_net_weighted': d_net['weighted'],
        '_labels':           clean['_labels'],
        '_preds':            clean['_preds'],
    }
    return result


def _vw(s):
    return sum(2 if unicodedata.east_asian_width(c) in ('F', 'W') else 1 for c in s)

def _pad(s, w):
    return s + ' ' * max(0, w - _vw(s))

def print_single_result(result, class_names):
    """Print formatted tables for a single experimental result (no mean/std)."""
    name_w, col_w = 38, 18
    cols = ['F1 (Test)', 'Drop-Phy F1', 'Drop-Net F1']
    keys = ['weighted', 'drop_phy_weighted', 'drop_net_weighted']

    # --- Summary table ---
    header = ('| ' + _pad('Experiment / Architecture', name_w) + ' | '
              + ' | '.join(_pad(c, col_w) for c in cols) + ' |')
    sep = '-' * _vw(header)
    print('\n' + sep)
    print(header)
    print(sep)
    cells = [_pad(f"{result[k]:.4f}", col_w) for k in keys]
    print('| ' + _pad('ORBA-Net', name_w) + ' | ' + ' | '.join(cells) + ' |')
    print(sep)

    # --- Per-class F1 table ---
    print("\nPer-class F1 Scores:")
    head = ('| ' + _pad('Experiment / Architecture', name_w) + ' | '
            + ' | '.join(_pad(class_names[i][:10], 11) for i in range(len(class_names))) + ' |')
    print(head)
    print('-' * _vw(head))
    per_class = result['per_class']
    cells = ' | '.join(_pad(f"{per_class[i]:.4f}", 11) for i in range(len(class_names)))
    print('| ' + _pad('ORBA-Net', name_w) + ' | ' + cells + ' |')
    print('-' * _vw(head))
    print("\nNote: 'Drop-Phy' and 'Drop-Net' indicate results obtained by "
          "masking the physical and network modalities respectively, "
          "i.e., using only network-layer data and only physical-layer data.\n")

def run_experiment(train_path, test_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("Loading data...")
    data = load_data(train_path, test_path)
    num_classes = data['num_classes']
    class_names = data['class_names']

    print(f"Classes: {num_classes}, Labels: {class_names}")

    seed = 1
    print(f"\n--- Running ORBA-Net (seed={seed}) ---")
    result = train_and_evaluate(data, seed, device, batch_size=128)
    print_single_result(result, class_names)

    os.makedirs('./Fig', exist_ok=True)
    save_path = './Fig/experiment_results.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(result, f)
    print(f"Results saved to {save_path}")


if __name__ == '__main__':
    train_path = '../train.csv'
    test_path  = '../test.csv'
    run_experiment(train_path, test_path)
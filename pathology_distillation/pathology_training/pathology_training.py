#!/usr/bin/env python3
"""
🏥 FUNDUS PATHOLOGY CLASSIFICATION — 8 Patoloji (v6)
   1701 görüntü, 95 hasta
   
   ★ 8 label (v5'e göre RI, HEM, PVK eklendi):
     Diffüz kapiller sızıntı  : 296 (17.4%)
     Optik disk boyanması     : 151 ( 8.9%)
     Vitreus inflamasyonu     :  84 ( 4.9%)
     Makula ödemi             :  57 ( 3.4%)
     Damar duvar boyanması    :  56 ( 3.3%)
     Retinal infiltrat        :  26 ( 1.5%)  ★ YENİ
     Hemoraji                 :  14 ( 0.8%)  ★ YENİ
     Perivasküler kılıflanma  :  17 ( 1.0%)  ★ YENİ
   
   ★ CBAM + 8 Per-Patoloji Attention Head
   ★ Focal Loss (γ=2, rare class'lar için özellikle kritik)
   ★ Threshold-Optimized F1 ile best model seçimi
   ★ Balanced Stratified Group K-Fold
"""

import os, time, random, warnings, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, average_precision_score, confusion_matrix
)

try:
    from tqdm import tqdm
except ImportError:
    os.system('pip install tqdm --break-system-packages -q')
    from tqdm import tqdm

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIG
# ============================================================================
class Config:
    DATA_ROOT = r'C:\Users\gtu\Documents\cerrahpasa\files\files'
    DATASET_CSV = os.path.join(DATA_ROOT, 'Resimler_birlesik_TF_plusYorum_clean_numeric.csv')
    RESULTS_DIR = os.path.join(DATA_ROOT, 'results_pathology_v6')

    MODEL_NAME = 'efficientnet_b4'
    IMG_SIZE = 380
    BATCH_SIZE = 8
    NUM_WORKERS = 0
    EPOCHS = 30
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10
    N_FOLDS = 5
    SEED = 42

    # ★ 8 label — RI, HEM, PVK eklendi
    LABEL_COLS = [
        'Diffüz kapiller sızıntı',    # DKS — 296
        'Optik disk boyanması',        # ODB — 151
        'Vitreus inflamasyonu',        # VI  — 84
        'Makula ödemi',                # MÖ  — 57
        'Damar duvar boyanması',       # DDB — 56
        'Retinal infiltrat',           # RI  — 26  ★ YENİ
        'Hemoraji',                    # HEM — 14  ★ YENİ
        'Perivasküler kılıflanma',     # PVK — 17  ★ YENİ
    ]
    N_LABELS = len(LABEL_COLS)

    # Rare labels (N ≤ 30) — raporlarda ayrı gruplanır
    RARE_LABELS = ['Retinal infiltrat', 'Hemoraji', 'Perivasküler kılıflanma']

    # Focal Loss
    FOCAL_GAMMA = 2.0
    FOCAL_ALPHA = None

    # Threshold optimization
    THRESHOLD_RANGE = np.arange(0.10, 0.91, 0.02)

    # Attention visualization
    N_VIS_PER_LABEL = 5
    SAVE_ATTENTION_MAPS = True


SHORT_NAMES = {
    'Diffüz kapiller sızıntı': 'DKS',
    'Optik disk boyanması':    'ODB',
    'Vitreus inflamasyonu':    'VI',
    'Makula ödemi':            'MÖ',
    'Damar duvar boyanması':   'DDB',
    'Retinal infiltrat':       'RI',
    'Hemoraji':                'HEM',
    'Perivasküler kılıflanma': 'PVK',
}


def short_name(col):
    return SHORT_NAMES.get(col, col[:3])


def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

seed_everything(Config.SEED)


# ============================================================================
# DATASET
# ============================================================================
class FundusDataset(Dataset):
    def __init__(self, df, label_cols, transform=None):
        self.df = df.reset_index(drop=True)
        self.label_cols = label_cols
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        if self.transform: img = self.transform(img)
        labels = torch.tensor(row[self.label_cols].values.astype(np.float32))
        return img, labels


class FundusDatasetWithPath(Dataset):
    def __init__(self, df, label_cols, transform=None):
        self.df = df.reset_index(drop=True)
        self.label_cols = label_cols
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        if self.transform: img = self.transform(img)
        labels = torch.tensor(row[self.label_cols].values.astype(np.float32))
        return img, labels, row['image_path']


# ============================================================================
# FOCAL LOSS
# ============================================================================
class FocalLossWithLogits(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * bce
        if self.pos_weight is not None:
            weight = targets * self.pos_weight.unsqueeze(0) + (1 - targets)
            loss = loss * weight
        return loss.mean()


# ============================================================================
# CBAM
# ============================================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid = max(in_channels // reduction, 64)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid), nn.ReLU(inplace=True), nn.Linear(mid, in_channels))

    def forward(self, x):
        avg_out = self.mlp(x.mean(dim=[2, 3]))
        max_out = self.mlp(x.amax(dim=[2, 3]))
        return x * torch.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size, padding=pad, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, bias=False))

    def forward(self, x):
        cat = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(cat))


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_attn(self.channel_attn(x))


# ============================================================================
# PER-PATHOLOGY ATTENTION HEAD
# ============================================================================
class PathologyAttentionHead(nn.Module):
    def __init__(self, in_channels, mid_channels=256):
        super().__init__()
        self.attn_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1, bias=False),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_channels, 1)
        )

    def forward(self, features):
        attn_map = torch.sigmoid(self.attn_conv(features))
        weighted = features * attn_map
        pooled = weighted.mean(dim=[2, 3])
        logit = self.classifier(pooled).squeeze(-1)
        return logit, attn_map.squeeze(1)


# ============================================================================
# FULL MODEL
# ============================================================================
class FundusPathologyAttentionModel(nn.Module):
    def __init__(self, model_name='efficientnet_b4', n_labels=8):
        super().__init__()
        import timm
        self.backbone = timm.create_model(model_name, pretrained=True,
                                          num_classes=0, global_pool='')
        n_features = self.backbone.num_features

        self.cbam = CBAM(n_features, reduction=16, kernel_size=7)

        self.patho_heads = nn.ModuleList([
            PathologyAttentionHead(n_features, mid_channels=256)
            for _ in range(n_labels)
        ])
        self.n_labels = n_labels

    def forward(self, x, return_attention=False):
        features = self.cbam(self.backbone(x))
        logits, attn_maps = [], []
        for head in self.patho_heads:
            logit, attn = head(features)
            logits.append(logit)
            attn_maps.append(attn)
        logits = torch.stack(logits, dim=1)
        if return_attention:
            return logits, torch.stack(attn_maps, dim=1)
        return logits


# ============================================================================
# BALANCED STRATIFIED GROUP K-FOLD
# ============================================================================
def balanced_stratified_group_kfold_multilabel(df, label_cols, group_col, n_splits, seed=42):
    rng = np.random.RandomState(seed)
    n_labels = len(label_cols)
    groups = df[group_col].values
    unique_groups = np.unique(groups)

    group_profiles = {}
    for g in unique_groups:
        mask = groups == g
        group_profiles[g] = df.loc[mask, label_cols].values.sum(axis=0).astype(float)

    total_per_label = np.zeros(n_labels)
    for p in group_profiles.values():
        total_per_label += p
    ideal = total_per_label / n_splits

    label_rarity = 1.0 / np.maximum(total_per_label, 1.0)
    sorted_groups = sorted(unique_groups,
        key=lambda g: -(np.sum(group_profiles[g] * label_rarity) + rng.uniform(0, 1e-8)))

    fold_counts = np.zeros((n_splits, n_labels))
    fold_assignments = {}
    for g in sorted_groups:
        profile = group_profiles[g]
        best_fold, best_score = -1, -float('inf')
        for f in range(n_splits):
            deficit = ideal - fold_counts[f]
            benefit = np.sum(np.minimum(profile, np.maximum(deficit, 0)) * label_rarity)
            overflow = np.sum(np.maximum(fold_counts[f] + profile - ideal, 0) * label_rarity)
            score = benefit - overflow - fold_counts[f].sum() * 1e-6
            if score > best_score:
                best_score = score; best_fold = f
        fold_assignments[g] = best_fold
        fold_counts[best_fold] += profile

    folds = np.zeros(len(df), dtype=int)
    for g, f in fold_assignments.items():
        folds[groups == g] = f

    for f in range(n_splits):
        val_mask = folds == f
        yield np.where(~val_mask)[0], np.where(val_mask)[0]


# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================
def find_optimal_thresholds(logits, labels, label_cols, threshold_range):
    probs = torch.sigmoid(logits).numpy()
    y_true = labels.numpy()
    thresholds = {}
    for i, col in enumerate(label_cols):
        gt = y_true[:, i]
        best_t, best_f1 = 0.5, 0
        for t in threshold_range:
            pred = (probs[:, i] >= t).astype(int)
            f1 = f1_score(gt, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1; best_t = t
        thresholds[col] = round(best_t, 2)
    return thresholds


def compute_metrics_with_thresholds(logits, labels, label_cols, thresholds):
    probs = torch.sigmoid(logits).numpy()
    y_true = labels.numpy()
    results = {}
    for i, col in enumerate(label_cols):
        gt = y_true[:, i]
        t = thresholds.get(col, 0.5)
        pr = (probs[:, i] >= t).astype(int)
        pb = probs[:, i]
        n_pos = int(gt.sum())
        auc = roc_auc_score(gt, pb) if 0 < n_pos < len(gt) else None
        ap = average_precision_score(gt, pb) if n_pos > 0 else None
        cm = confusion_matrix(gt, pr, labels=[0, 1])
        tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        results[col] = {
            'n_pos': n_pos, 'threshold': t, 'auc': auc, 'ap': ap,
            'f1': f1_score(gt, pr, zero_division=0),
            'prec': precision_score(gt, pr, zero_division=0),
            'rec': recall_score(gt, pr, zero_division=0),
            'acc': accuracy_score(gt, pr),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        }

    # All macro
    macro = {k: np.mean([r[k] for r in results.values() if r.get(k) is not None])
             for k in ['f1','prec','rec','acc']}
    aucs = [r['auc'] for r in results.values() if r['auc'] is not None]
    aps  = [r['ap']  for r in results.values() if r['ap']  is not None]
    macro['auc'] = np.mean(aucs) if aucs else None
    macro['ap']  = np.mean(aps)  if aps  else None
    results['__macro__'] = macro

    # Rare-only macro (RI, HEM, PVK)
    rare_labels = [c for c in label_cols if c in Config.RARE_LABELS]
    if rare_labels:
        rare_f1s  = [results[c]['f1']  for c in rare_labels]
        rare_aucs = [results[c]['auc'] for c in rare_labels if results[c]['auc'] is not None]
        rare_aps  = [results[c]['ap']  for c in rare_labels if results[c]['ap']  is not None]
        rare_prec = [results[c]['prec'] for c in rare_labels]
        rare_rec  = [results[c]['rec']  for c in rare_labels]
        results['__rare__'] = {
            'f1':   np.mean(rare_f1s),
            'auc':  np.mean(rare_aucs) if rare_aucs else None,
            'ap':   np.mean(rare_aps)  if rare_aps  else None,
            'prec': np.mean(rare_prec),
            'rec':  np.mean(rare_rec),
        }
    return results


# ============================================================================
# TRAINING
# ============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, n_epochs):
    model.train()
    total_loss, n_total = 0, 0
    pbar = tqdm(loader, desc=f"  Train {epoch+1:2d}/{n_epochs}",
                bar_format='{l_bar}{bar:30}{r_bar}', leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        n_total += imgs.size(0)
        pbar.set_postfix(loss=f"{total_loss/n_total:.4f}")
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


# ============================================================================
# ATTENTION VISUALIZATION
# ============================================================================
@torch.no_grad()
def extract_attention_maps(model, loader, device):
    model.eval()
    all_attn, all_logits, all_labels, all_paths = [], [], [], []
    for imgs, labels, paths in tqdm(loader, desc="  Attention extraction",
                                     bar_format='{l_bar}{bar:30}{r_bar}', leave=False):
        imgs = imgs.to(device)
        logits, attn_maps = model(imgs, return_attention=True)
        all_attn.append(attn_maps.cpu())
        all_logits.append(logits.cpu())
        all_labels.append(labels)
        all_paths.extend(paths)
    return torch.cat(all_attn), torch.cat(all_logits), torch.cat(all_labels), all_paths


def save_attention_visualization(img_path, attn_maps, gt_labels, pred_labels,
                                  label_names, save_path, img_size=380):
    img = Image.open(img_path).convert('RGB').resize((img_size, img_size))
    img_np = np.array(img) / 255.0
    n = len(label_names)
    fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))

    gt_str   = " + ".join([short_name(label_names[i]) for i in range(n) if gt_labels[i] == 1])   or "Temiz"
    pred_str = " + ".join([short_name(label_names[i]) for i in range(n) if pred_labels[i] == 1]) or "Temiz"
    match = "✓" if (gt_labels == pred_labels).all() else "✗"
    fig.suptitle(f'GT=[{gt_str}]  Pred=[{pred_str}]  ({match})', fontsize=11, fontweight='bold')

    axes[0].imshow(img_np); axes[0].set_title('Original'); axes[0].axis('off')
    for i in range(n):
        ax = axes[i + 1]
        attn = attn_maps[i].numpy()
        attn_up = np.array(Image.fromarray(attn).resize((img_size, img_size), Image.BILINEAR))
        if attn_up.max() > attn_up.min():
            attn_up = (attn_up - attn_up.min()) / (attn_up.max() - attn_up.min())
        ax.imshow(img_np)
        ax.imshow(attn_up, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        sn = short_name(label_names[i])
        gt_s = "VAR" if gt_labels[i] == 1 else "YOK"
        pred_s = "VAR" if pred_labels[i] == 1 else "YOK"
        color = 'green' if gt_labels[i] == pred_labels[i] else 'red'
        ax.set_title(f'{sn}\nGT={gt_s} P={pred_s}', fontsize=9, color=color, fontweight='bold')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=90, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================
def main():
    cfg = Config()
    results_dir = Path(cfg.RESULTS_DIR)
    results_dir.mkdir(exist_ok=True, parents=True)
    (results_dir / 'attention_vis').mkdir(exist_ok=True)

    device = torch.device('mps'  if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
                     else 'cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("🏥 FUNDUS PATHOLOGY — 8 Label (v6 — RI/HEM/PVK eklendi)")
    print(f"   Model: {cfg.MODEL_NAME} + CBAM + 8 Pathology Attention Heads")
    print(f"   Device: {device}")
    print(f"   Loss: Focal Loss (γ={cfg.FOCAL_GAMMA})")
    print("=" * 80)

    # ── Load data ──
    df = pd.read_csv(cfg.DATASET_CSV, encoding='utf-8')
    df.columns = [c.strip().replace('\xa0', '') for c in df.columns]
    df['patient_id'] = df['Klasör'].astype(str)
    df['image_name'] = df['Dosya ismi'].astype(str)
    df['image_path'] = df.apply(
        lambda r: os.path.join(cfg.DATA_ROOT, r['patient_id'], r['image_name']), axis=1)

    before = len(df)
    df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)
    if len(df) < before:
        print(f"  ⚠️ {before - len(df)} dosya bulunamadı")

    # ★ Eksik label sütunlarını kontrol et ve 0 ile doldur
    for c in cfg.LABEL_COLS:
        if c not in df.columns:
            print(f"  ⚠️ '{c}' sütunu bulunamadı, 0 ile dolduruldu")
            df[c] = 0
        else:
            df[c] = df[c].astype(int)

    # ★ Sütun adlarında whitespace/encoding sorunları için ek kontrol
    col_map = {col.strip().replace('\xa0', ''): col for col in df.columns}
    for label in cfg.LABEL_COLS:
        clean = label.strip().replace('\xa0', '')
        if clean in col_map and col_map[clean] != label:
            df[label] = df[col_map[clean]]

    print(f"\n  Dataset: {len(df)} görüntü, {df['patient_id'].nunique()} hasta")
    print(f"\n  Label dağılımı (8 label):")
    for col in cfg.LABEL_COLS:
        n = int(df[col].sum())
        rare_tag = " ★RARE" if col in cfg.RARE_LABELS else ""
        print(f"    {short_name(col):4s}  {col:35s}: {n:5d} ({n/len(df)*100:5.1f}%){rare_tag}")

    # ── Class weights (rare sınıflar için daha yüksek) ──
    pos_counts = df[cfg.LABEL_COLS].sum().values
    neg_counts = len(df) - pos_counts
    pos_weight = torch.tensor(neg_counts / np.maximum(pos_counts, 1), dtype=torch.float32).to(device)
    pos_weight = torch.clamp(pos_weight, min=1.0, max=50.0)  # Rare sınıflar için max 50
    print(f"\n  Pos weights (Focal Loss alpha):")
    for i, col in enumerate(cfg.LABEL_COLS):
        rare_tag = " ★" if col in cfg.RARE_LABELS else ""
        print(f"    {short_name(col):4s}: {pos_weight[i].item():.2f}{rare_tag}")

    # ── Transforms ──
    train_transform = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ── Balanced folds ──
    fold_splits = list(balanced_stratified_group_kfold_multilabel(
        df, cfg.LABEL_COLS, 'patient_id', cfg.N_FOLDS, cfg.SEED))

    sns = [short_name(c) for c in cfg.LABEL_COLS]
    total_per_label = df[cfg.LABEL_COLS].sum().values
    print(f"\n  ★ Balanced Fold dağılımı:")
    header = f"    Fold |"
    for sn in sns: header += f" {sn:>7s} |"
    header += "  Total"
    print(header); print(f"  {'─'*len(header)}")
    for fi, (_, vi) in enumerate(fold_splits):
        vdf = df.iloc[vi]; row_str = f"     {fi+1}/{cfg.N_FOLDS} |"
        for i, col in enumerate(cfg.LABEL_COLS):
            n = int(vdf[col].sum()); total = int(total_per_label[i])
            pct = n/total*100 if total > 0 else 0
            row_str += f" {n:>3d}({pct:2.0f}%) |"
        row_str += f"  {len(vi):>5d}"; print(row_str)

    # ══════════════════════════════════════════════════════════════════════════
    # TRAINING
    # ══════════════════════════════════════════════════════════════════════════
    all_fold_results = []
    all_val_logits = []
    all_val_labels = []
    all_val_indices = []
    all_val_attn = []
    all_val_paths = []
    all_fold_thresholds = []

    t0 = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"\n{'='*80}")
        print(f"  FOLD {fold_idx+1}/{cfg.N_FOLDS}")
        print(f"{'='*80}")

        train_df = df.iloc[train_idx]; val_df = df.iloc[val_idx]
        val_pos = val_df[cfg.LABEL_COLS].sum()
        pos_str = " | ".join([f"{short_name(c)}:{int(val_pos[c])}" for c in cfg.LABEL_COLS])
        print(f"  Train: {len(train_df)} imgs, {train_df['patient_id'].nunique()} pts")
        print(f"  Val:   {len(val_df)} imgs, {val_df['patient_id'].nunique()} pts")
        print(f"  Val pos: {pos_str}")

        overlap = set(train_df['patient_id']) & set(val_df['patient_id'])
        print(f"  {'⚠️ LEAK!' if overlap else '✓ No leak'}")

        train_ds = FundusDataset(train_df, cfg.LABEL_COLS, train_transform)
        val_ds   = FundusDataset(val_df,   cfg.LABEL_COLS, val_transform)
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                  num_workers=cfg.NUM_WORKERS, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False,
                                  num_workers=cfg.NUM_WORKERS, pin_memory=True)

        model = FundusPathologyAttentionModel(cfg.MODEL_NAME, cfg.N_LABELS).to(device)
        criterion = FocalLossWithLogits(gamma=cfg.FOCAL_GAMMA, pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

        best_macro_f1 = 0
        patience_counter = 0
        best_logits, best_labels = None, None
        best_state = None
        best_thresholds = {c: 0.5 for c in cfg.LABEL_COLS}

        for epoch in range(cfg.EPOCHS):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                         epoch, cfg.EPOCHS)
            val_logits, val_labels = evaluate(model, val_loader, device)
            scheduler.step()

            epoch_thresholds = find_optimal_thresholds(
                val_logits, val_labels, cfg.LABEL_COLS, cfg.THRESHOLD_RANGE)
            epoch_metrics = compute_metrics_with_thresholds(
                val_logits, val_labels, cfg.LABEL_COLS, epoch_thresholds)
            macro_f1 = epoch_metrics['__macro__']['f1']

            # Rare macro for monitoring
            rare_f1 = epoch_metrics.get('__rare__', {}).get('f1', 0)

            label_str = " ".join([
                f"{short_name(c)}={epoch_metrics[c]['f1']:.2f}"
                for c in cfg.LABEL_COLS])

            improved = macro_f1 > best_macro_f1
            if improved:
                best_macro_f1 = macro_f1
                patience_counter = 0
                best_logits = val_logits
                best_labels = val_labels
                best_thresholds = epoch_thresholds
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(model.state_dict(), results_dir / f'model_fold{fold_idx}.pth')
            else:
                patience_counter += 1

            star = " ★" if improved else ""
            print(f"    Ep {epoch+1:2d}/{cfg.EPOCHS}: loss={train_loss:.4f} "
                  f"mF1={macro_f1:.3f} rareF1={rare_f1:.3f} | {label_str}{star}")

            if patience_counter >= cfg.PATIENCE:
                print(f"    ⏹ Early stop @ epoch {epoch+1}")
                break

        # ── Fold results ──
        fold_metrics = compute_metrics_with_thresholds(
            best_logits, best_labels, cfg.LABEL_COLS, best_thresholds)
        macro = fold_metrics['__macro__']
        rare  = fold_metrics.get('__rare__', {})

        all_fold_results.append(fold_metrics)
        all_val_logits.append(best_logits)
        all_val_labels.append(best_labels)
        all_val_indices.extend(val_idx.tolist())
        all_fold_thresholds.append(best_thresholds)

        print(f"\n  ✅ Fold {fold_idx+1}: mF1={macro['f1']:.4f} AUC={macro['auc']:.4f} "
              f"rareF1={rare.get('f1', 0):.4f}")

        print(f"  {'Label':6s} {'t':>5s} {'F1':>6s} {'AUC':>6s} {'Prec':>6s} {'Rec':>6s} {'TP':>4s} {'FP':>4s} {'FN':>4s}")
        print(f"  {'─'*60}")
        for col in cfg.LABEL_COLS:
            m = fold_metrics[col]
            auc_s = f"{m['auc']:.3f}" if m['auc'] else " N/A"
            rare_tag = "★" if col in cfg.RARE_LABELS else " "
            print(f"  {rare_tag}{short_name(col):5s} {m['threshold']:>5.2f} {m['f1']:6.3f} "
                  f"{auc_s:>6s} {m['prec']:6.3f} {m['rec']:6.3f} "
                  f"{m['tp']:4d} {m['fp']:4d} {m['fn']:4d}")

        # ── Attention maps ──
        if cfg.SAVE_ATTENTION_MAPS and best_state:
            print(f"\n  🔍 Attention map extraction (Fold {fold_idx+1})...")
            model.load_state_dict(best_state); model.to(device)
            vis_ds = FundusDatasetWithPath(val_df, cfg.LABEL_COLS, val_transform)
            vis_loader = DataLoader(vis_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                                    num_workers=cfg.NUM_WORKERS)
            fold_attn, fold_logits_vis, fold_labels_vis, fold_paths = extract_attention_maps(
                model, vis_loader, device)

            all_val_attn.append(fold_attn)
            all_val_paths.extend(fold_paths)

            fold_probs = torch.sigmoid(fold_logits_vis).numpy()
            fold_preds = np.zeros_like(fold_probs, dtype=int)
            for i, col in enumerate(cfg.LABEL_COLS):
                fold_preds[:, i] = (fold_probs[:, i] >= best_thresholds[col]).astype(int)
            fold_gt = fold_labels_vis.numpy().astype(int)

            vis_dir = results_dir / 'attention_vis' / f'fold{fold_idx+1}'
            vis_dir.mkdir(exist_ok=True, parents=True)
            n_saved = 0

            for li, col in enumerate(cfg.LABEL_COLS):
                sn = short_name(col)
                tp_mask = (fold_gt[:, li] == 1) & (fold_preds[:, li] == 1)
                for j, idx in enumerate(np.where(tp_mask)[0][:cfg.N_VIS_PER_LABEL]):
                    save_attention_visualization(
                        fold_paths[idx], fold_attn[idx], fold_gt[idx], fold_preds[idx],
                        cfg.LABEL_COLS, vis_dir / f'TP_{sn}_{j+1}.png', cfg.IMG_SIZE)
                    n_saved += 1

                fn_mask = (fold_gt[:, li] == 1) & (fold_preds[:, li] == 0)
                for j, idx in enumerate(np.where(fn_mask)[0][:3]):
                    save_attention_visualization(
                        fold_paths[idx], fold_attn[idx], fold_gt[idx], fold_preds[idx],
                        cfg.LABEL_COLS, vis_dir / f'FN_{sn}_{j+1}.png', cfg.IMG_SIZE)
                    n_saved += 1

            print(f"  💾 {n_saved} attention vis → {vis_dir}/")

    elapsed = (time.time() - t0) / 60

    # ══════════════════════════════════════════════════════════════════════════
    # AGGREGATE + FINAL THRESHOLD OPTIMIZATION
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("📊 FINAL SONUÇLAR — 5-Fold CV")
    print(f"{'='*80}")

    all_logits_cat = torch.cat(all_val_logits)
    all_labels_cat = torch.cat(all_val_labels)

    final_thresholds = find_optimal_thresholds(
        all_logits_cat, all_labels_cat, cfg.LABEL_COLS, cfg.THRESHOLD_RANGE)

    overall = compute_metrics_with_thresholds(
        all_logits_cat, all_labels_cat, cfg.LABEL_COLS, final_thresholds)
    macro = overall['__macro__']
    rare  = overall.get('__rare__', {})

    print(f"\n  Final Thresholds: {final_thresholds}")
    print(f"\n  ── Macro (8 label) ──")
    print(f"  Macro F1:        {macro['f1']:.4f}")
    print(f"  Macro AUC:       {macro['auc']:.4f}")
    print(f"  Macro AP:        {macro['ap']:.4f}")
    print(f"  Macro Precision: {macro['prec']:.4f}")
    print(f"  Macro Recall:    {macro['rec']:.4f}")

    print(f"\n  ── Rare-class macro (RI, HEM, PVK) ──")
    print(f"  Rare F1:         {rare.get('f1', 0):.4f}")
    print(f"  Rare AUC:        {rare.get('auc', 0):.4f}")
    print(f"  Rare Prec:       {rare.get('prec', 0):.4f}")
    print(f"  Rare Rec:        {rare.get('rec', 0):.4f}")

    print(f"\n  {'':1s}{'Patoloji':35s} {'t':>5s} {'N':>5s} {'F1':>7s} {'AUC':>7s} {'AP':>7s} {'Prec':>7s} {'Rec':>7s} {'TP':>4s} {'FP':>4s} {'FN':>4s}")
    print(f"  {'─'*100}")
    for col in cfg.LABEL_COLS:
        m = overall[col]
        auc_s = f"{m['auc']:.4f}" if m['auc'] else "  N/A"
        ap_s  = f"{m['ap']:.4f}"  if m['ap']  else "  N/A"
        rare_tag = "★" if col in cfg.RARE_LABELS else " "
        print(f"  {rare_tag}{col:35s} {m['threshold']:>5.2f} {m['n_pos']:>5d} "
              f"{m['f1']:>7.4f} {auc_s:>7s} {ap_s:>7s} "
              f"{m['prec']:>7.4f} {m['rec']:>7.4f} {m['tp']:>4d} {m['fp']:>4d} {m['fn']:>4d}")

    print(f"\n  {'':36s} {'':>5s} {'':>5s} {macro['f1']:>7.4f} {macro['auc']:>7.4f} {macro['ap']:>7.4f}")
    print(f"  Macro avg")

    # t=0.5 comparison
    baseline = compute_metrics_with_thresholds(
        all_logits_cat, all_labels_cat, cfg.LABEL_COLS, {c: 0.5 for c in cfg.LABEL_COLS})
    base_macro = baseline['__macro__']
    print(f"\n  Threshold opt. gain: mF1 {base_macro['f1']:.4f} → {macro['f1']:.4f} "
          f"({macro['f1']-base_macro['f1']:+.4f})")

    # Per-fold summary
    print(f"\n  Per-fold:")
    for i, fm in enumerate(all_fold_results):
        m = fm['__macro__']; r = fm.get('__rare__', {})
        print(f"    Fold {i+1}: mF1={m['f1']:.4f} AUC={m['auc']:.4f} "
              f"rareF1={r.get('f1',0):.4f}")

    # ── Save ──
    save_results = {
        'pipeline': 'Pathology_8_v6_attention_focal',
        'model': f'{cfg.MODEL_NAME} + CBAM + 8 Attention Heads',
        'n_labels': cfg.N_LABELS, 'label_cols': cfg.LABEL_COLS,
        'rare_labels': cfg.RARE_LABELS,
        'n_images': len(df), 'n_patients': int(df['patient_id'].nunique()),
        'n_folds': cfg.N_FOLDS, 'elapsed_min': round(elapsed, 1),
        'final_thresholds': {k: float(v) for k, v in final_thresholds.items()},
        'macro_f1': round(float(macro['f1']), 4),
        'macro_auc': round(float(macro['auc']), 4) if macro['auc'] else None,
        'macro_ap': round(float(macro['ap']), 4) if macro['ap'] else None,
        'macro_prec': round(float(macro['prec']), 4),
        'macro_rec': round(float(macro['rec']), 4),
        'rare_f1': round(float(rare['f1']), 4) if rare.get('f1') else None,
        'rare_auc': round(float(rare['auc']), 4) if rare.get('auc') else None,
        'per_label': {},
    }
    for col in cfg.LABEL_COLS:
        m = overall[col]
        save_results['per_label'][col] = {
            k: round(float(v), 4) if v is not None and not isinstance(v, int) else v
            for k, v in m.items()
        }

    with open(results_dir / 'summary_pathology_v6.json', 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)

    pred_df = df.iloc[all_val_indices].copy()
    probs_all = torch.sigmoid(all_logits_cat).numpy()
    for i, col in enumerate(cfg.LABEL_COLS):
        pred_df[f'prob_{col}'] = probs_all[:, i]
        pred_df[f'pred_{col}'] = (probs_all[:, i] >= final_thresholds[col]).astype(int)
    pred_df.to_csv(results_dir / 'predictions_pathology_v6.csv', index=False)

    with open(results_dir / 'optimal_thresholds.json', 'w') as f:
        json.dump({k: float(v) for k, v in final_thresholds.items()}, f, indent=2)

    print(f"\n  💾 {results_dir}/")
    print(f"  ⏱ {elapsed:.1f} dk")
    print(f"\n{'='*80}")
    print(f"✅ PATHOLOGY v6 TAMAMLANDI! (8 label: DKS/ODB/VI/MÖ/DDB/RI/HEM/PVK)")
    print(f"   Macro F1: {macro['f1']:.4f}  AUC: {macro['auc']:.4f}  AP: {macro['ap']:.4f}")
    print(f"   Rare F1:  {rare.get('f1',0):.4f}  (RI/HEM/PVK)")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()#!/usr/bin/env python3
"""
🏥 FUNDUS PATHOLOGY CLASSIFICATION — 8 Patoloji (v6)
   1701 görüntü, 95 hasta
   
   ★ 8 label (v5'e göre RI, HEM, PVK eklendi):
     Diffüz kapiller sızıntı  : 296 (17.4%)
     Optik disk boyanması     : 151 ( 8.9%)
     Vitreus inflamasyonu     :  84 ( 4.9%)
     Makula ödemi             :  57 ( 3.4%)
     Damar duvar boyanması    :  56 ( 3.3%)
     Retinal infiltrat        :  26 ( 1.5%)  ★ YENİ
     Hemoraji                 :  14 ( 0.8%)  ★ YENİ
     Perivasküler kılıflanma  :  17 ( 1.0%)  ★ YENİ
   
   ★ CBAM + 8 Per-Patoloji Attention Head
   ★ Focal Loss (γ=2, rare class'lar için özellikle kritik)
   ★ Threshold-Optimized F1 ile best model seçimi
   ★ Balanced Stratified Group K-Fold
"""

import os, time, random, warnings, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, average_precision_score, confusion_matrix
)

try:
    from tqdm import tqdm
except ImportError:
    os.system('pip install tqdm --break-system-packages -q')
    from tqdm import tqdm

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIG
# ============================================================================
class Config:
    DATA_ROOT = r'C:\Users\gtu\Documents\cerrahpasa\files\files'
    DATASET_CSV = os.path.join(DATA_ROOT, 'Resimler_birlesik_TF_plusYorum_clean_numeric.csv')
    RESULTS_DIR = os.path.join(DATA_ROOT, 'results_pathology_v6')

    MODEL_NAME = 'efficientnet_b4'
    IMG_SIZE = 380
    BATCH_SIZE = 8
    NUM_WORKERS = 0
    EPOCHS = 30
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10
    N_FOLDS = 5
    SEED = 42

    # ★ 8 label — RI, HEM, PVK eklendi
    LABEL_COLS = [
        'Diffüz kapiller sızıntı',    # DKS — 296
        'Optik disk boyanması',        # ODB — 151
        'Vitreus inflamasyonu',        # VI  — 84
        'Makula ödemi',                # MÖ  — 57
        'Damar duvar boyanması',       # DDB — 56
        'Retinal infiltrat',           # RI  — 26  ★ YENİ
        'Hemoraji',                    # HEM — 14  ★ YENİ
        'Perivasküler kılıflanma',     # PVK — 17  ★ YENİ
    ]
    N_LABELS = len(LABEL_COLS)

    # Rare labels (N ≤ 30) — raporlarda ayrı gruplanır
    RARE_LABELS = ['Retinal infiltrat', 'Hemoraji', 'Perivasküler kılıflanma']

    # Focal Loss
    FOCAL_GAMMA = 2.0
    FOCAL_ALPHA = None

    # Threshold optimization
    THRESHOLD_RANGE = np.arange(0.10, 0.91, 0.02)

    # Attention visualization
    N_VIS_PER_LABEL = 5
    SAVE_ATTENTION_MAPS = True


SHORT_NAMES = {
    'Diffüz kapiller sızıntı': 'DKS',
    'Optik disk boyanması':    'ODB',
    'Vitreus inflamasyonu':    'VI',
    'Makula ödemi':            'MÖ',
    'Damar duvar boyanması':   'DDB',
    'Retinal infiltrat':       'RI',
    'Hemoraji':                'HEM',
    'Perivasküler kılıflanma': 'PVK',
}


def short_name(col):
    return SHORT_NAMES.get(col, col[:3])


def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

seed_everything(Config.SEED)


# ============================================================================
# DATASET
# ============================================================================
class FundusDataset(Dataset):
    def __init__(self, df, label_cols, transform=None):
        self.df = df.reset_index(drop=True)
        self.label_cols = label_cols
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        if self.transform: img = self.transform(img)
        labels = torch.tensor(row[self.label_cols].values.astype(np.float32))
        return img, labels


class FundusDatasetWithPath(Dataset):
    def __init__(self, df, label_cols, transform=None):
        self.df = df.reset_index(drop=True)
        self.label_cols = label_cols
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        if self.transform: img = self.transform(img)
        labels = torch.tensor(row[self.label_cols].values.astype(np.float32))
        return img, labels, row['image_path']


# ============================================================================
# FOCAL LOSS
# ============================================================================
class FocalLossWithLogits(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * bce
        if self.pos_weight is not None:
            weight = targets * self.pos_weight.unsqueeze(0) + (1 - targets)
            loss = loss * weight
        return loss.mean()


# ============================================================================
# CBAM
# ============================================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid = max(in_channels // reduction, 64)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid), nn.ReLU(inplace=True), nn.Linear(mid, in_channels))

    def forward(self, x):
        avg_out = self.mlp(x.mean(dim=[2, 3]))
        max_out = self.mlp(x.amax(dim=[2, 3]))
        return x * torch.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size, padding=pad, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, bias=False))

    def forward(self, x):
        cat = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(cat))


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_attn(self.channel_attn(x))


# ============================================================================
# PER-PATHOLOGY ATTENTION HEAD
# ============================================================================
class PathologyAttentionHead(nn.Module):
    def __init__(self, in_channels, mid_channels=256):
        super().__init__()
        self.attn_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1, bias=False),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_channels, 1)
        )

    def forward(self, features):
        attn_map = torch.sigmoid(self.attn_conv(features))
        weighted = features * attn_map
        pooled = weighted.mean(dim=[2, 3])
        logit = self.classifier(pooled).squeeze(-1)
        return logit, attn_map.squeeze(1)


# ============================================================================
# FULL MODEL
# ============================================================================
class FundusPathologyAttentionModel(nn.Module):
    def __init__(self, model_name='efficientnet_b4', n_labels=8):
        super().__init__()
        import timm
        self.backbone = timm.create_model(model_name, pretrained=True,
                                          num_classes=0, global_pool='')
        n_features = self.backbone.num_features

        self.cbam = CBAM(n_features, reduction=16, kernel_size=7)

        self.patho_heads = nn.ModuleList([
            PathologyAttentionHead(n_features, mid_channels=256)
            for _ in range(n_labels)
        ])
        self.n_labels = n_labels

    def forward(self, x, return_attention=False):
        features = self.cbam(self.backbone(x))
        logits, attn_maps = [], []
        for head in self.patho_heads:
            logit, attn = head(features)
            logits.append(logit)
            attn_maps.append(attn)
        logits = torch.stack(logits, dim=1)
        if return_attention:
            return logits, torch.stack(attn_maps, dim=1)
        return logits


# ============================================================================
# BALANCED STRATIFIED GROUP K-FOLD
# ============================================================================
def balanced_stratified_group_kfold_multilabel(df, label_cols, group_col, n_splits, seed=42):
    rng = np.random.RandomState(seed)
    n_labels = len(label_cols)
    groups = df[group_col].values
    unique_groups = np.unique(groups)

    group_profiles = {}
    for g in unique_groups:
        mask = groups == g
        group_profiles[g] = df.loc[mask, label_cols].values.sum(axis=0).astype(float)

    total_per_label = np.zeros(n_labels)
    for p in group_profiles.values():
        total_per_label += p
    ideal = total_per_label / n_splits

    label_rarity = 1.0 / np.maximum(total_per_label, 1.0)
    sorted_groups = sorted(unique_groups,
        key=lambda g: -(np.sum(group_profiles[g] * label_rarity) + rng.uniform(0, 1e-8)))

    fold_counts = np.zeros((n_splits, n_labels))
    fold_assignments = {}
    for g in sorted_groups:
        profile = group_profiles[g]
        best_fold, best_score = -1, -float('inf')
        for f in range(n_splits):
            deficit = ideal - fold_counts[f]
            benefit = np.sum(np.minimum(profile, np.maximum(deficit, 0)) * label_rarity)
            overflow = np.sum(np.maximum(fold_counts[f] + profile - ideal, 0) * label_rarity)
            score = benefit - overflow - fold_counts[f].sum() * 1e-6
            if score > best_score:
                best_score = score; best_fold = f
        fold_assignments[g] = best_fold
        fold_counts[best_fold] += profile

    folds = np.zeros(len(df), dtype=int)
    for g, f in fold_assignments.items():
        folds[groups == g] = f

    for f in range(n_splits):
        val_mask = folds == f
        yield np.where(~val_mask)[0], np.where(val_mask)[0]


# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================
def find_optimal_thresholds(logits, labels, label_cols, threshold_range):
    probs = torch.sigmoid(logits).numpy()
    y_true = labels.numpy()
    thresholds = {}
    for i, col in enumerate(label_cols):
        gt = y_true[:, i]
        best_t, best_f1 = 0.5, 0
        for t in threshold_range:
            pred = (probs[:, i] >= t).astype(int)
            f1 = f1_score(gt, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1; best_t = t
        thresholds[col] = round(best_t, 2)
    return thresholds


def compute_metrics_with_thresholds(logits, labels, label_cols, thresholds):
    probs = torch.sigmoid(logits).numpy()
    y_true = labels.numpy()
    results = {}
    for i, col in enumerate(label_cols):
        gt = y_true[:, i]
        t = thresholds.get(col, 0.5)
        pr = (probs[:, i] >= t).astype(int)
        pb = probs[:, i]
        n_pos = int(gt.sum())
        auc = roc_auc_score(gt, pb) if 0 < n_pos < len(gt) else None
        ap = average_precision_score(gt, pb) if n_pos > 0 else None
        cm = confusion_matrix(gt, pr, labels=[0, 1])
        tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        results[col] = {
            'n_pos': n_pos, 'threshold': t, 'auc': auc, 'ap': ap,
            'f1': f1_score(gt, pr, zero_division=0),
            'prec': precision_score(gt, pr, zero_division=0),
            'rec': recall_score(gt, pr, zero_division=0),
            'acc': accuracy_score(gt, pr),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        }

    # All macro
    macro = {k: np.mean([r[k] for r in results.values() if r.get(k) is not None])
             for k in ['f1','prec','rec','acc']}
    aucs = [r['auc'] for r in results.values() if r['auc'] is not None]
    aps  = [r['ap']  for r in results.values() if r['ap']  is not None]
    macro['auc'] = np.mean(aucs) if aucs else None
    macro['ap']  = np.mean(aps)  if aps  else None
    results['__macro__'] = macro

    # Rare-only macro (RI, HEM, PVK)
    rare_labels = [c for c in label_cols if c in Config.RARE_LABELS]
    if rare_labels:
        rare_f1s  = [results[c]['f1']  for c in rare_labels]
        rare_aucs = [results[c]['auc'] for c in rare_labels if results[c]['auc'] is not None]
        rare_aps  = [results[c]['ap']  for c in rare_labels if results[c]['ap']  is not None]
        rare_prec = [results[c]['prec'] for c in rare_labels]
        rare_rec  = [results[c]['rec']  for c in rare_labels]
        results['__rare__'] = {
            'f1':   np.mean(rare_f1s),
            'auc':  np.mean(rare_aucs) if rare_aucs else None,
            'ap':   np.mean(rare_aps)  if rare_aps  else None,
            'prec': np.mean(rare_prec),
            'rec':  np.mean(rare_rec),
        }
    return results


# ============================================================================
# TRAINING
# ============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, n_epochs):
    model.train()
    total_loss, n_total = 0, 0
    pbar = tqdm(loader, desc=f"  Train {epoch+1:2d}/{n_epochs}",
                bar_format='{l_bar}{bar:30}{r_bar}', leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        n_total += imgs.size(0)
        pbar.set_postfix(loss=f"{total_loss/n_total:.4f}")
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


# ============================================================================
# ATTENTION VISUALIZATION
# ============================================================================
@torch.no_grad()
def extract_attention_maps(model, loader, device):
    model.eval()
    all_attn, all_logits, all_labels, all_paths = [], [], [], []
    for imgs, labels, paths in tqdm(loader, desc="  Attention extraction",
                                     bar_format='{l_bar}{bar:30}{r_bar}', leave=False):
        imgs = imgs.to(device)
        logits, attn_maps = model(imgs, return_attention=True)
        all_attn.append(attn_maps.cpu())
        all_logits.append(logits.cpu())
        all_labels.append(labels)
        all_paths.extend(paths)
    return torch.cat(all_attn), torch.cat(all_logits), torch.cat(all_labels), all_paths


def save_attention_visualization(img_path, attn_maps, gt_labels, pred_labels,
                                  label_names, save_path, img_size=380):
    img = Image.open(img_path).convert('RGB').resize((img_size, img_size))
    img_np = np.array(img) / 255.0
    n = len(label_names)
    fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))

    gt_str   = " + ".join([short_name(label_names[i]) for i in range(n) if gt_labels[i] == 1])   or "Temiz"
    pred_str = " + ".join([short_name(label_names[i]) for i in range(n) if pred_labels[i] == 1]) or "Temiz"
    match = "✓" if (gt_labels == pred_labels).all() else "✗"
    fig.suptitle(f'GT=[{gt_str}]  Pred=[{pred_str}]  ({match})', fontsize=11, fontweight='bold')

    axes[0].imshow(img_np); axes[0].set_title('Original'); axes[0].axis('off')
    for i in range(n):
        ax = axes[i + 1]
        attn = attn_maps[i].numpy()
        attn_up = np.array(Image.fromarray(attn).resize((img_size, img_size), Image.BILINEAR))
        if attn_up.max() > attn_up.min():
            attn_up = (attn_up - attn_up.min()) / (attn_up.max() - attn_up.min())
        ax.imshow(img_np)
        ax.imshow(attn_up, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        sn = short_name(label_names[i])
        gt_s = "VAR" if gt_labels[i] == 1 else "YOK"
        pred_s = "VAR" if pred_labels[i] == 1 else "YOK"
        color = 'green' if gt_labels[i] == pred_labels[i] else 'red'
        ax.set_title(f'{sn}\nGT={gt_s} P={pred_s}', fontsize=9, color=color, fontweight='bold')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=90, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================
def main():
    cfg = Config()
    results_dir = Path(cfg.RESULTS_DIR)
    results_dir.mkdir(exist_ok=True, parents=True)
    (results_dir / 'attention_vis').mkdir(exist_ok=True)

    device = torch.device('mps'  if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
                     else 'cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("🏥 FUNDUS PATHOLOGY — 8 Label (v6 — RI/HEM/PVK eklendi)")
    print(f"   Model: {cfg.MODEL_NAME} + CBAM + 8 Pathology Attention Heads")
    print(f"   Device: {device}")
    print(f"   Loss: Focal Loss (γ={cfg.FOCAL_GAMMA})")
    print("=" * 80)

    # ── Load data ──
    df = pd.read_csv(cfg.DATASET_CSV, encoding='utf-8')
    df.columns = [c.strip().replace('\xa0', '') for c in df.columns]
    df['patient_id'] = df['Klasör'].astype(str)
    df['image_name'] = df['Dosya ismi'].astype(str)
    df['image_path'] = df.apply(
        lambda r: os.path.join(cfg.DATA_ROOT, r['patient_id'], r['image_name']), axis=1)

    before = len(df)
    df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)
    if len(df) < before:
        print(f"  ⚠️ {before - len(df)} dosya bulunamadı")

    # ★ Eksik label sütunlarını kontrol et ve 0 ile doldur
    for c in cfg.LABEL_COLS:
        if c not in df.columns:
            print(f"  ⚠️ '{c}' sütunu bulunamadı, 0 ile dolduruldu")
            df[c] = 0
        else:
            df[c] = df[c].astype(int)

    # ★ Sütun adlarında whitespace/encoding sorunları için ek kontrol
    col_map = {col.strip().replace('\xa0', ''): col for col in df.columns}
    for label in cfg.LABEL_COLS:
        clean = label.strip().replace('\xa0', '')
        if clean in col_map and col_map[clean] != label:
            df[label] = df[col_map[clean]]

    print(f"\n  Dataset: {len(df)} görüntü, {df['patient_id'].nunique()} hasta")
    print(f"\n  Label dağılımı (8 label):")
    for col in cfg.LABEL_COLS:
        n = int(df[col].sum())
        rare_tag = " ★RARE" if col in cfg.RARE_LABELS else ""
        print(f"    {short_name(col):4s}  {col:35s}: {n:5d} ({n/len(df)*100:5.1f}%){rare_tag}")

    # ── Class weights (rare sınıflar için daha yüksek) ──
    pos_counts = df[cfg.LABEL_COLS].sum().values
    neg_counts = len(df) - pos_counts
    pos_weight = torch.tensor(neg_counts / np.maximum(pos_counts, 1), dtype=torch.float32).to(device)
    pos_weight = torch.clamp(pos_weight, min=1.0, max=50.0)  # Rare sınıflar için max 50
    print(f"\n  Pos weights (Focal Loss alpha):")
    for i, col in enumerate(cfg.LABEL_COLS):
        rare_tag = " ★" if col in cfg.RARE_LABELS else ""
        print(f"    {short_name(col):4s}: {pos_weight[i].item():.2f}{rare_tag}")

    # ── Transforms ──
    train_transform = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ── Balanced folds ──
    fold_splits = list(balanced_stratified_group_kfold_multilabel(
        df, cfg.LABEL_COLS, 'patient_id', cfg.N_FOLDS, cfg.SEED))

    sns = [short_name(c) for c in cfg.LABEL_COLS]
    total_per_label = df[cfg.LABEL_COLS].sum().values
    print(f"\n  ★ Balanced Fold dağılımı:")
    header = f"    Fold |"
    for sn in sns: header += f" {sn:>7s} |"
    header += "  Total"
    print(header); print(f"  {'─'*len(header)}")
    for fi, (_, vi) in enumerate(fold_splits):
        vdf = df.iloc[vi]; row_str = f"     {fi+1}/{cfg.N_FOLDS} |"
        for i, col in enumerate(cfg.LABEL_COLS):
            n = int(vdf[col].sum()); total = int(total_per_label[i])
            pct = n/total*100 if total > 0 else 0
            row_str += f" {n:>3d}({pct:2.0f}%) |"
        row_str += f"  {len(vi):>5d}"; print(row_str)

    # ══════════════════════════════════════════════════════════════════════════
    # TRAINING
    # ══════════════════════════════════════════════════════════════════════════
    all_fold_results = []
    all_val_logits = []
    all_val_labels = []
    all_val_indices = []
    all_val_attn = []
    all_val_paths = []
    all_fold_thresholds = []

    t0 = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"\n{'='*80}")
        print(f"  FOLD {fold_idx+1}/{cfg.N_FOLDS}")
        print(f"{'='*80}")

        train_df = df.iloc[train_idx]; val_df = df.iloc[val_idx]
        val_pos = val_df[cfg.LABEL_COLS].sum()
        pos_str = " | ".join([f"{short_name(c)}:{int(val_pos[c])}" for c in cfg.LABEL_COLS])
        print(f"  Train: {len(train_df)} imgs, {train_df['patient_id'].nunique()} pts")
        print(f"  Val:   {len(val_df)} imgs, {val_df['patient_id'].nunique()} pts")
        print(f"  Val pos: {pos_str}")

        overlap = set(train_df['patient_id']) & set(val_df['patient_id'])
        print(f"  {'⚠️ LEAK!' if overlap else '✓ No leak'}")

        train_ds = FundusDataset(train_df, cfg.LABEL_COLS, train_transform)
        val_ds   = FundusDataset(val_df,   cfg.LABEL_COLS, val_transform)
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                  num_workers=cfg.NUM_WORKERS, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False,
                                  num_workers=cfg.NUM_WORKERS, pin_memory=True)

        model = FundusPathologyAttentionModel(cfg.MODEL_NAME, cfg.N_LABELS).to(device)
        criterion = FocalLossWithLogits(gamma=cfg.FOCAL_GAMMA, pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

        best_macro_f1 = 0
        patience_counter = 0
        best_logits, best_labels = None, None
        best_state = None
        best_thresholds = {c: 0.5 for c in cfg.LABEL_COLS}

        for epoch in range(cfg.EPOCHS):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                         epoch, cfg.EPOCHS)
            val_logits, val_labels = evaluate(model, val_loader, device)
            scheduler.step()

            epoch_thresholds = find_optimal_thresholds(
                val_logits, val_labels, cfg.LABEL_COLS, cfg.THRESHOLD_RANGE)
            epoch_metrics = compute_metrics_with_thresholds(
                val_logits, val_labels, cfg.LABEL_COLS, epoch_thresholds)
            macro_f1 = epoch_metrics['__macro__']['f1']

            # Rare macro for monitoring
            rare_f1 = epoch_metrics.get('__rare__', {}).get('f1', 0)

            label_str = " ".join([
                f"{short_name(c)}={epoch_metrics[c]['f1']:.2f}"
                for c in cfg.LABEL_COLS])

            improved = macro_f1 > best_macro_f1
            if improved:
                best_macro_f1 = macro_f1
                patience_counter = 0
                best_logits = val_logits
                best_labels = val_labels
                best_thresholds = epoch_thresholds
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(model.state_dict(), results_dir / f'model_fold{fold_idx}.pth')
            else:
                patience_counter += 1

            star = " ★" if improved else ""
            print(f"    Ep {epoch+1:2d}/{cfg.EPOCHS}: loss={train_loss:.4f} "
                  f"mF1={macro_f1:.3f} rareF1={rare_f1:.3f} | {label_str}{star}")

            if patience_counter >= cfg.PATIENCE:
                print(f"    ⏹ Early stop @ epoch {epoch+1}")
                break

        # ── Fold results ──
        fold_metrics = compute_metrics_with_thresholds(
            best_logits, best_labels, cfg.LABEL_COLS, best_thresholds)
        macro = fold_metrics['__macro__']
        rare  = fold_metrics.get('__rare__', {})

        all_fold_results.append(fold_metrics)
        all_val_logits.append(best_logits)
        all_val_labels.append(best_labels)
        all_val_indices.extend(val_idx.tolist())
        all_fold_thresholds.append(best_thresholds)

        print(f"\n  ✅ Fold {fold_idx+1}: mF1={macro['f1']:.4f} AUC={macro['auc']:.4f} "
              f"rareF1={rare.get('f1', 0):.4f}")

        print(f"  {'Label':6s} {'t':>5s} {'F1':>6s} {'AUC':>6s} {'Prec':>6s} {'Rec':>6s} {'TP':>4s} {'FP':>4s} {'FN':>4s}")
        print(f"  {'─'*60}")
        for col in cfg.LABEL_COLS:
            m = fold_metrics[col]
            auc_s = f"{m['auc']:.3f}" if m['auc'] else " N/A"
            rare_tag = "★" if col in cfg.RARE_LABELS else " "
            print(f"  {rare_tag}{short_name(col):5s} {m['threshold']:>5.2f} {m['f1']:6.3f} "
                  f"{auc_s:>6s} {m['prec']:6.3f} {m['rec']:6.3f} "
                  f"{m['tp']:4d} {m['fp']:4d} {m['fn']:4d}")

        # ── Attention maps ──
        if cfg.SAVE_ATTENTION_MAPS and best_state:
            print(f"\n  🔍 Attention map extraction (Fold {fold_idx+1})...")
            model.load_state_dict(best_state); model.to(device)
            vis_ds = FundusDatasetWithPath(val_df, cfg.LABEL_COLS, val_transform)
            vis_loader = DataLoader(vis_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                                    num_workers=cfg.NUM_WORKERS)
            fold_attn, fold_logits_vis, fold_labels_vis, fold_paths = extract_attention_maps(
                model, vis_loader, device)

            all_val_attn.append(fold_attn)
            all_val_paths.extend(fold_paths)

            fold_probs = torch.sigmoid(fold_logits_vis).numpy()
            fold_preds = np.zeros_like(fold_probs, dtype=int)
            for i, col in enumerate(cfg.LABEL_COLS):
                fold_preds[:, i] = (fold_probs[:, i] >= best_thresholds[col]).astype(int)
            fold_gt = fold_labels_vis.numpy().astype(int)

            vis_dir = results_dir / 'attention_vis' / f'fold{fold_idx+1}'
            vis_dir.mkdir(exist_ok=True, parents=True)
            n_saved = 0

            for li, col in enumerate(cfg.LABEL_COLS):
                sn = short_name(col)
                tp_mask = (fold_gt[:, li] == 1) & (fold_preds[:, li] == 1)
                for j, idx in enumerate(np.where(tp_mask)[0][:cfg.N_VIS_PER_LABEL]):
                    save_attention_visualization(
                        fold_paths[idx], fold_attn[idx], fold_gt[idx], fold_preds[idx],
                        cfg.LABEL_COLS, vis_dir / f'TP_{sn}_{j+1}.png', cfg.IMG_SIZE)
                    n_saved += 1

                fn_mask = (fold_gt[:, li] == 1) & (fold_preds[:, li] == 0)
                for j, idx in enumerate(np.where(fn_mask)[0][:3]):
                    save_attention_visualization(
                        fold_paths[idx], fold_attn[idx], fold_gt[idx], fold_preds[idx],
                        cfg.LABEL_COLS, vis_dir / f'FN_{sn}_{j+1}.png', cfg.IMG_SIZE)
                    n_saved += 1

            print(f"  💾 {n_saved} attention vis → {vis_dir}/")

    elapsed = (time.time() - t0) / 60

    # ══════════════════════════════════════════════════════════════════════════
    # AGGREGATE + FINAL THRESHOLD OPTIMIZATION
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("📊 FINAL SONUÇLAR — 5-Fold CV")
    print(f"{'='*80}")

    all_logits_cat = torch.cat(all_val_logits)
    all_labels_cat = torch.cat(all_val_labels)

    final_thresholds = find_optimal_thresholds(
        all_logits_cat, all_labels_cat, cfg.LABEL_COLS, cfg.THRESHOLD_RANGE)

    overall = compute_metrics_with_thresholds(
        all_logits_cat, all_labels_cat, cfg.LABEL_COLS, final_thresholds)
    macro = overall['__macro__']
    rare  = overall.get('__rare__', {})

    print(f"\n  Final Thresholds: {final_thresholds}")
    print(f"\n  ── Macro (8 label) ──")
    print(f"  Macro F1:        {macro['f1']:.4f}")
    print(f"  Macro AUC:       {macro['auc']:.4f}")
    print(f"  Macro AP:        {macro['ap']:.4f}")
    print(f"  Macro Precision: {macro['prec']:.4f}")
    print(f"  Macro Recall:    {macro['rec']:.4f}")

    print(f"\n  ── Rare-class macro (RI, HEM, PVK) ──")
    print(f"  Rare F1:         {rare.get('f1', 0):.4f}")
    print(f"  Rare AUC:        {rare.get('auc', 0):.4f}")
    print(f"  Rare Prec:       {rare.get('prec', 0):.4f}")
    print(f"  Rare Rec:        {rare.get('rec', 0):.4f}")

    print(f"\n  {'':1s}{'Patoloji':35s} {'t':>5s} {'N':>5s} {'F1':>7s} {'AUC':>7s} {'AP':>7s} {'Prec':>7s} {'Rec':>7s} {'TP':>4s} {'FP':>4s} {'FN':>4s}")
    print(f"  {'─'*100}")
    for col in cfg.LABEL_COLS:
        m = overall[col]
        auc_s = f"{m['auc']:.4f}" if m['auc'] else "  N/A"
        ap_s  = f"{m['ap']:.4f}"  if m['ap']  else "  N/A"
        rare_tag = "★" if col in cfg.RARE_LABELS else " "
        print(f"  {rare_tag}{col:35s} {m['threshold']:>5.2f} {m['n_pos']:>5d} "
              f"{m['f1']:>7.4f} {auc_s:>7s} {ap_s:>7s} "
              f"{m['prec']:>7.4f} {m['rec']:>7.4f} {m['tp']:>4d} {m['fp']:>4d} {m['fn']:>4d}")

    print(f"\n  {'':36s} {'':>5s} {'':>5s} {macro['f1']:>7.4f} {macro['auc']:>7.4f} {macro['ap']:>7.4f}")
    print(f"  Macro avg")

    # t=0.5 comparison
    baseline = compute_metrics_with_thresholds(
        all_logits_cat, all_labels_cat, cfg.LABEL_COLS, {c: 0.5 for c in cfg.LABEL_COLS})
    base_macro = baseline['__macro__']
    print(f"\n  Threshold opt. gain: mF1 {base_macro['f1']:.4f} → {macro['f1']:.4f} "
          f"({macro['f1']-base_macro['f1']:+.4f})")

    # Per-fold summary
    print(f"\n  Per-fold:")
    for i, fm in enumerate(all_fold_results):
        m = fm['__macro__']; r = fm.get('__rare__', {})
        print(f"    Fold {i+1}: mF1={m['f1']:.4f} AUC={m['auc']:.4f} "
              f"rareF1={r.get('f1',0):.4f}")

    # ── Save ──
    save_results = {
        'pipeline': 'Pathology_8_v6_attention_focal',
        'model': f'{cfg.MODEL_NAME} + CBAM + 8 Attention Heads',
        'n_labels': cfg.N_LABELS, 'label_cols': cfg.LABEL_COLS,
        'rare_labels': cfg.RARE_LABELS,
        'n_images': len(df), 'n_patients': int(df['patient_id'].nunique()),
        'n_folds': cfg.N_FOLDS, 'elapsed_min': round(elapsed, 1),
        'final_thresholds': {k: float(v) for k, v in final_thresholds.items()},
        'macro_f1': round(float(macro['f1']), 4),
        'macro_auc': round(float(macro['auc']), 4) if macro['auc'] else None,
        'macro_ap': round(float(macro['ap']), 4) if macro['ap'] else None,
        'macro_prec': round(float(macro['prec']), 4),
        'macro_rec': round(float(macro['rec']), 4),
        'rare_f1': round(float(rare['f1']), 4) if rare.get('f1') else None,
        'rare_auc': round(float(rare['auc']), 4) if rare.get('auc') else None,
        'per_label': {},
    }
    for col in cfg.LABEL_COLS:
        m = overall[col]
        save_results['per_label'][col] = {
            k: round(float(v), 4) if v is not None and not isinstance(v, int) else v
            for k, v in m.items()
        }

    with open(results_dir / 'summary_pathology_v6.json', 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)

    pred_df = df.iloc[all_val_indices].copy()
    probs_all = torch.sigmoid(all_logits_cat).numpy()
    for i, col in enumerate(cfg.LABEL_COLS):
        pred_df[f'prob_{col}'] = probs_all[:, i]
        pred_df[f'pred_{col}'] = (probs_all[:, i] >= final_thresholds[col]).astype(int)
    pred_df.to_csv(results_dir / 'predictions_pathology_v6.csv', index=False)

    with open(results_dir / 'optimal_thresholds.json', 'w') as f:
        json.dump({k: float(v) for k, v in final_thresholds.items()}, f, indent=2)

    print(f"\n  💾 {results_dir}/")
    print(f"  ⏱ {elapsed:.1f} dk")
    print(f"\n{'='*80}")
    print(f"✅ PATHOLOGY v6 TAMAMLANDI! (8 label: DKS/ODB/VI/MÖ/DDB/RI/HEM/PVK)")
    print(f"   Macro F1: {macro['f1']:.4f}  AUC: {macro['auc']:.4f}  AP: {macro['ap']:.4f}")
    print(f"   Rare F1:  {rare.get('f1',0):.4f}  (RI/HEM/PVK)")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

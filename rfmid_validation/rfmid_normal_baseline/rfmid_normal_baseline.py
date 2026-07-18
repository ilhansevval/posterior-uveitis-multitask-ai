#!/usr/bin/env python3
"""
🌐 RFMiD NORMAL BASELINE (KD'siz) — Aşama 1, 5-Fold
   EfficientNet-B4, 14 eğitilebilir label, 5-Fold CV
   Cerrahpaşa baseline'larıyla AYNI yapı (focal, threshold opt, 5-fold, batch progress)
   Çıktı formatı v8 ile aynı: epoch'ta per-label F1, fold sonunda detaylı tablo,
   final'de GLOBAL tablo + confusion matrix.

   3 RFMiD set'i (train+val+test = ~3170) birleştirilip 5-fold yapılır.
   Amaç: RFMiD'de KD-öncesi referans. Aşama 3 (RFMiD+KD) ile karşılaştırılacak.

   14 eğitilebilir label (train≥30, test≥10):
     DR, MH, ODC, TSLN, DN, ARMD, MYA, BRVO, ODP, ODE*, LS, RS*, CSR, CRS*
     (* = senin overlapping label'ların)
"""

import os, time, random, warnings, json
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score, confusion_matrix
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
    RFMID_ROOT = r'C:\Users\gtu\Downloads\A. RFMiD_All_Classes_Dataset'
    IMG_DIR = os.path.join(RFMID_ROOT, '1. Original Images')
    GT_DIR = os.path.join(RFMID_ROOT, '2. Groundtruths')
    SETS = {
        'train': (os.path.join(IMG_DIR, 'a. Training Set'),
                  os.path.join(GT_DIR, 'a. RFMiD_Training_Labels.csv')),
        'val':   (os.path.join(IMG_DIR, 'b. Validation Set'),
                  os.path.join(GT_DIR, 'b. RFMiD_Validation_Labels.csv')),
        'test':  (os.path.join(IMG_DIR, 'c. Testing Set'),
                  os.path.join(GT_DIR, 'c. RFMiD_Testing_Labels.csv')),
    }
    SAVE_DIR = os.path.join(r'C:\Users\gtu\Documents\cerrahpasa\files\files',
                            'results_rfmid_baseline')

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

    LABEL_COLS = ['DR', 'MH', 'ODC', 'TSLN', 'DN', 'ARMD', 'MYA',
                  'BRVO', 'ODP', 'ODE', 'LS', 'RS', 'CSR', 'CRS']
    N_LABELS = len(LABEL_COLS)
    OVERLAP_LABELS = ['ODE', 'RS', 'CRS']

    FOCAL_GAMMA = 2.0
    THRESHOLD_RANGE = np.arange(0.05, 0.96, 0.02)
    BATCH_LOG_EVERY = 20   # her N batch'te ara satır


def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
seed_everything(Config.SEED)


# ============================================================================
# DATASET
# ============================================================================
class RFMiDDataset(Dataset):
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


# ============================================================================
# FOCAL LOSS
# ============================================================================
class FocalLossWithLogits(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma; self.pos_weight = pos_weight
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = (1 - p_t) ** self.gamma * bce
        if self.pos_weight is not None:
            w = targets * self.pos_weight.unsqueeze(0) + (1 - targets)
            loss = loss * w
        return loss.mean()


# ============================================================================
# MODEL
# ============================================================================
class RFMiDBaseline(nn.Module):
    def __init__(self, n_labels):
        super().__init__()
        import timm
        self.backbone = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)
        n_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(n_features, 512),
            nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(512, n_labels))
    def forward(self, x):
        return self.classifier(self.backbone(x))


# ============================================================================
# IMAGE-LEVEL STRATIFIED K-FOLD (RFMiD'de hasta grubu yok)
# ============================================================================
def stratified_kfold_multilabel(df, label_cols, n_splits, seed=42):
    np.random.seed(seed)
    n = len(df)
    fold_assign = np.full(n, -1, dtype=int)
    counts = sorted([(c, int(df[c].sum())) for c in label_cols], key=lambda x: x[1])
    for col, _ in counts:
        pos = np.where(df[col].values == 1)[0]
        unassigned = pos[fold_assign[pos] == -1]
        if len(unassigned) == 0: continue
        np.random.shuffle(unassigned)
        already = np.zeros(n_splits, dtype=int)
        for i in pos[fold_assign[pos] != -1]:
            already[fold_assign[i]] += 1
        for i in unassigned:
            t = int(np.argmin(already)); fold_assign[i] = t; already[t] += 1
    remaining = np.where(fold_assign == -1)[0]
    np.random.shuffle(remaining)
    sizes = np.array([np.sum(fold_assign == f) for f in range(n_splits)])
    for i in remaining:
        t = int(np.argmin(sizes)); fold_assign[i] = t; sizes[t] += 1
    for f in range(n_splits):
        val_mask = fold_assign == f
        yield np.where(~val_mask)[0], np.where(val_mask)[0]


# ============================================================================
# THRESHOLD + METRICS
# ============================================================================
def find_optimal_thresholds(logits, labels, label_cols, threshold_range):
    probs = torch.sigmoid(logits).numpy()
    y_true = labels.numpy()
    thresholds = {}
    for i, col in enumerate(label_cols):
        gt = y_true[:, i]; best_t, best_f1 = 0.5, 0
        for t in threshold_range:
            pred = (probs[:, i] >= t).astype(int)
            f1 = f1_score(gt, pred, zero_division=0)
            if f1 > best_f1: best_f1 = f1; best_t = t
        thresholds[col] = round(float(best_t), 2)
    return thresholds


def print_detailed_table(results, label_cols, overlap_labels, title=""):
    """v8c tarzı tam tablo: TP/FP/FN/TN/F1/Sens/Spec/Prec/AUC/AP + MACRO + OVERLAP."""
    if title:
        print(f"\n  {title}")
    print(f"  {'Label':6s} {'N':>5s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'TN':>5s} "
          f"{'F1':>7s} {'Sens':>7s} {'Spec':>7s} {'Prec':>7s} {'AUC':>7s} {'AP':>7s}")
    print(f"  {'─'*86}")
    for col in label_cols:
        r = results[col]
        auc_s = f"{r['auc']:.4f}" if r['auc'] is not None else "  N/A"
        ap_s = f"{r['ap']:.4f}" if r['ap'] is not None else "  N/A"
        tag = " 🎯" if col in overlap_labels else ""
        print(f"  {col:6s} {r['n_pos']:5d} {r['tp']:5d} {r['fp']:5d} {r['fn']:5d} {r['tn']:5d} "
              f"{r['f1']:7.4f} {r['sens']:7.4f} {r['spec']:7.4f} {r['prec']:7.4f} "
              f"{auc_s:>7s} {ap_s:>7s}{tag}")
    print(f"  {'─'*86}")
    m = results['__macro__']
    tot_tp = sum(results[c]['tp'] for c in label_cols)
    tot_fp = sum(results[c]['fp'] for c in label_cols)
    tot_fn = sum(results[c]['fn'] for c in label_cols)
    tot_tn = sum(results[c]['tn'] for c in label_cols)
    print(f"  {'MACRO':6s} {'':5s} {tot_tp:5d} {tot_fp:5d} {tot_fn:5d} {tot_tn:5d} "
          f"{m['f1']:7.4f} {m['sens']:7.4f} {m['spec']:7.4f} {m['prec']:7.4f} "
          f"{m['auc']:7.4f} {m['ap']:7.4f}")
    o = results['__overlap__']
    o_auc = f"{o['auc']:.4f}" if o['auc'] is not None else "  N/A"
    o_ap = f"{o['ap']:.4f}" if o['ap'] is not None else "  N/A"
    print(f"  {'OVLAP':6s} {'':5s} {'':5s} {'':5s} {'':5s} {'':5s} "
          f"{o['f1']:7.4f} {'':7s} {'':7s} {'':7s} {o_auc:>7s} {o_ap:>7s}  🎯 ODE/RS/CRS")


def print_confusion_matrices(results, label_cols, overlap_labels):
    """Her label için 2x2 binary confusion matrix (multi-label olduğu için per-label)."""
    print(f"\n  Confusion Matrix (her label, eşik uygulanmış):")
    print(f"  {'Label':6s} {'Pred=0':>14s} {'Pred=1':>14s}")
    print(f"  {'─'*40}")
    for col in label_cols:
        r = results[col]
        tag = " 🎯" if col in overlap_labels else ""
        print(f"  {col:6s}{tag}")
        print(f"    {'GT=0':6s} TN={r['tn']:5d}      FP={r['fp']:5d}")
        print(f"    {'GT=1':6s} FN={r['fn']:5d}      TP={r['tp']:5d}")


def print_epoch_label_f1(results, label_cols, overlap_labels):
    """Epoch sonunda v8 tarzı per-label F1 satırları (7+7 split)."""
    f1_strs = []
    for c in label_cols:
        tag = "🎯" if c in overlap_labels else ""
        f1_strs.append(f"{c}{tag}={results[c]['f1']:.2f}")
    print(f"         {' '.join(f1_strs[:7])}")
    print(f"         {' '.join(f1_strs[7:])}")


def compute_metrics(logits, labels, label_cols, thresholds, overlap_labels):
    probs = torch.sigmoid(logits).numpy()
    y_true = labels.numpy()
    results = {}
    for i, col in enumerate(label_cols):
        gt = y_true[:, i]; t = thresholds.get(col, 0.5)
        pr = (probs[:, i] >= t).astype(int); pb = probs[:, i]
        n_pos = int(gt.sum())
        auc = roc_auc_score(gt, pb) if 0 < n_pos < len(gt) else None
        ap = average_precision_score(gt, pb) if n_pos > 0 else None
        cm = confusion_matrix(gt, pr, labels=[0, 1])
        tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # recall/sensitivity
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # specificity
        results[col] = {
            'n_pos': n_pos, 'threshold': t, 'auc': auc, 'ap': ap,
            'f1': f1_score(gt, pr, zero_division=0),
            'prec': precision_score(gt, pr, zero_division=0),
            'rec': recall_score(gt, pr, zero_division=0),
            'sens': sens, 'spec': spec,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        }
    macro = {k: np.mean([r[k] for r in results.values() if r.get(k) is not None])
             for k in ['f1','prec','rec','sens','spec']}
    aucs = [r['auc'] for r in results.values() if r['auc'] is not None]
    aps  = [r['ap']  for r in results.values() if r['ap']  is not None]
    macro['auc'] = np.mean(aucs) if aucs else None
    macro['ap']  = np.mean(aps)  if aps  else None
    results['__macro__'] = macro
    ov_f1 = [results[c]['f1'] for c in overlap_labels if c in results]
    ov_auc = [results[c]['auc'] for c in overlap_labels if c in results and results[c]['auc'] is not None]
    ov_ap = [results[c]['ap'] for c in overlap_labels if c in results and results[c]['ap'] is not None]
    results['__overlap__'] = {
        'f1': np.mean(ov_f1) if ov_f1 else None,
        'auc': np.mean(ov_auc) if ov_auc else None,
        'ap': np.mean(ov_ap) if ov_ap else None,
    }
    return results


# ============================================================================
# TRAIN (batch progress yazılı) / EVAL
# ============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, n_epochs, log_every):
    model.train()
    total_loss, n_seen = 0, 0
    n_batches = len(loader)
    for bi, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward(); optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        n_seen += imgs.size(0)
        if (bi + 1) % log_every == 0 or (bi + 1) == n_batches:
            print(f"    Ep {epoch+1:2d}/{n_epochs} batch {bi+1:4d}/{n_batches} "
                  f"| loss={total_loss/n_seen:.4f}     ", end='\r')
    print()
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        all_logits.append(model(imgs).cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


# ============================================================================
# LOAD RFMiD (3 set birleşik)
# ============================================================================
def load_rfmid_all(cfg):
    parts = []
    for split, (img_dir, csv_path) in cfg.SETS.items():
        df = pd.read_csv(csv_path)
        df['image_path'] = df['ID'].apply(lambda i: os.path.join(img_dir, f"{int(i)}.png"))
        df = df[df['image_path'].apply(os.path.exists)]
        parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    for c in cfg.LABEL_COLS:
        full[c] = full[c].astype(int)
    return full


# ============================================================================
# MAIN
# ============================================================================
def main():
    cfg = Config()
    results_dir = Path(cfg.SAVE_DIR)
    results_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("🌐 RFMiD NORMAL BASELINE (KD'siz) — Aşama 1, 5-Fold")
    print(f"   Model: {cfg.MODEL_NAME} | Device: {device} | {cfg.N_LABELS} label")
    print("=" * 80)

    df = load_rfmid_all(cfg)
    print(f"\n  Toplam görüntü (3 set birleşik): {len(df)}")
    print(f"\n  Label dağılımı:")
    for col in cfg.LABEL_COLS:
        n = int(df[col].sum())
        tag = " 🎯" if col in cfg.OVERLAP_LABELS else ""
        print(f"    {col:6s}: {n:5d} ({n/len(df)*100:4.1f}%){tag}")

    pos = df[cfg.LABEL_COLS].sum().values
    neg = len(df) - pos
    pos_weight = torch.tensor(neg / np.maximum(pos, 1), dtype=torch.float32)
    pos_weight = torch.clamp(pos_weight, min=1.0, max=50.0).to(device)

    train_tf = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    fold_splits = list(stratified_kfold_multilabel(df, cfg.LABEL_COLS, cfg.N_FOLDS, cfg.SEED))

    print(f"\n  Fold dağılımı (overlapping val pozitifleri):")
    for fi, (_, vi) in enumerate(fold_splits):
        vdf = df.iloc[vi]
        ov = " ".join([f"{c}:{int(vdf[c].sum())}" for c in cfg.OVERLAP_LABELS])
        print(f"    Fold {fi+1}: {len(vi):4d} val | {ov}")

    all_val_logits, all_val_labels, all_val_indices = [], [], []
    all_fold_results = []
    t0 = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"\n{'='*80}\n  FOLD {fold_idx+1}/{cfg.N_FOLDS}\n{'='*80}")
        train_df = df.iloc[train_idx]; val_df = df.iloc[val_idx]
        print(f"  Train: {len(train_df)} | Val: {len(val_df)}")

        train_ds = RFMiDDataset(train_df, cfg.LABEL_COLS, train_tf)
        val_ds = RFMiDDataset(val_df, cfg.LABEL_COLS, val_tf)
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                  num_workers=cfg.NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                                num_workers=cfg.NUM_WORKERS, pin_memory=True)

        model = RFMiDBaseline(cfg.N_LABELS).to(device)
        criterion = FocalLossWithLogits(gamma=cfg.FOCAL_GAMMA, pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

        best_f1 = 0; patience_counter = 0
        best_logits, best_labels, best_thr = None, None, None

        for epoch in range(cfg.EPOCHS):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                         device, epoch, cfg.EPOCHS, cfg.BATCH_LOG_EVERY)
            val_logits, val_labels = evaluate(model, val_loader, device)
            scheduler.step()

            thr = find_optimal_thresholds(val_logits, val_labels, cfg.LABEL_COLS, cfg.THRESHOLD_RANGE)
            m = compute_metrics(val_logits, val_labels, cfg.LABEL_COLS, thr, cfg.OVERLAP_LABELS)
            mf1 = m['__macro__']['f1']; ovf1 = m['__overlap__']['f1']

            improved = mf1 > best_f1
            if improved:
                best_f1 = mf1; patience_counter = 0
                best_logits = val_logits; best_labels = val_labels; best_thr = thr
                torch.save(model.state_dict(), results_dir / f'rfmid_fold{fold_idx}.pth')
            else:
                patience_counter += 1
            star = " ★" if improved else ""
            print(f"  Ep {epoch+1:2d}/{cfg.EPOCHS}: loss={train_loss:.4f} "
                  f"mF1={mf1:.3f} overlapF1={ovf1:.3f}{star}")
            print_epoch_label_f1(m, cfg.LABEL_COLS, cfg.OVERLAP_LABELS)
            if patience_counter >= cfg.PATIENCE:
                print(f"    ⏹ Early stop @ epoch {epoch+1}")
                break

        # ── FOLD SONU: v8 tarzı detaylı tablo ──
        fm = compute_metrics(best_logits, best_labels, cfg.LABEL_COLS, best_thr, cfg.OVERLAP_LABELS)
        macro = fm['__macro__']; overlap = fm['__overlap__']
        all_fold_results.append(fm)
        all_val_logits.append(best_logits); all_val_labels.append(best_labels)
        all_val_indices.extend(val_idx.tolist())

        print_detailed_table(fm, cfg.LABEL_COLS, cfg.OVERLAP_LABELS,
                             title=f"✅ FOLD {fold_idx+1} RESULTS")

    elapsed = (time.time() - t0) / 60

    # ── FINAL: tüm fold val tahminleri birleşik (pooled) ──
    print(f"\n{'='*80}\n📊 GLOBAL RESULTS — RFMiD Normal Baseline (5-Fold pooled)\n{'='*80}")
    all_logits_cat = torch.cat(all_val_logits)
    all_labels_cat = torch.cat(all_val_labels)
    final_thr = find_optimal_thresholds(all_logits_cat, all_labels_cat, cfg.LABEL_COLS, cfg.THRESHOLD_RANGE)
    overall = compute_metrics(all_logits_cat, all_labels_cat, cfg.LABEL_COLS, final_thr, cfg.OVERLAP_LABELS)
    macro = overall['__macro__']; overlap = overall['__overlap__']

    print_detailed_table(overall, cfg.LABEL_COLS, cfg.OVERLAP_LABELS,
                         title="GLOBAL METRICS")
    print_confusion_matrices(overall, cfg.LABEL_COLS, cfg.OVERLAP_LABELS)

    print(f"\n  Per-fold özet:")
    for i, fm in enumerate(all_fold_results):
        m = fm['__macro__']; o = fm['__overlap__']
        print(f"    Fold {i+1}: mF1={m['f1']:.4f} AUC={m['auc']:.4f} overlapF1={o['f1']:.4f}")

    save = {
        'stage': 'RFMiD Normal Baseline (no KD), 5-fold',
        'model': cfg.MODEL_NAME, 'n_labels': cfg.N_LABELS, 'n_images': len(df),
        'macro_14': {'f1': float(macro['f1']), 'auc': float(macro['auc']), 'ap': float(macro['ap'])},
        'overlap_3': {'f1': float(overlap['f1']), 'auc': float(overlap['auc']), 'ap': float(overlap['ap'])},
        'thresholds': final_thr,
        'per_label': {col: {k: (float(v) if v is not None and not isinstance(v, int) else v)
                            for k, v in overall[col].items()} for col in cfg.LABEL_COLS},
        'per_fold': [{'macro_f1': float(fm['__macro__']['f1']),
                      'overlap_f1': float(fm['__overlap__']['f1'])} for fm in all_fold_results],
        'elapsed_min': round(elapsed, 1),
    }
    with open(results_dir / 'rfmid_baseline_summary.json', 'w', encoding='utf-8') as f:
        json.dump(save, f, indent=2, ensure_ascii=False)

    print(f"\n  💾 {results_dir}/")
    print(f"  ⏱ {elapsed:.1f} dk")
    print(f"\n{'='*80}")
    print(f"✅ RFMiD NORMAL BASELINE (5-Fold) TAMAMLANDI")
    print(f"   Macro F1 (14): {macro['f1']:.4f} | Overlap F1 (3): {overlap['f1']:.4f}")
    print(f"   → Aşama 3 (RFMiD+KD) ile karşılaştırılacak")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

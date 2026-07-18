#!/usr/bin/env python3
"""
🏥 FUNDUS PATHOLOGY — ResNet-50 BASELINE (KD'siz)
   End-to-end fine-tune, 5-Fold CV
   Aynı split, aynı 8 label, aynı threshold optimization, aynı metrics
   Amaç: KD framework karşılaştırması için saf CNN baseline
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
    RESULTS_DIR = os.path.join(DATA_ROOT, 'results_resnet50_baseline')

    MODEL_NAME = 'resnet50'
    IMG_SIZE = 380
    BATCH_SIZE = 8
    NUM_WORKERS = 0
    EPOCHS = 30
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10
    N_FOLDS = 5
    SEED = 42

    LABEL_COLS = [
        'Diffüz kapiller sızıntı',    # DKS
        'Optik disk boyanması',        # ODB
        'Vitreus inflamasyonu',        # VI
        'Makula ödemi',                # MÖ
        'Damar duvar boyanması',       # DDB
        'Retinal infiltrat',           # RI
        'Hemoraji',                    # HEM
        'Perivasküler kılıflanma',     # PVK
    ]
    N_LABELS = len(LABEL_COLS)
    RARE_LABELS = ['Retinal infiltrat', 'Hemoraji', 'Perivasküler kılıflanma']
    FOCAL_GAMMA = 2.0
    THRESHOLD_RANGE = np.arange(0.10, 0.91, 0.02)


SHORT_NAMES = {
    'Diffüz kapiller sızıntı': 'DKS', 'Optik disk boyanması': 'ODB',
    'Vitreus inflamasyonu': 'VI', 'Makula ödemi': 'MÖ',
    'Damar duvar boyanması': 'DDB', 'Retinal infiltrat': 'RI',
    'Hemoraji': 'HEM', 'Perivasküler kılıflanma': 'PVK',
}
def short_name(c): return SHORT_NAMES.get(c, c[:3])


def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
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


# ============================================================================
# FOCAL LOSS (v6 ile AYNI)
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
# MODEL — ResNet-50 (vanilla, düz classifier)
# ============================================================================
class ResNet50Baseline(nn.Module):
    def __init__(self, n_labels=8):
        super().__init__()
        import timm
        self.backbone = timm.create_model('resnet50', pretrained=True, num_classes=0)
        n_features = self.backbone.num_features  # 2048
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(n_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, n_labels),
        )
    def forward(self, x):
        return self.classifier(self.backbone(x))


# ============================================================================
# BALANCED STRATIFIED GROUP K-FOLD (v6 ile AYNI)
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
# THRESHOLD + METRICS (v6 ile AYNI)
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
        thresholds[col] = round(float(best_t), 2)
    return thresholds


def compute_metrics(logits, labels, label_cols, thresholds):
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
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        }
    macro = {k: np.mean([r[k] for r in results.values() if r.get(k) is not None])
             for k in ['f1','prec','rec','acc']}
    aucs = [r['auc'] for r in results.values() if r['auc'] is not None]
    aps  = [r['ap']  for r in results.values() if r['ap']  is not None]
    macro['auc'] = np.mean(aucs) if aucs else None
    macro['ap']  = np.mean(aps)  if aps  else None
    results['__macro__'] = macro
    rare_labels = [c for c in label_cols if c in Config.RARE_LABELS]
    rare_f1s = [results[c]['f1'] for c in rare_labels]
    rare_aucs = [results[c]['auc'] for c in rare_labels if results[c]['auc'] is not None]
    results['__rare__'] = {
        'f1': np.mean(rare_f1s),
        'auc': np.mean(rare_aucs) if rare_aucs else None,
        'prec': np.mean([results[c]['prec'] for c in rare_labels]),
        'rec': np.mean([results[c]['rec'] for c in rare_labels]),
    }
    return results


# ============================================================================
# TRAIN / EVAL
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
# MAIN
# ============================================================================
def main():
    cfg = Config()
    results_dir = Path(cfg.RESULTS_DIR)
    results_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device('cuda' if torch.cuda.is_available() else
                          ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
                           else 'cpu'))

    print("=" * 80)
    print("🏥 ResNet-50 BASELINE (KD'siz) — 8 Label")
    print(f"   Device: {device} | Loss: Focal (γ={cfg.FOCAL_GAMMA})")
    print("=" * 80)

    # ── Load data ──
    df = pd.read_csv(cfg.DATASET_CSV, encoding='utf-8')
    df.columns = [c.strip().replace('\xa0', '') for c in df.columns]
    df['patient_id'] = df['Klasör'].astype(str)
    df['image_name'] = df['Dosya ismi'].astype(str)
    df['image_path'] = df.apply(
        lambda r: os.path.join(cfg.DATA_ROOT, r['patient_id'], r['image_name']), axis=1)
    df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)
    for c in cfg.LABEL_COLS:
        if c not in df.columns: df[c] = 0
        df[c] = df[c].astype(int)

    print(f"\n  Dataset: {len(df)} görüntü, {df['patient_id'].nunique()} hasta")

    # Class weights
    pos_counts = df[cfg.LABEL_COLS].sum().values
    neg_counts = len(df) - pos_counts
    pos_weight = torch.tensor(neg_counts / np.maximum(pos_counts, 1), dtype=torch.float32).to(device)
    pos_weight = torch.clamp(pos_weight, min=1.0, max=50.0)

    # Transforms
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

    fold_splits = list(balanced_stratified_group_kfold_multilabel(
        df, cfg.LABEL_COLS, 'patient_id', cfg.N_FOLDS, cfg.SEED))

    all_val_logits, all_val_labels, all_val_indices = [], [], []
    all_fold_results = []
    t0 = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"\n{'='*80}\n  FOLD {fold_idx+1}/{cfg.N_FOLDS}\n{'='*80}")
        train_df = df.iloc[train_idx]; val_df = df.iloc[val_idx]
        print(f"  Train: {len(train_df)} imgs, {train_df['patient_id'].nunique()} pts")
        print(f"  Val:   {len(val_df)} imgs, {val_df['patient_id'].nunique()} pts")
        overlap = set(train_df['patient_id']) & set(val_df['patient_id'])
        print(f"  {'⚠️ LEAK!' if overlap else '✓ No leak'}")

        train_ds = FundusDataset(train_df, cfg.LABEL_COLS, train_transform)
        val_ds = FundusDataset(val_df, cfg.LABEL_COLS, val_transform)
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                  num_workers=cfg.NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                                num_workers=cfg.NUM_WORKERS, pin_memory=True)

        model = ResNet50Baseline(cfg.N_LABELS).to(device)
        criterion = FocalLossWithLogits(gamma=cfg.FOCAL_GAMMA, pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

        best_macro_f1 = 0; patience_counter = 0
        best_logits, best_labels, best_thresholds = None, None, None

        for epoch in range(cfg.EPOCHS):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, cfg.EPOCHS)
            val_logits, val_labels = evaluate(model, val_loader, device)
            scheduler.step()

            epoch_thr = find_optimal_thresholds(val_logits, val_labels, cfg.LABEL_COLS, cfg.THRESHOLD_RANGE)
            epoch_metrics = compute_metrics(val_logits, val_labels, cfg.LABEL_COLS, epoch_thr)
            macro_f1 = epoch_metrics['__macro__']['f1']
            rare_f1 = epoch_metrics['__rare__']['f1']

            label_str = " ".join([f"{short_name(c)}={epoch_metrics[c]['f1']:.2f}" for c in cfg.LABEL_COLS])
            improved = macro_f1 > best_macro_f1
            if improved:
                best_macro_f1 = macro_f1; patience_counter = 0
                best_logits = val_logits; best_labels = val_labels; best_thresholds = epoch_thr
                torch.save(model.state_dict(), results_dir / f'model_fold{fold_idx}.pth')
            else:
                patience_counter += 1
            star = " ★" if improved else ""
            print(f"    Ep {epoch+1:2d}/{cfg.EPOCHS}: loss={train_loss:.4f} "
                  f"mF1={macro_f1:.3f} rareF1={rare_f1:.3f} | {label_str}{star}")
            if patience_counter >= cfg.PATIENCE:
                print(f"    ⏹ Early stop @ epoch {epoch+1}")
                break

        fold_metrics = compute_metrics(best_logits, best_labels, cfg.LABEL_COLS, best_thresholds)
        macro = fold_metrics['__macro__']; rare = fold_metrics['__rare__']
        all_fold_results.append(fold_metrics)
        all_val_logits.append(best_logits); all_val_labels.append(best_labels)
        all_val_indices.extend(val_idx.tolist())

        print(f"\n  ✅ Fold {fold_idx+1}: mF1={macro['f1']:.4f} AUC={macro['auc']:.4f} rareF1={rare['f1']:.4f}")

    elapsed = (time.time() - t0) / 60

    # ── Final ──
    print(f"\n{'='*80}\n📊 FINAL SONUÇLAR — ResNet-50 5-Fold CV\n{'='*80}")
    all_logits_cat = torch.cat(all_val_logits)
    all_labels_cat = torch.cat(all_val_labels)
    final_thr = find_optimal_thresholds(all_logits_cat, all_labels_cat, cfg.LABEL_COLS, cfg.THRESHOLD_RANGE)
    overall = compute_metrics(all_logits_cat, all_labels_cat, cfg.LABEL_COLS, final_thr)
    macro = overall['__macro__']; rare = overall['__rare__']

    print(f"\n  {'Patoloji':28s} {'N':>4s} {'F1':>7s} {'AUC':>7s} {'AP':>7s} {'Prec':>7s} {'Rec':>7s}")
    print(f"  {'─'*70}")
    for col in cfg.LABEL_COLS:
        m = overall[col]
        auc_s = f"{m['auc']:.4f}" if m['auc'] else "  N/A"
        ap_s = f"{m['ap']:.4f}" if m['ap'] else "  N/A"
        tag = "★" if col in cfg.RARE_LABELS else " "
        print(f"  {tag}{col:27s} {m['n_pos']:>4d} {m['f1']:>7.4f} {auc_s:>7s} {ap_s:>7s} "
              f"{m['prec']:>7.4f} {m['rec']:>7.4f}")
    print(f"  {'─'*70}")
    print(f"  {'MACRO':28s} {'':>4s} {macro['f1']:>7.4f} {macro['auc']:>7.4f} "
          f"{macro['ap']:>7.4f} {macro['prec']:>7.4f} {macro['rec']:>7.4f}")
    print(f"  {'RARE (RI/HEM/PVK)':28s} {'':>4s} {rare['f1']:>7.4f} {rare['auc']:>7.4f}")

    print(f"\n  Per-fold:")
    for i, fm in enumerate(all_fold_results):
        m = fm['__macro__']; r = fm['__rare__']
        print(f"    Fold {i+1}: mF1={m['f1']:.4f} AUC={m['auc']:.4f} rareF1={r['f1']:.4f}")

    print(f"\n  ── KARŞILAŞTIRMA ──")
    print(f"  {'ResNet-50 (noKD)':18s} mF1={macro['f1']:.4f}  rareF1={rare['f1']:.4f}")
    print(f"  {'EffNet+CBAM(noKD)':18s} mF1=0.4323  rareF1=0.2818")
    print(f"  {'KD only (v3)':18s} mF1=0.7950")
    print(f"  {'KD+Pseudo (v8c)':18s} mF1=0.9420  rareF1=0.9620")

    # Save
    save = {
        'model': 'ResNet-50 (no KD)', 'macro_f1': float(macro['f1']),
        'macro_auc': float(macro['auc']), 'macro_ap': float(macro['ap']),
        'rare_f1': float(rare['f1']), 'rare_auc': float(rare['auc']),
        'thresholds': final_thr,
        'per_label': {col: {k: (float(v) if v is not None and not isinstance(v, int) else v)
                            for k, v in overall[col].items()} for col in cfg.LABEL_COLS},
        'per_fold': [{'macro_f1': float(fm['__macro__']['f1']),
                      'rare_f1': float(fm['__rare__']['f1'])} for fm in all_fold_results],
    }
    with open(results_dir / 'resnet50_summary.json', 'w', encoding='utf-8') as f:
        json.dump(save, f, indent=2, ensure_ascii=False)

    pred_df = df.iloc[all_val_indices].copy()
    probs_all = torch.sigmoid(all_logits_cat).numpy()
    for i, col in enumerate(cfg.LABEL_COLS):
        pred_df[f'prob_{col}'] = probs_all[:, i]
        pred_df[f'pred_{col}'] = (probs_all[:, i] >= final_thr[col]).astype(int)
    pred_df.to_csv(results_dir / 'predictions_resnet50.csv', index=False)

    print(f"\n  💾 {results_dir}/")
    print(f"  ⏱ {elapsed:.1f} dk")
    print(f"\n{'='*80}\n✅ ResNet-50 BASELINE TAMAMLANDI — mF1={macro['f1']:.4f}\n{'='*80}")


if __name__ == '__main__':
    main()

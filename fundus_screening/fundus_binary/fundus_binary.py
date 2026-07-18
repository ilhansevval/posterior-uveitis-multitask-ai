#!/usr/bin/env python3
"""
🏥 FUNDUS BINARY CLASSIFICATION — Normal vs Anormal (v3)
   Yeni veri: 1701 görüntü, 95 hasta
   Normal (Göz normal=1): 996 (58.6%)
   Anormal (Göz normal=0): 705 (41.4%)
     - Bilinen patoloji: 560
     - Others (doktor yorumu): 130
     - Others + patoloji: 41  
     - Belirsiz (label yok): 15
   
   Tümü anormal olarak eğitilir. Others ayrıca patoloji olarak sınıflandırılacak.
   EfficientNet-B4 + BCEWithLogitsLoss + Patient-level 5-Fold CV
"""

import os, time, random, warnings, json
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import GroupKFold
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
    DATA_ROOT = '/Users/sevvalilhan/Downloads/files'
    DATASET_CSV = os.path.join(DATA_ROOT, 'Resimler_birlesik_TF_plusYorum_clean_numeric.csv')
    RESULTS_DIR = os.path.join(DATA_ROOT, 'results_binary_v3')

    MODEL_NAME = 'efficientnet_b4'
    IMG_SIZE = 380
    BATCH_SIZE = 16
    NUM_WORKERS = 0
    EPOCHS = 30
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE = 7
    N_FOLDS = 5
    SEED = 42


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
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(row['is_abnormal'], dtype=torch.float32)
        return img, label


# ============================================================================
# MODEL
# ============================================================================
class FundusBinaryModel(nn.Module):
    def __init__(self, model_name='efficientnet_b4', dropout=0.4):
        super().__init__()
        import timm
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        n_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(-1)


# ============================================================================
# TRAINING
# ============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, n_epochs):
    model.train()
    total_loss = 0
    n_correct = 0
    n_total = 0

    pbar = tqdm(loader, desc=f"  Train {epoch+1:2d}/{n_epochs}",
                bar_format='{l_bar}{bar:30}{r_bar}', leave=False)

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = imgs.size(0)
        total_loss += loss.item() * batch_size
        preds = (torch.sigmoid(logits) >= 0.5).float()
        n_correct += (preds == labels).sum().item()
        n_total += batch_size

        pbar.set_postfix(loss=f"{total_loss/n_total:.4f}", acc=f"{n_correct/n_total:.3f}")

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_logits, all_labels = [], []

    pbar = tqdm(loader, desc="  Val",
                bar_format='{l_bar}{bar:30}{r_bar}', leave=False)

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    return total_loss / len(loader.dataset), all_logits, all_labels


# ============================================================================
# MAIN
# ============================================================================
def main():
    cfg = Config()
    results_dir = Path(cfg.RESULTS_DIR)
    results_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("🏥 FUNDUS BINARY CLASSIFICATION — Normal vs Anormal (v3)")
    print(f"   Model: {cfg.MODEL_NAME} | Device: {device}")
    print(f"   Epochs: {cfg.EPOCHS} | Folds: {cfg.N_FOLDS}")
    print("=" * 80)

    # ── Load CSV ──
    df = pd.read_csv(cfg.DATASET_CSV, encoding='utf-8')
    df.columns = [c.strip().replace('\xa0', '') for c in df.columns]

    # Build image_path
    df['patient_id'] = df['Klasör'].astype(str)
    df['image_name'] = df['Dosya ismi'].astype(str)
    df['image_path'] = df.apply(
        lambda r: os.path.join(cfg.DATA_ROOT, r['patient_id'], r['image_name']), axis=1)

    # Check files exist
    before = len(df)
    df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"  ⚠️ {dropped} görüntü dosyası bulunamadı → çıkarıldı")

    # Target: is_abnormal = 1 - Göz normal mi?
    df['is_abnormal'] = (1 - df['Göz normal mi?'].astype(int)).astype(int)

    # Sub-group labels for tracking
    patho_cols = ['Diffüz kapiller sızıntı', 'Optik disk boyanması', 'Vitreus inflamasyonu',
                  'Makula ödemi', 'Damar duvar boyanması', 'Retinal infiltrat',
                  'Perivasküler kılıflanma', 'Hemoraji', 'Retina sinir lif defekti',
                  'Ghost vessel']
    df['has_patho'] = (df[patho_cols].sum(axis=1) > 0).astype(int)
    df['has_others'] = df['Others'].astype(int)

    # 4 alt-grup
    df['subgroup'] = 'normal'
    df.loc[(df['is_abnormal'] == 1) & (df['has_patho'] == 1) & (df['has_others'] == 0), 'subgroup'] = 'patolojik'
    df.loc[(df['is_abnormal'] == 1) & (df['has_patho'] == 1) & (df['has_others'] == 1), 'subgroup'] = 'patolojik+others'
    df.loc[(df['is_abnormal'] == 1) & (df['has_patho'] == 0) & (df['has_others'] == 1), 'subgroup'] = 'others_only'
    df.loc[(df['is_abnormal'] == 1) & (df['has_patho'] == 0) & (df['has_others'] == 0), 'subgroup'] = 'belirsiz'

    print(f"\n  Dataset: {len(df)} görüntü, {df['patient_id'].nunique()} hasta")
    print(f"\n  Sınıf dağılımı:")
    n_normal = (df['is_abnormal'] == 0).sum()
    n_abnormal = (df['is_abnormal'] == 1).sum()
    print(f"    Normal  : {n_normal:5d} ({n_normal/len(df)*100:.1f}%)")
    print(f"    Anormal : {n_abnormal:5d} ({n_abnormal/len(df)*100:.1f}%)")

    print(f"\n  Anormal alt-gruplar:")
    for grp in ['patolojik', 'patolojik+others', 'others_only', 'belirsiz']:
        n = (df['subgroup'] == grp).sum()
        print(f"    {grp:25s}: {n:5d} ({n/len(df)*100:.1f}%)")

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

    # Pos weight
    pos_weight = torch.tensor([n_normal / n_abnormal], dtype=torch.float32).to(device)
    print(f"\n  Pos weight: {pos_weight.item():.2f} (normal={n_normal}, anormal={n_abnormal})")

    # Patient-level GroupKFold
    gkf = GroupKFold(n_splits=cfg.N_FOLDS)
    groups = df['patient_id'].values

    all_fold_results = []
    all_val_logits = []
    all_val_labels = []
    all_val_indices = []
    all_val_subgroups = []

    t0 = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups)):
        print(f"\n{'='*80}")
        print(f"  FOLD {fold_idx+1}/{cfg.N_FOLDS}")
        print(f"{'='*80}")

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        print(f"  Train: {len(train_df)} imgs, {train_df['patient_id'].nunique()} pts")
        print(f"  Val:   {len(val_df)} imgs, {val_df['patient_id'].nunique()} pts")
        v_normal = (val_df['is_abnormal'] == 0).sum()
        v_anormal = (val_df['is_abnormal'] == 1).sum()
        print(f"  Val: Normal={v_normal} Anormal={v_anormal}")
        # Alt-grup bilgisi (sadece sayı)
        sg_counts = val_df['subgroup'].value_counts().to_dict()
        sg_info = ", ".join([f"{k}={v}" for k, v in sorted(sg_counts.items())])
        print(f"  Val alt-gruplar: {sg_info}")

        train_ds = FundusDataset(train_df, train_transform)
        val_ds = FundusDataset(val_df, val_transform)
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                  num_workers=cfg.NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                                num_workers=cfg.NUM_WORKERS, pin_memory=True)

        model = FundusBinaryModel(cfg.MODEL_NAME).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

        best_val_loss = float('inf')
        patience_counter = 0
        best_logits, best_labels = None, None

        for epoch in range(cfg.EPOCHS):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                         epoch, cfg.EPOCHS)
            val_loss, val_logits, val_labels = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            probs_ep = torch.sigmoid(val_logits).numpy()
            y_true_ep = val_labels.numpy()
            y_pred_ep = (probs_ep >= 0.5).astype(int)
            acc = accuracy_score(y_true_ep, y_pred_ep)
            f1 = f1_score(y_true_ep, y_pred_ep, zero_division=0)
            auc = roc_auc_score(y_true_ep, probs_ep) if len(np.unique(y_true_ep)) > 1 else 0

            # Per-class accuracy (Normal vs Anormal only)
            norm_mask = y_true_ep == 0
            abn_mask = y_true_ep == 1
            acc_norm = accuracy_score(y_true_ep[norm_mask], y_pred_ep[norm_mask]) if norm_mask.sum() > 0 else 0
            acc_abn = accuracy_score(y_true_ep[abn_mask], y_pred_ep[abn_mask]) if abn_mask.sum() > 0 else 0

            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                patience_counter = 0
                best_logits = val_logits
                best_labels = val_labels
                torch.save(model.state_dict(), results_dir / f'model_fold{fold_idx}.pth')
            else:
                patience_counter += 1

            star = " ★" if improved else ""
            print(f"    Ep {epoch+1:2d}/{cfg.EPOCHS}: "
                  f"loss={train_loss:.4f}/{val_loss:.4f} "
                  f"acc={acc:.3f} f1={f1:.3f} auc={auc:.3f} | "
                  f"Normal={acc_norm:.2f} Anormal={acc_abn:.2f}{star}")

            if patience_counter >= cfg.PATIENCE:
                print(f"    ⏹ Early stop @ epoch {epoch+1}")
                break

        # Fold metrics
        probs_f = torch.sigmoid(best_logits).numpy()
        y_true_f = best_labels.numpy()
        y_pred_f = (probs_f >= 0.5).astype(int)
        auc_f = roc_auc_score(y_true_f, probs_f)
        f1_f = f1_score(y_true_f, y_pred_f)
        acc_f = accuracy_score(y_true_f, y_pred_f)
        prec_f = precision_score(y_true_f, y_pred_f)
        rec_f = recall_score(y_true_f, y_pred_f)
        cm_f = confusion_matrix(y_true_f, y_pred_f)

        all_fold_results.append({'auc': auc_f, 'f1': f1_f, 'acc': acc_f,
                                 'prec': prec_f, 'rec': rec_f})
        all_val_logits.append(best_logits)
        all_val_labels.append(best_labels)
        all_val_indices.extend(val_idx.tolist())
        all_val_subgroups.extend(val_df['subgroup'].values.tolist())

        print(f"\n  ✅ Fold {fold_idx+1}: AUC={auc_f:.4f}  F1={f1_f:.4f}  Acc={acc_f:.4f}")
        print(f"     Precision={prec_f:.4f}  Recall={rec_f:.4f}")
        print(f"     Normal acc:  {cm_f[0,0]}/{cm_f[0,0]+cm_f[0,1]} ({cm_f[0,0]/(cm_f[0,0]+cm_f[0,1])*100:.1f}%)")
        print(f"     Anormal acc: {cm_f[1,1]}/{cm_f[1,0]+cm_f[1,1]} ({cm_f[1,1]/(cm_f[1,0]+cm_f[1,1])*100:.1f}%)")
        # Alt-grup sayıları (sadece bilgi)
        val_sg = val_df['subgroup'].values
        sg_n = {grp: int((val_sg == grp).sum()) for grp in ['patolojik', 'patolojik+others', 'others_only', 'belirsiz'] if (val_sg == grp).sum() > 0}
        print(f"     Anormal alt-gruplar: {sg_n}")

    elapsed = (time.time() - t0) / 60

    # ============================================================================
    # AGGREGATE
    # ============================================================================
    from sklearn.metrics import classification_report

    print(f"\n{'='*80}")
    print("📊 SONUÇLAR — 5-Fold CV (Tüm Validation Birleşik)")
    print(f"{'='*80}")

    all_logits_cat = torch.cat(all_val_logits)
    all_labels_cat = torch.cat(all_val_labels)
    all_sg_arr = np.array(all_val_subgroups)

    probs = torch.sigmoid(all_logits_cat).numpy()
    y_true = all_labels_cat.numpy()
    y_pred = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y_true, probs)
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    ap = average_precision_score(y_true, probs)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n  Overall Metrics:")
    print(f"    AUC:       {auc:.4f}")
    print(f"    F1:        {f1:.4f}")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    AP:        {ap:.4f}")

    # sklearn classification_report
    print(f"\n  Classification Report (sklearn):")
    report = classification_report(y_true, y_pred, target_names=['Normal', 'Anormal'], digits=4)
    for line in report.split('\n'):
        print(f"    {line}")

    # Confusion Matrix
    print(f"\n  Confusion Matrix:")
    print(f"    {'':15s} {'Pred Normal':>12s} {'Pred Anormal':>13s} {'Toplam':>8s}")
    print(f"    {'─'*52}")
    print(f"    {'GT Normal':15s} {cm[0,0]:>12d} {cm[0,1]:>13d} {cm[0,0]+cm[0,1]:>8d}")
    print(f"    {'GT Anormal':15s} {cm[1,0]:>12d} {cm[1,1]:>13d} {cm[1,0]+cm[1,1]:>8d}")
    print(f"    {'─'*52}")
    print(f"    {'Toplam':15s} {cm[0,0]+cm[1,0]:>12d} {cm[0,1]+cm[1,1]:>13d} {len(y_true):>8d}")

    # Alt-grup sayıları (sadece bilgi)
    print(f"\n  Anormal alt-grup dağılımı (sadece bilgi, metrik hesabına dahil değil):")
    for grp in ['patolojik', 'patolojik+others', 'others_only', 'belirsiz']:
        mask = all_sg_arr == grp
        if mask.sum() == 0: continue
        print(f"    {grp:25s}: N={int(mask.sum())}")

    # Per-fold
    print(f"\n  Per-fold:")
    for i, fr in enumerate(all_fold_results):
        print(f"    Fold {i+1}: AUC={fr['auc']:.4f}  F1={fr['f1']:.4f}  Acc={fr['acc']:.4f}")

    # Save
    save_results = {
        'pipeline': 'Binary_Normal_Abnormal_v3',
        'model': cfg.MODEL_NAME,
        'n_images': len(df), 'n_patients': df['patient_id'].nunique(),
        'n_folds': cfg.N_FOLDS, 'elapsed_min': round(elapsed, 1),
        'overall': {
            'auc': round(auc, 4), 'f1': round(f1, 4), 'acc': round(acc, 4),
            'prec': round(prec, 4), 'rec': round(rec, 4), 'ap': round(ap, 4),
        },
        'subgroup_breakdown': {},
    }
    for grp in ['normal', 'patolojik', 'patolojik+others', 'others_only', 'belirsiz']:
        mask = all_sg_arr == grp
        if mask.sum() > 0:
            save_results['subgroup_breakdown'][grp] = {
                'n': int(mask.sum()),
                'acc': round(float(accuracy_score(y_true[mask], y_pred[mask])), 4),
                'avg_prob': round(float(probs[mask].mean()), 4),
            }

    with open(results_dir / 'summary_binary_v3.json', 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)

    pred_df = df.iloc[all_val_indices].copy()
    pred_df['prob_abnormal'] = probs
    pred_df['pred_abnormal'] = y_pred
    pred_df.to_csv(results_dir / 'predictions_binary_v3.csv', index=False)

    print(f"\n  💾 {results_dir}/")
    print(f"  ⏱ {elapsed:.1f} dk")
    print(f"\n{'='*80}")
    print(f"✅ BINARY v3 TAMAMLANDI!")
    print(f"   AUC: {auc:.4f}  F1: {f1:.4f}  Acc: {acc:.4f}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

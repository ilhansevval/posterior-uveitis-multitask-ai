#!/usr/bin/env python3
"""
🏥 FUNDUS PATHOLOGY — CLASSICAL ML BASELINES
   Pipeline: EfficientNet-B4 (ImageNet, frozen) feature extraction
             → LogReg / SVM / RandomForest / XGBoost
   Aynı 5-fold split, aynı 8 label, aynı threshold optimization
   Amaç: KD framework'ünü ML baseline'ları ile karşılaştırmak (TMI tablosu)
"""

import os, time, warnings, json
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, average_precision_score, confusion_matrix
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    os.system('pip install xgboost --break-system-packages -q')
    try:
        from xgboost import XGBClassifier
        HAS_XGB = True
    except ImportError:
        HAS_XGB = False
        print("⚠️ XGBoost yüklenemedi, atlanacak")

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIG
# ============================================================================
class Config:
    DATA_ROOT = r'C:\Users\gtu\Documents\cerrahpasa\files\files'
    DATASET_CSV = os.path.join(DATA_ROOT, 'Resimler_birlesik_TF_plusYorum_clean_numeric.csv')
    RESULTS_DIR = os.path.join(DATA_ROOT, 'results_ml_baselines')

    MODEL_NAME = 'efficientnet_b4'   # feature extractor (ImageNet pretrained, frozen)
    IMG_SIZE = 380
    BATCH_SIZE = 16
    NUM_WORKERS = 0
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
    THRESHOLD_RANGE = np.arange(0.05, 0.96, 0.02)


SHORT_NAMES = {
    'Diffüz kapiller sızıntı': 'DKS', 'Optik disk boyanması': 'ODB',
    'Vitreus inflamasyonu': 'VI', 'Makula ödemi': 'MÖ',
    'Damar duvar boyanması': 'DDB', 'Retinal infiltrat': 'RI',
    'Hemoraji': 'HEM', 'Perivasküler kılıflanma': 'PVK',
}
def short_name(c): return SHORT_NAMES.get(c, c[:3])


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
# FEATURE EXTRACTOR (EfficientNet-B4, ImageNet pretrained, frozen)
# ============================================================================
class FeatureDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        img = self.transform(img)
        return img, idx


@torch.no_grad()
def extract_features(df, device):
    """EfficientNet-B4 global-pooled features (1792-dim)."""
    import timm
    model = timm.create_model(Config.MODEL_NAME, pretrained=True, num_classes=0)
    model.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = FeatureDataset(df, transform)
    loader = DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=False,
                        num_workers=Config.NUM_WORKERS, pin_memory=True)

    feats = np.zeros((len(df), model.num_features), dtype=np.float32)
    print(f"  Feature extraction ({len(df)} imgs, dim={model.num_features})...")
    t0 = time.time()
    for imgs, idxs in loader:
        imgs = imgs.to(device)
        f = model(imgs).cpu().numpy()
        feats[idxs.numpy()] = f
    print(f"  ✓ Features extracted in {(time.time()-t0)/60:.1f} min")
    return feats


# ============================================================================
# THRESHOLD + METRICS (v6 ile AYNI format)
# ============================================================================
def find_optimal_thresholds(probs, y_true, label_cols, threshold_range):
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


def compute_metrics(probs, y_true, label_cols, thresholds):
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
# CLASSIFIER FACTORY
# ============================================================================
def get_classifier(name, seed=42):
    if name == 'LogReg':
        return LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0, random_state=seed)
    elif name == 'SVM':
        return SVC(kernel='rbf', probability=True, class_weight='balanced', C=1.0, random_state=seed)
    elif name == 'RandomForest':
        return RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                      max_depth=None, n_jobs=-1, random_state=seed)
    elif name == 'XGBoost':
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric='logloss', random_state=seed, n_jobs=-1)
    else:
        raise ValueError(name)


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
    print("🏥 CLASSICAL ML BASELINES — EfficientNet-B4 features")
    print(f"   Device: {device}")
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
    for col in cfg.LABEL_COLS:
        n = int(df[col].sum())
        tag = " ★RARE" if col in cfg.RARE_LABELS else ""
        print(f"    {short_name(col):4s}: {n:4d} ({n/len(df)*100:.1f}%){tag}")

    # ── Extract features (1 kez) ──
    print(f"\n{'='*80}\n  FEATURE EXTRACTION\n{'='*80}")
    feat_cache = results_dir / 'features.npy'
    if feat_cache.exists():
        features = np.load(feat_cache)
        print(f"  ✓ Cached features loaded: {features.shape}")
    else:
        features = extract_features(df, device)
        np.save(feat_cache, features)
        print(f"  ✓ Features saved: {features.shape}")

    labels = df[cfg.LABEL_COLS].values.astype(int)

    # ── Folds ──
    fold_splits = list(balanced_stratified_group_kfold_multilabel(
        df, cfg.LABEL_COLS, 'patient_id', cfg.N_FOLDS, cfg.SEED))

    # ── Classifiers ──
    classifier_names = ['LogReg', 'SVM', 'RandomForest']
    if HAS_XGB:
        classifier_names.append('XGBoost')

    all_results = {}

    for clf_name in classifier_names:
        print(f"\n{'='*80}")
        print(f"  🤖 {clf_name}")
        print(f"{'='*80}")

        # OOF (out-of-fold) probability matrix
        oof_probs = np.zeros((len(df), cfg.N_LABELS), dtype=np.float32)
        t0 = time.time()

        for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
            X_train, X_val = features[train_idx], features[val_idx]
            y_train = labels[train_idx]

            # Standardize features
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            # Her label için ayrı binary classifier (multi-label)
            for li in range(cfg.N_LABELS):
                y_li = y_train[:, li]
                if y_li.sum() == 0:
                    # Bu fold'da pozitif yok, hepsini 0 tahmin et
                    oof_probs[val_idx, li] = 0.0
                    continue
                clf = get_classifier(clf_name, cfg.SEED)
                clf.fit(X_train_s, y_li)
                if hasattr(clf, 'predict_proba'):
                    oof_probs[val_idx, li] = clf.predict_proba(X_val_s)[:, 1]
                else:
                    oof_probs[val_idx, li] = clf.decision_function(X_val_s)

            print(f"    Fold {fold_idx+1}/{cfg.N_FOLDS} done "
                  f"({len(train_idx)} train, {len(val_idx)} val)")

        # Global threshold optimization + metrics
        thresholds = find_optimal_thresholds(oof_probs, labels, cfg.LABEL_COLS, cfg.THRESHOLD_RANGE)
        metrics = compute_metrics(oof_probs, labels, cfg.LABEL_COLS, thresholds)
        macro = metrics['__macro__']
        rare = metrics['__rare__']
        elapsed = (time.time() - t0) / 60

        print(f"\n  ── {clf_name} Sonuçları ──")
        print(f"  {'Patoloji':28s} {'N':>4s} {'F1':>7s} {'AUC':>7s} {'AP':>7s} {'Prec':>7s} {'Rec':>7s}")
        print(f"  {'─'*70}")
        for col in cfg.LABEL_COLS:
            m = metrics[col]
            auc_s = f"{m['auc']:.4f}" if m['auc'] else "  N/A"
            ap_s = f"{m['ap']:.4f}" if m['ap'] else "  N/A"
            tag = "★" if col in cfg.RARE_LABELS else " "
            print(f"  {tag}{col:27s} {m['n_pos']:>4d} {m['f1']:>7.4f} {auc_s:>7s} {ap_s:>7s} "
                  f"{m['prec']:>7.4f} {m['rec']:>7.4f}")
        print(f"  {'─'*70}")
        print(f"  {'MACRO':28s} {'':>4s} {macro['f1']:>7.4f} {macro['auc']:>7.4f} "
              f"{macro['ap']:>7.4f} {macro['prec']:>7.4f} {macro['rec']:>7.4f}")
        print(f"  {'RARE (RI/HEM/PVK)':28s} {'':>4s} {rare['f1']:>7.4f} "
              f"{rare['auc']:>7.4f}" if rare['auc'] else "")
        print(f"\n  ⏱ {clf_name}: {elapsed:.1f} min")

        all_results[clf_name] = {
            'macro_f1': float(macro['f1']),
            'macro_auc': float(macro['auc']) if macro['auc'] else None,
            'macro_ap': float(macro['ap']) if macro['ap'] else None,
            'macro_prec': float(macro['prec']),
            'macro_rec': float(macro['rec']),
            'rare_f1': float(rare['f1']),
            'rare_auc': float(rare['auc']) if rare['auc'] else None,
            'thresholds': thresholds,
            'per_label': {col: {k: (float(v) if v is not None and not isinstance(v, int) else v)
                                for k, v in metrics[col].items()} for col in cfg.LABEL_COLS},
        }

    # ══════════════════════════════════════════════════════════════════════════
    # COMPARISON TABLE
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"📊 ML BASELINE KARŞILAŞTIRMA")
    print(f"{'='*80}")
    print(f"\n  {'Model':16s} {'Macro F1':>10s} {'Macro AUC':>10s} {'Macro AP':>10s} {'Rare F1':>10s}")
    print(f"  {'─'*60}")
    for clf_name in classifier_names:
        r = all_results[clf_name]
        print(f"  {clf_name:16s} {r['macro_f1']:>10.4f} "
              f"{r['macro_auc']:>10.4f} {r['macro_ap']:>10.4f} {r['rare_f1']:>10.4f}")
    print(f"  {'─'*60}")
    print(f"  {'EffNet+CBAM(noKD)':16s} {'0.4323':>10s} {'0.9077':>10s} {'0.3928':>10s} {'0.2818':>10s}")
    print(f"  {'KD only (v3)':16s} {'0.7950':>10s} {'—':>10s} {'—':>10s} {'—':>10s}")
    print(f"  {'KD+Pseudo (v8c)':16s} {'0.9420':>10s} {'—':>10s} {'—':>10s} {'0.9620':>10s}")

    # Save
    with open(results_dir / 'ml_baselines_summary.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 {results_dir}/ml_baselines_summary.json")
    print(f"\n{'='*80}")
    print(f"✅ ML BASELINES TAMAMLANDI")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

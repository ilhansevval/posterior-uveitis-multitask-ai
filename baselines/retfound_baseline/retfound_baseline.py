#!/usr/bin/env python3
"""
🏥 FUNDUS PATHOLOGY — RETFound BASELINE (frozen feature extractor)
   RETFound (ViT-Large, retinal MAE pretrained on 1.6M images)
   → frozen 1024-dim feature → trainable classifier head
   Aynı 5-fold split, aynı 8 label, aynı threshold optimization, aynı metrics

   RETFound ağırlığı: HuggingFace transformers fork (iszt/RETFound_mae_meh)
   → AutoModel ile yüklenir. ⚠️ Repo GATED: erişim isteyip HF token gerekir.
     1) https://huggingface.co/iszt/RETFound_mae_meh → "Agree and access"
     2) https://huggingface.co/settings/tokens → read token oluştur
     3) Token'ı ortam değişkeniyle ver:  $env:HF_TOKEN = "hf_xxx"
        (veya en alttaki Config.HF_TOKEN'a yapıştır — önerilmez)

   İki değerlendirme:
     (A) Linear probe   : frozen feature → tek linear layer (sklearn LogReg)
     (B) MLP head       : frozen feature → küçük MLP (torch, focal loss, threshold opt)
   İkisinden iyisi tabloya yazılır.
"""

import os, time, random, warnings, json
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
    RESULTS_DIR = os.path.join(DATA_ROOT, 'results_retfound_baseline')

    # RETFound transformers fork — GATED repo, token gerekir
    HF_MODEL = 'iszt/RETFound_mae_meh'

    # ── HF TOKEN ──
    # Önce HF_TOKEN ortam değişkenine bakılır. Yoksa aşağıdaki string kullanılır.
    # ⚠️ Token'ı buraya yazarsan dosyayı paylaşma! Tercih: ortam değişkeni.
    HF_TOKEN = os.environ.get("HF_TOKEN", "")  # koda gomme! export HF_TOKEN=hf_...

    IMG_SIZE = 224
    BATCH_SIZE = 16
    NUM_WORKERS = 0

    # MLP head training
    EPOCHS = 50
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 12
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
    THRESHOLD_RANGE = np.arange(0.05, 0.96, 0.02)


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
# RETFound FEATURE EXTRACTION (frozen)
# ============================================================================
class ImageDataset(Dataset):
    def __init__(self, df, processor, img_size):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.img_size = img_size
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        # processor returns pixel_values
        pixel = self.processor(images=img, return_tensors='pt')['pixel_values'][0]
        return pixel, idx


@torch.no_grad()
def extract_retfound_features(df, device):
    """RETFound ViT-Large CLS/mean-pooled features (1024-dim)."""
    from transformers import AutoModel, AutoImageProcessor

    token = Config.HF_TOKEN
    if not token:
        raise RuntimeError(
            "\n  ❌ HF token bulunamadı. Bu repo GATED.\n"
            "     1) https://huggingface.co/iszt/RETFound_mae_meh → 'Agree and access'\n"
            "     2) https://huggingface.co/settings/tokens → read token oluştur\n"
            "     3) Token'ı ver:  PowerShell ->  $env:HF_TOKEN = \"hf_xxx\"\n"
            "        (veya Config.HF_TOKEN'a yapıştır)\n"
        )

    print(f"  RETFound yükleniyor: {Config.HF_MODEL} ...")
    try:
        processor = AutoImageProcessor.from_pretrained(Config.HF_MODEL, token=token)
        model = AutoModel.from_pretrained(Config.HF_MODEL, token=token)
    except Exception as e:
        raise RuntimeError(
            f"\n  ❌ Model yüklenemedi: {e}\n"
            "     Olası neden: erişim isteğin henüz onaylanmadı ya da token yanlış/yetkisiz.\n"
            "     https://huggingface.co/iszt/RETFound_mae_meh sayfasında erişimin "
            "'granted' mı kontrol et.\n"
        ) from e

    model.eval().to(device)

    # feature dim
    hidden = model.config.hidden_size  # 1024
    ds = ImageDataset(df, processor, Config.IMG_SIZE)
    loader = DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=False,
                        num_workers=Config.NUM_WORKERS, pin_memory=True)

    feats = np.zeros((len(df), hidden), dtype=np.float32)
    print(f"  Feature extraction ({len(df)} imgs, dim={hidden})...")
    t0 = time.time()
    for pixels, idxs in tqdm(loader, desc="  RETFound feat", bar_format='{l_bar}{bar:30}{r_bar}', leave=False):
        pixels = pixels.to(device)
        out = model(pixel_values=pixels)
        # last_hidden_state: (B, num_tokens, hidden). token 0 = CLS
        last = out.last_hidden_state
        cls = last[:, 0, :]                 # CLS token
        mean = last[:, 1:, :].mean(dim=1)   # patch mean
        feat = (cls + mean) / 2.0           # robust pooling
        feats[idxs.numpy()] = feat.cpu().numpy()
    print(f"  ✓ Features extracted in {(time.time()-t0)/60:.1f} min")
    return feats


# ============================================================================
# FOCAL LOSS + MLP HEAD
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


class MLPHead(nn.Module):
    def __init__(self, in_dim, n_labels):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(0.3),
            nn.Linear(in_dim, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, n_labels),
        )
    def forward(self, x): return self.net(x)


# ============================================================================
# THRESHOLD + METRICS (v6 ile AYNI)
# ============================================================================
def find_optimal_thresholds(probs, y_true, label_cols, threshold_range):
    thresholds = {}
    for i, col in enumerate(label_cols):
        gt = y_true[:, i]; best_t, best_f1 = 0.5, 0
        for t in threshold_range:
            pred = (probs[:, i] >= t).astype(int)
            f1 = f1_score(gt, pred, zero_division=0)
            if f1 > best_f1: best_f1 = f1; best_t = t
        thresholds[col] = round(float(best_t), 2)
    return thresholds


def compute_metrics(probs, y_true, label_cols, thresholds):
    results = {}
    for i, col in enumerate(label_cols):
        gt = y_true[:, i]; t = thresholds.get(col, 0.5)
        pr = (probs[:, i] >= t).astype(int); pb = probs[:, i]
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
    rare = [c for c in label_cols if c in Config.RARE_LABELS]
    rare_aucs = [results[c]['auc'] for c in rare if results[c]['auc'] is not None]
    results['__rare__'] = {
        'f1': np.mean([results[c]['f1'] for c in rare]),
        'auc': np.mean(rare_aucs) if rare_aucs else None,
        'prec': np.mean([results[c]['prec'] for c in rare]),
        'rec': np.mean([results[c]['rec'] for c in rare]),
    }
    return results


# ============================================================================
# MAIN
# ============================================================================
def main():
    cfg = Config()
    results_dir = Path(cfg.RESULTS_DIR)
    results_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("🏥 RETFound BASELINE (frozen feature) — 8 Label")
    print(f"   Device: {device} | Model: {cfg.HF_MODEL}")
    print(f"   HF token: {'set ✓' if cfg.HF_TOKEN else 'YOK ✗'}")
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

    # ── Extract RETFound features (1 kez, cache) ──
    print(f"\n{'='*80}\n  RETFound FEATURE EXTRACTION\n{'='*80}")
    feat_cache = results_dir / 'retfound_features.npy'
    if feat_cache.exists():
        features = np.load(feat_cache)
        print(f"  ✓ Cached features: {features.shape}")
    else:
        features = extract_retfound_features(df, device)
        np.save(feat_cache, features)
        print(f"  ✓ Features saved: {features.shape}")

    labels = df[cfg.LABEL_COLS].values.astype(int)
    fold_splits = list(balanced_stratified_group_kfold_multilabel(
        df, cfg.LABEL_COLS, 'patient_id', cfg.N_FOLDS, cfg.SEED))

    pos_counts = labels.sum(axis=0)
    neg_counts = len(df) - pos_counts
    pos_weight = torch.tensor(neg_counts / np.maximum(pos_counts, 1), dtype=torch.float32)
    pos_weight = torch.clamp(pos_weight, min=1.0, max=50.0).to(device)

    # ══════════════════════════════════════════════════════════════════════════
    # (A) LINEAR PROBE — sklearn LogReg per label
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}\n  (A) LINEAR PROBE (LogReg)\n{'='*80}")
    oof_lp = np.zeros((len(df), cfg.N_LABELS), dtype=np.float32)
    for fold_idx, (tr, va) in enumerate(fold_splits):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(features[tr]); Xva = scaler.transform(features[va])
        for li in range(cfg.N_LABELS):
            y = labels[tr, li]
            if y.sum() == 0:
                oof_lp[va, li] = 0.0; continue
            clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
            clf.fit(Xtr, y)
            oof_lp[va, li] = clf.predict_proba(Xva)[:, 1]
        print(f"    Fold {fold_idx+1}/{cfg.N_FOLDS} done")
    thr_lp = find_optimal_thresholds(oof_lp, labels, cfg.LABEL_COLS, cfg.THRESHOLD_RANGE)
    m_lp = compute_metrics(oof_lp, labels, cfg.LABEL_COLS, thr_lp)
    print(f"\n  Linear Probe: Macro F1={m_lp['__macro__']['f1']:.4f} "
          f"AUC={m_lp['__macro__']['auc']:.4f} rareF1={m_lp['__rare__']['f1']:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # (B) MLP HEAD — torch, focal loss, 5-fold
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}\n  (B) MLP HEAD (focal loss)\n{'='*80}")
    feat_t = torch.tensor(features, dtype=torch.float32)
    label_t = torch.tensor(labels, dtype=torch.float32)
    oof_mlp = np.zeros((len(df), cfg.N_LABELS), dtype=np.float32)

    for fold_idx, (tr, va) in enumerate(fold_splits):
        # standardize per-fold
        mu = feat_t[tr].mean(0, keepdim=True); sd = feat_t[tr].std(0, keepdim=True) + 1e-6
        Xtr = ((feat_t[tr] - mu) / sd).to(device); ytr = label_t[tr].to(device)
        Xva = ((feat_t[va] - mu) / sd).to(device); yva = label_t[va]

        model = MLPHead(features.shape[1], cfg.N_LABELS).to(device)
        criterion = FocalLossWithLogits(cfg.FOCAL_GAMMA, pos_weight)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.EPOCHS)

        best_f1, patience, best_probs = 0, 0, None
        for epoch in range(cfg.EPOCHS):
            model.train()
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), 64):
                idx = perm[i:i+64]
                opt.zero_grad()
                loss = criterion(model(Xtr[idx]), ytr[idx])
                loss.backward(); opt.step()
            sched.step()
            model.eval()
            with torch.no_grad():
                probs = torch.sigmoid(model(Xva)).cpu().numpy()
            thr = find_optimal_thresholds(probs, yva.numpy(), cfg.LABEL_COLS, cfg.THRESHOLD_RANGE)
            mf1 = compute_metrics(probs, yva.numpy(), cfg.LABEL_COLS, thr)['__macro__']['f1']
            if mf1 > best_f1:
                best_f1 = mf1; patience = 0; best_probs = probs
            else:
                patience += 1
            if patience >= cfg.PATIENCE: break
        oof_mlp[va] = best_probs
        print(f"    Fold {fold_idx+1}/{cfg.N_FOLDS}: best mF1={best_f1:.4f} (ep stop)")

    thr_mlp = find_optimal_thresholds(oof_mlp, labels, cfg.LABEL_COLS, cfg.THRESHOLD_RANGE)
    m_mlp = compute_metrics(oof_mlp, labels, cfg.LABEL_COLS, thr_mlp)
    print(f"\n  MLP Head: Macro F1={m_mlp['__macro__']['f1']:.4f} "
          f"AUC={m_mlp['__macro__']['auc']:.4f} rareF1={m_mlp['__rare__']['f1']:.4f}")

    # ── İyisini seç ──
    best_name = 'MLP Head' if m_mlp['__macro__']['f1'] >= m_lp['__macro__']['f1'] else 'Linear Probe'
    best_m = m_mlp if best_name == 'MLP Head' else m_lp
    macro = best_m['__macro__']; rare = best_m['__rare__']

    print(f"\n{'='*80}\n📊 FINAL — RETFound ({best_name} seçildi)\n{'='*80}")
    print(f"\n  {'Patoloji':28s} {'N':>4s} {'F1':>7s} {'AUC':>7s} {'AP':>7s} {'Prec':>7s} {'Rec':>7s}")
    print(f"  {'─'*70}")
    for col in cfg.LABEL_COLS:
        m = best_m[col]
        auc_s = f"{m['auc']:.4f}" if m['auc'] else "  N/A"
        ap_s = f"{m['ap']:.4f}" if m['ap'] else "  N/A"
        tag = "★" if col in cfg.RARE_LABELS else " "
        print(f"  {tag}{col:27s} {m['n_pos']:>4d} {m['f1']:>7.4f} {auc_s:>7s} {ap_s:>7s} "
              f"{m['prec']:>7.4f} {m['rec']:>7.4f}")
    print(f"  {'─'*70}")
    print(f"  {'MACRO':28s} {'':>4s} {macro['f1']:>7.4f} {macro['auc']:>7.4f} "
          f"{macro['ap']:>7.4f} {macro['prec']:>7.4f} {macro['rec']:>7.4f}")
    print(f"  {'RARE (RI/HEM/PVK)':28s} {'':>4s} {rare['f1']:>7.4f} {rare['auc']:>7.4f}")

    print(f"\n  ── İki yöntem ──")
    print(f"  RETFound Linear Probe: mF1={m_lp['__macro__']['f1']:.4f} rareF1={m_lp['__rare__']['f1']:.4f}")
    print(f"  RETFound MLP Head:     mF1={m_mlp['__macro__']['f1']:.4f} rareF1={m_mlp['__rare__']['f1']:.4f}")

    print(f"\n  ── KARŞILAŞTIRMA ──")
    print(f"  {'RETFound (best)':18s} mF1={macro['f1']:.4f}  rareF1={rare['f1']:.4f}")
    print(f"  {'DenseNet-121(noKD)':18s} mF1=0.5042  rareF1=0.4343")
    print(f"  {'EffNet vanilla':18s} mF1=0.4585  rareF1=0.3583")
    print(f"  {'ResNet-50 (noKD)':18s} mF1=0.4437  rareF1=0.3503")
    print(f"  {'EffNet+CBAM(noKD)':18s} mF1=0.4323  rareF1=0.2818")
    print(f"  {'ViT-Base/16(noKD)':18s} mF1=0.4212  rareF1=0.3209")
    print(f"  {'KD only (v3)':18s} mF1=0.7950")
    print(f"  {'KD+Pseudo (v8c)':18s} mF1=0.9420  rareF1=0.9620")

    save = {
        'model': f'RETFound frozen ({best_name})',
        'linear_probe': {'macro_f1': float(m_lp['__macro__']['f1']),
                         'macro_auc': float(m_lp['__macro__']['auc']),
                         'rare_f1': float(m_lp['__rare__']['f1'])},
        'mlp_head': {'macro_f1': float(m_mlp['__macro__']['f1']),
                     'macro_auc': float(m_mlp['__macro__']['auc']),
                     'rare_f1': float(m_mlp['__rare__']['f1'])},
        'best': {'macro_f1': float(macro['f1']), 'macro_auc': float(macro['auc']),
                 'macro_ap': float(macro['ap']), 'rare_f1': float(rare['f1']),
                 'rare_auc': float(rare['auc'])},
        'per_label': {col: {k: (float(v) if v is not None and not isinstance(v, int) else v)
                            for k, v in best_m[col].items()} for col in cfg.LABEL_COLS},
    }
    with open(results_dir / 'retfound_summary.json', 'w', encoding='utf-8') as f:
        json.dump(save, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 {results_dir}/retfound_summary.json")
    print(f"\n{'='*80}\n✅ RETFound BASELINE TAMAMLANDI — mF1={macro['f1']:.4f}\n{'='*80}")


if __name__ == '__main__':
    main()

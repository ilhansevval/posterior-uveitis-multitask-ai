# ============================================================================
# v8c-GT-025: GT Reduction %25 — Full Pseudo + Subsampled GT
# ============================================================================
# Pseudo (1393): SABİT, her fold'a tam ekleniyor
# GT (561): her fold içinde train tarafından %10 stratified subsample
# Validation: her zaman tam (subsample yok)
# Soru: "Az etiketli veriyle pseudo-label çalışıyor mu?"
# ============================================================================

import os, time, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import timm

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_ROOT = r'C:\Users\gtu\Documents\cerrahpasa\files\files'
EMB_REF_PATH = os.path.join(DATA_ROOT, 'teacher_embeddings_ref', 'teacher_embeddings.npy')
EMB_NOREF_PATH = os.path.join(DATA_ROOT, 'teacher_embeddings', 'teacher_embeddings.npy')
META_PATH = os.path.join(DATA_ROOT, 'teacher_embeddings_ref', 'teacher_metadata.csv')
PSEUDO_PATH = os.path.join(DATA_ROOT, 'pseudo_labels_agreed.csv')

# ★ GT RATIO — train tarafından kaç % GT alınacak
GT_RATIO = 0.25  
SAVE_DIR = os.path.join(DATA_ROOT, f'results_v8c_gt_{int(GT_RATIO*100):03d}')
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE = 380
BATCH_SIZE = 4
NUM_EPOCHS = 40
LR = 1e-4
NUM_FOLDS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAIN_LABELS = ['gt_DKS', 'gt_ODB', 'gt_VI', 'gt_MÖ']
RARE_LABELS = ['gt_DDB', 'gt_RI', 'gt_HEM', 'gt_PVK']
ALL_LABELS = MAIN_LABELS + RARE_LABELS
MAIN_SHORT = ['DKS', 'ODB', 'VI', 'MÖ']
RARE_SHORT = ['DDB', 'RI', 'HEM', 'PVK']
ALL_SHORT = MAIN_SHORT + RARE_SHORT
NUM_MAIN = len(MAIN_LABELS)
NUM_RARE = len(RARE_LABELS)
NUM_CLASSES = NUM_MAIN + NUM_RARE

LAMBDA_LABEL = 1.0
LAMBDA_RARE = 2.0
LAMBDA_EMB = 1.0
LAMBDA_CONTRASTIVE = 0.5
PSEUDO_WEIGHT = 0.5

MIN_THRESHOLDS = {'DDB': 0.15, 'RI': 0.15, 'HEM': 0.25, 'PVK': 0.15}
CHECKPOINT_WEIGHTS = {
    'DKS': 1.0, 'ODB': 1.0, 'VI': 1.0, 'MÖ': 1.0,
    'DDB': 2.0, 'RI': 2.0, 'HEM': 2.0, 'PVK': 2.0,
}

print(f"📂 Config: Device={DEVICE}, IMG={IMG_SIZE}, BS={BATCH_SIZE}, Ep={NUM_EPOCHS}")
print(f"🧪 EXPERIMENT: GT Ratio = {int(GT_RATIO*100)}% (Pseudo = FULL)")

# ─────────────────────────────────────────────
# LOAD GT + EMBEDDINGS
# ─────────────────────────────────────────────
meta = pd.read_csv(META_PATH)
emb_ref = np.load(EMB_REF_PATH)
emb_noref = np.load(EMB_NOREF_PATH)
embeddings_dual = np.concatenate([emb_ref, emb_noref], axis=1)
norms = np.linalg.norm(embeddings_dual, axis=1, keepdims=True)
embeddings_dual = embeddings_dual / (norms + 1e-8)
EMB_DIM = embeddings_dual.shape[1]

print(f"\n📂 GT data: {len(meta)} images, dual embedding: {embeddings_dual.shape}")
print(f"   Full GT per-label counts:")
for sn, col in zip(ALL_SHORT, ALL_LABELS):
    n = int(meta[col].sum())
    print(f"     {sn:5s}: {n:4d}")

# ─────────────────────────────────────────────
# LOAD PSEUDO (FULL - subsample yok)
# ─────────────────────────────────────────────
pseudo_df = pd.read_csv(PSEUDO_PATH)

# Convert pseudo probabilities to soft labels
for sn in ALL_SHORT:
    pseudo_df[f'gt_{sn}'] = pseudo_df[f'prob_avg_{sn}'].values

# Filter: keep only images that exist
pseudo_df = pseudo_df[pseudo_df['image_path'].apply(os.path.exists)].reset_index(drop=True)

# Keep only samples with at least 1 agreed label
pseudo_df = pseudo_df[pseudo_df['n_agreed'] > 0].reset_index(drop=True)

print(f"\n📂 Pseudo data (FULL): {len(pseudo_df)} images")
print(f"   Per-label pseudo counts (agreed ≥0.7):")
for sn in ALL_SHORT:
    n = (pseudo_df[f'prob_avg_{sn}'] >= 0.7).sum()
    print(f"     {sn:5s}: {n:4d}")

# ─────────────────────────────────────────────
# STRATIFIED SPLIT (FULL GT)
# ─────────────────────────────────────────────
def stratified_group_kfold(df, n_splits=5, random_state=42):
    np.random.seed(random_state)
    all_label_cols = MAIN_LABELS + RARE_LABELS
    n_samples = len(df)
    fold_assign = np.full(n_samples, -1, dtype=int)
    label_counts = sorted([(col, int(df[col].sum())) for col in all_label_cols], key=lambda x: x[1])
    for col, _ in label_counts:
        pos = df.index[df[col] == 1].values
        unassigned = pos[fold_assign[pos] == -1]
        if len(unassigned) == 0: continue
        np.random.shuffle(unassigned)
        already = np.zeros(n_splits, dtype=int)
        for idx in pos[fold_assign[pos] != -1]: already[fold_assign[idx]] += 1
        for idx in unassigned:
            fold_assign[idx] = int(np.argmin(already))
            already[fold_assign[idx]] += 1
    remaining = np.where(fold_assign == -1)[0]
    np.random.shuffle(remaining)
    sizes = np.array([np.sum(fold_assign == f) for f in range(n_splits)])
    for idx in remaining:
        t = int(np.argmin(sizes)); fold_assign[idx] = t; sizes[t] += 1
    return [(np.where(fold_assign != f)[0], np.where(fold_assign == f)[0]) for f in range(n_splits)]

splits = stratified_group_kfold(meta, NUM_FOLDS)

# ─────────────────────────────────────────────
# ★ GT TRAIN SUBSAMPLE (Stratified, Rare-Priority)
# ─────────────────────────────────────────────
def stratified_gt_subsample(train_df, train_indices_global, ratio, seed=42):
    """
    Train GT'den stratified %ratio subsample yap.
    Rare class'lara öncelik ver.
    Returns: filtered_df, filtered_global_indices (embedding için)
    """
    np.random.seed(seed)
    train_df = train_df.reset_index(drop=False)  # 'index' kolonu eski global idx tutar
    
    # Per-label target counts (train içindeki sayılara göre)
    target_counts = {}
    for sn, col in zip(ALL_SHORT, ALL_LABELS):
        n_in_train = int(train_df[col].sum())
        target_counts[sn] = max(1, int(round(n_in_train * ratio))) if n_in_train > 0 else 0
    
    selected_local_indices = set()
    
    # Rarity order: HEM (en rare) önce
    label_order = ['HEM', 'PVK', 'RI', 'DDB', 'MÖ', 'VI', 'ODB', 'DKS']
    
    for label in label_order:
        col = f'gt_{label}'
        target = target_counts[label]
        
        # Bu label = 1 olan train örnekleri
        candidates = train_df.index[train_df[col] == 1].tolist()
        available = [i for i in candidates if i not in selected_local_indices]
        already_have = sum(1 for i in selected_local_indices if train_df.loc[i, col] == 1)
        need = max(0, target - already_have)
        
        if need > 0 and len(available) > 0:
            chosen = np.random.choice(available, 
                                       size=min(need, len(available)), 
                                       replace=False)
            selected_local_indices.update(chosen.tolist())
    
    selected_local = sorted(selected_local_indices)
    filtered_df = train_df.loc[selected_local].copy()
    # 'index' kolonu eski global idx → embedding için
    filtered_global_indices = filtered_df['index'].values
    filtered_df = filtered_df.drop(columns=['index']).reset_index(drop=True)
    
    return filtered_df, filtered_global_indices

# ─────────────────────────────────────────────
# LOSSES
# ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.pos_weight = pos_weight
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction='none')
        pt = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - pt) ** self.gamma * bce).mean()

class SoftFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        pt = torch.clamp(pt, min=1e-6, max=1.0)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - pt) ** self.gamma * bce).mean()

class LabelAwareContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temp = temperature
    def forward(self, embeddings, labels):
        B = embeddings.shape[0]
        if B < 2: return torch.tensor(0.0, device=embeddings.device)
        sim = torch.mm(embeddings, embeddings.t()) / self.temp
        label_overlap = torch.mm(labels, labels.t())
        label_counts = labels.sum(dim=1, keepdim=True)
        union = label_counts + label_counts.t() - label_overlap
        label_sim = label_overlap / (union + 1e-8)
        mask = torch.eye(B, device=embeddings.device).bool()
        sim = sim.masked_fill(mask, -1e9)
        log_sm = F.log_softmax(sim, dim=1)
        pw = label_sim.masked_fill(mask, 0)
        pw_sum = pw.sum(dim=1)
        valid = pw_sum > 0
        if valid.sum() == 0: return torch.tensor(0.0, device=embeddings.device)
        loss = -(pw * log_sm).sum(dim=1)
        return (loss[valid] / pw_sum[valid]).mean()

# ─────────────────────────────────────────────
# DATASETS
# ─────────────────────────────────────────────
class GTDataset(Dataset):
    def __init__(self, df, emb_array, transform=None):
        self.df = df.reset_index(drop=True)
        self.emb_array = emb_array
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        if self.transform: img = self.transform(img)
        ml = torch.tensor([row[c] for c in MAIN_LABELS], dtype=torch.float32)
        rl = torch.tensor([row[c] for c in RARE_LABELS], dtype=torch.float32)
        all_l = torch.tensor([row[c] for c in ALL_LABELS], dtype=torch.float32)
        emb = torch.tensor(self.emb_array[idx], dtype=torch.float32)
        is_pseudo = torch.tensor(0.0)
        return img, ml, rl, all_l, emb, is_pseudo

class PseudoDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.dummy_emb = np.zeros(EMB_DIM, dtype=np.float32)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        if self.transform: img = self.transform(img)
        ml = torch.tensor([row[f'gt_{sn}'] for sn in MAIN_SHORT], dtype=torch.float32)
        rl = torch.tensor([row[f'gt_{sn}'] for sn in RARE_SHORT], dtype=torch.float32)
        all_l = torch.cat([ml, rl])
        emb = torch.tensor(self.dummy_emb, dtype=torch.float32)
        is_pseudo = torch.tensor(1.0)
        return img, ml, rl, all_l, emb, is_pseudo

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 1), nn.ReLU(), nn.Conv2d(64, 1, 1), nn.Sigmoid())
    def forward(self, x):
        return (x * self.conv(x)).mean(dim=[2, 3])

class StudentModelV8(nn.Module):
    def __init__(self, num_main=NUM_MAIN, num_rare=NUM_RARE, emb_dim=EMB_DIM):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=True,
                                           features_only=True, out_indices=[2, 3, 4])
        self.ch2, self.ch3, self.ch4 = 56, 160, 448
        self.attn2 = SpatialAttention(self.ch2)
        self.attn3 = SpatialAttention(self.ch3)
        self.attn4 = SpatialAttention(self.ch4)
        fused_dim = self.ch2 + self.ch3 + self.ch4
        self.rare_attn = nn.ModuleList([SpatialAttention(self.ch4) for _ in range(num_rare)])
        self.main_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(fused_dim, 512),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, num_main))
        self.rare_heads = nn.ModuleList([nn.Sequential(
            nn.Dropout(0.4), nn.Linear(self.ch4, 128),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1)) for _ in range(num_rare)])
        self.emb_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(fused_dim, 1024),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, emb_dim))

    def forward(self, x):
        feats = self.backbone(x)
        f2, f3, f4 = feats
        v2 = self.attn2(f2); v3 = self.attn3(f3); v4 = self.attn4(f4)
        fused = torch.cat([v2, v3, v4], dim=1)
        main_logits = self.main_head(fused)
        rare_outs = [head(attn(f4)) for attn, head in zip(self.rare_attn, self.rare_heads)]
        rare_logits = torch.cat(rare_outs, dim=1)
        emb = F.normalize(self.emb_head(fused), p=2, dim=1)
        return main_logits, rare_logits, emb

# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────
def weighted_f1(f1_list, label_names, weights):
    tw=tf=0
    for i, sn in enumerate(label_names):
        w=weights.get(sn,1.0); tf+=f1_list[i]*w; tw+=w
    return tf/tw

def find_thresholds(y_true, y_prob, labels, min_t=None):
    th = np.zeros(len(labels))
    for i, sn in enumerate(labels):
        best_f1=best_t=0; mt = min_t.get(sn, 0.10) if min_t else 0.10
        for t in np.arange(mt, 0.90, 0.01):
            f1 = f1_score(y_true[:,i], (y_prob[:,i]>=t).astype(int), zero_division=0)
            if f1 > best_f1: best_f1=f1; best_t=t
        th[i]=best_t
    return th

def compute_metrics(y_true, y_pred, y_prob, labels):
    res=[]
    for i,sn in enumerate(labels):
        g=y_true[:,i];p=y_pred[:,i];pb=y_prob[:,i]
        tp=int(((g==1)&(p==1)).sum());fp=int(((g==0)&(p==1)).sum())
        fn=int(((g==1)&(p==0)).sum());tn=int(((g==0)&(p==0)).sum())
        f1=f1_score(g,p,zero_division=0)
        se=tp/(tp+fn) if (tp+fn)>0 else 0;sp=tn/(tn+fp) if (tn+fp)>0 else 0
        pr=tp/(tp+fp) if (tp+fp)>0 else 0
        try:auc=roc_auc_score(g,pb) if 0<g.sum()<len(g) else 0
        except:auc=0
        try:ap=average_precision_score(g,pb) if g.sum()>0 else 0
        except:ap=0
        res.append({'label':sn,'N':int(g.sum()),'TP':tp,'FP':fp,'FN':fn,'TN':tn,
                    'F1':f1,'Sens':se,'Spec':sp,'Prec':pr,'AUC':auc,'AP':ap})
    return res

def print_table(results, title=""):
    if title: print(f"\n  {title}")
    print(f"  {'Label':6s} {'N':>5s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'TN':>5s} "
          f"{'F1':>7s} {'Sens':>7s} {'Spec':>7s} {'Prec':>7s} {'AUC':>7s} {'AP':>7s}")
    print(f"  {'─'*82}")
    for r in results:
        t=" ⚠️" if r['label'] in RARE_SHORT else ""
        print(f"  {r['label']:6s} {r['N']:5d} {r['TP']:5d} {r['FP']:5d} {r['FN']:5d} {r['TN']:5d} "
              f"{r['F1']:7.4f} {r['Sens']:7.4f} {r['Spec']:7.4f} {r['Prec']:7.4f} {r['AUC']:7.4f} {r['AP']:7.4f}{t}")
    m={k:np.mean([r[k] for r in results]) for k in ['F1','Sens','Spec','Prec','AUC','AP']}
    print(f"  {'─'*82}")
    print(f"  {'MACRO':6s} {'':5s} {'':5s} {'':5s} {'':5s} {'':5s} "
          f"{m['F1']:7.4f} {m['Sens']:7.4f} {m['Spec']:7.4f} {m['Prec']:7.4f} {m['AUC']:7.4f} {m['AP']:7.4f}")
    return m

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"🚀 TRAINING v8c-GT-{int(GT_RATIO*100):03d} ({DEVICE})")
print(f"   GT: ~{int(len(meta)*GT_RATIO)} ({int(GT_RATIO*100)}%) | Pseudo: {len(pseudo_df)} (FULL)")
print(f"{'='*80}")

all_fold_results = []
all_probs = np.zeros((len(meta), NUM_CLASSES))
all_trues = np.zeros((len(meta), NUM_CLASSES))
val_mask_global = np.zeros(len(meta), dtype=bool)  # hangi indeksler val olarak kullanıldı

for fold, (train_idx, val_idx) in enumerate(splits):
    print(f"\n{'─'*80}")
    print(f"  FOLD {fold+1}/{NUM_FOLDS}")
    vdf = meta.iloc[val_idx]
    tdf_gt_full = meta.iloc[train_idx]
    ve = embeddings_dual[val_idx]
    
    # ★ GT TRAIN SUBSAMPLE
    tdf_gt, gt_global_indices = stratified_gt_subsample(
        tdf_gt_full, train_idx, GT_RATIO, seed=42 + fold
    )
    te = embeddings_dual[gt_global_indices]
    
    print(f"  Original train GT: {len(tdf_gt_full)}")
    print(f"  Subsampled train GT: {len(tdf_gt)} ({int(GT_RATIO*100)}%)")
    print(f"  Pseudo (full): {len(pseudo_df)}")
    print(f"  Val (GT only, full): {len(vdf)}")
    print(f"  Per-label GT subsample:")
    for sn, col in zip(ALL_SHORT, ALL_LABELS):
        n_full = int(tdf_gt_full[col].sum())
        n_sub = int(tdf_gt[col].sum())
        print(f"    {sn:5s}: train_full={n_full:3d} → subsample={n_sub:3d}")

    gt_train_ds = GTDataset(tdf_gt, te, train_transform)
    pseudo_train_ds = PseudoDataset(pseudo_df, train_transform)
    combined_ds = ConcatDataset([gt_train_ds, pseudo_train_ds])

    # Sampler weights: rare GT boost, pseudo lower
    gt_weights = np.ones(len(gt_train_ds))
    for i, (_, row) in enumerate(tdf_gt.reset_index(drop=True).iterrows()):
        for sn, col in zip(RARE_SHORT, RARE_LABELS):
            if row[col] == 1:
                gt_weights[i] *= 3.0
    pseudo_weights = np.ones(len(pseudo_train_ds)) * 0.3
    all_weights = np.concatenate([gt_weights, pseudo_weights])
    all_weights = all_weights / all_weights.sum() * len(combined_ds)
    sampler = WeightedRandomSampler(torch.DoubleTensor(all_weights),
                                     len(combined_ds) * 2, replacement=True)

    tl = DataLoader(combined_ds, batch_size=BATCH_SIZE, sampler=sampler,
                    num_workers=0, pin_memory=True)
    val_ds = GTDataset(vdf, ve, val_transform)
    vl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    print(f"{'─'*80}")

    model = StudentModelV8().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Pos weights from subsampled GT
    mp = tdf_gt[MAIN_LABELS].sum().values
    mn = len(tdf_gt) - mp
    main_loss_fn = FocalLoss(0.75, 2.0, torch.tensor(mn/(mp+1), dtype=torch.float32).to(DEVICE))
    rp = tdf_gt[RARE_LABELS].sum().values
    rn = len(tdf_gt) - rp
    rare_loss_fn = FocalLoss(0.85, 2.5, torch.tensor(rn/(rp+1), dtype=torch.float32).to(DEVICE))
    soft_main_loss = SoftFocalLoss(0.75, 2.0)
    soft_rare_loss = SoftFocalLoss(0.85, 2.5)
    mse_loss_fn = nn.MSELoss()
    con_loss_fn = LabelAwareContrastiveLoss(temperature=0.1)

    best_wf1 = 0; patience = 12; no_imp = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        lm=lr_=le=lc=lp=0; nb=0
        progress = epoch / NUM_EPOCHS
        lamb_emb = LAMBDA_EMB * (1.0 - progress * 0.7)
        lamb_con = LAMBDA_CONTRASTIVE * min(1.0, progress * 2)

        for imgs, ml, rl, all_l, temb, is_pseudo in tl:
            imgs=imgs.to(DEVICE); ml=ml.to(DEVICE); rl=rl.to(DEVICE)
            all_l=all_l.to(DEVICE); temb=temb.to(DEVICE); is_pseudo=is_pseudo.to(DEVICE)

            mlog, rlog, pemb = model(imgs)

            gt_mask = (is_pseudo < 0.5)
            ps_mask = (is_pseudo >= 0.5)
            n_gt = gt_mask.sum().item()
            n_ps = ps_mask.sum().item()

            loss = torch.tensor(0.0, device=DEVICE)

            if n_gt > 0:
                l1 = main_loss_fn(mlog[gt_mask], ml[gt_mask])
                l2 = rare_loss_fn(rlog[gt_mask], rl[gt_mask])
                l3 = mse_loss_fn(pemb[gt_mask], temb[gt_mask])
                l4 = con_loss_fn(pemb[gt_mask], all_l[gt_mask]) if n_gt >= 2 else torch.tensor(0.0, device=DEVICE)
                loss = loss + LAMBDA_LABEL*l1 + LAMBDA_RARE*l2 + lamb_emb*l3 + lamb_con*l4
                lm+=l1.item(); lr_+=l2.item(); le+=l3.item(); lc+=l4.item()

            if n_ps > 0:
                l_ps_main = soft_main_loss(mlog[ps_mask], ml[ps_mask])
                l_ps_rare = soft_rare_loss(rlog[ps_mask], rl[ps_mask])
                loss = loss + PSEUDO_WEIGHT * (l_ps_main + l_ps_rare)
                lp += (l_ps_main.item() + l_ps_rare.item())

            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            nb += 1

            if nb%25==0:
                print(f"    Ep {epoch+1:2d} b{nb:4d} | gt={lm/max(nb,1):.3f}+{lr_/max(nb,1):.3f} "
                      f"emb={le/max(nb,1):.4f} pseudo={lp/max(nb,1):.4f}", end='\r')

        scheduler.step()

        model.eval()
        vmp=[]; vrp=[]; vml=[]; vrl=[]
        with torch.no_grad():
            for imgs,ml,rl,_,_,_ in vl:
                imgs=imgs.to(DEVICE)
                m,r,_=model(imgs)
                vmp.append(torch.sigmoid(m).cpu().numpy())
                vrp.append(torch.sigmoid(r).cpu().numpy())
                vml.append(ml.numpy()); vrl.append(rl.numpy())

        vap=np.hstack([np.vstack(vmp),np.vstack(vrp)])
        val_l=np.hstack([np.vstack(vml),np.vstack(vrl)])
        vad=np.zeros_like(vap)
        for i,sn in enumerate(ALL_SHORT):
            t=MIN_THRESHOLDS.get(sn,0.3) if sn in RARE_SHORT else 0.5
            vad[:,i]=(vap[:,i]>=t).astype(int)

        vf1=[f1_score(val_l[:,i],vad[:,i],zero_division=0) for i in range(NUM_CLASSES)]
        vf1m=np.mean(vf1)
        vf1w=weighted_f1(vf1, ALL_SHORT, CHECKPOINT_WEIGHTS)

        f1m=" ".join([f"{ALL_SHORT[i]}={vf1[i]:.2f}" for i in range(NUM_MAIN)])
        f1r=" ".join([f"{ALL_SHORT[NUM_MAIN+i]}={vf1[NUM_MAIN+i]:.2f}" for i in range(NUM_RARE)])
        print(f"\n  Ep {epoch+1:2d} | gt_main={lm/max(nb,1):.3f} gt_rare={lr_/max(nb,1):.3f} "
              f"pseudo={lp/max(nb,1):.4f} | F1m={vf1m:.3f} wF1={vf1w:.3f}")
        print(f"         Main: {f1m}")
        print(f"         Rare: {f1r}")

        if vf1w > best_wf1:
            best_wf1=vf1w; no_imp=0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'best_fold{fold}.pt'))
        else:
            no_imp+=1
            if no_imp>=patience:
                print(f"  ⏹ Early stop at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, f'best_fold{fold}.pt'), weights_only=True))
    model.eval()
    vmp2=[]; vrp2=[]; vml2=[]; vrl2=[]
    with torch.no_grad():
        for imgs,ml,rl,_,_,_ in vl:
            imgs=imgs.to(DEVICE); m,r,_=model(imgs)
            vmp2.append(torch.sigmoid(m).cpu().numpy()); vrp2.append(torch.sigmoid(r).cpu().numpy())
            vml2.append(ml.numpy()); vrl2.append(rl.numpy())

    vap_f=np.hstack([np.vstack(vmp2),np.vstack(vrp2)])
    val_f=np.hstack([np.vstack(vml2),np.vstack(vrl2)])
    th = find_thresholds(val_f, vap_f, ALL_SHORT, MIN_THRESHOLDS)
    vad_f=np.zeros_like(vap_f)
    for i in range(NUM_CLASSES): vad_f[:,i]=(vap_f[:,i]>=th[i]).astype(int)

    fm = compute_metrics(val_f, vad_f, vap_f, ALL_SHORT)
    fmacro = print_table(fm, title=f"✅ FOLD {fold+1} RESULTS")

    all_fold_results.append({'fold':fold+1, 'f1':fmacro['F1'], 'auc':fmacro['AUC'],
                              'gt_used':len(tdf_gt)})
    all_probs[val_idx,:NUM_CLASSES] = vap_f
    all_trues[val_idx,:NUM_CLASSES] = val_f
    val_mask_global[val_idx] = True

# ═══════════════════════════════════════════════════════════════
# GLOBAL
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"📊 GLOBAL RESULTS — v8c GT {int(GT_RATIO*100)}% + Full Pseudo")
print(f"{'='*80}")

gp=all_probs[:,:NUM_CLASSES]; gt=all_trues[:,:NUM_CLASSES]
gth = find_thresholds(gt, gp, ALL_SHORT, MIN_THRESHOLDS)
gd=np.zeros_like(gp)
for i in range(NUM_CLASSES): gd[:,i]=(gp[:,i]>=gth[i]).astype(int)

gm = compute_metrics(gt, gd, gp, ALL_SHORT)
gmacro = print_table(gm, title=f"GLOBAL METRICS (GT {int(GT_RATIO*100)}% + Full Pseudo)")

# Compare
print(f"\n{'='*80}")
print(f"📊 COMPARISON: Baseline vs GT-{int(GT_RATIO*100)}% vs Full")
print(f"{'='*80}")
baseline_f1={'DKS':.655,'ODB':.728,'VI':.451,'MÖ':.516,'DDB':.263,'RI':.500,'HEM':.141,'PVK':.204}
full_f1={'DKS':.954,'ODB':.977,'VI':.898,'MÖ':.966,'DDB':.850,'RI':.980,'HEM':.966,'PVK':.941}
f1s=[r['F1'] for r in gm]

print(f"\n  {'Lab':5s} {'N':>3s} {'Base':>7s} {f'GT{int(GT_RATIO*100)}':>7s} {'Full':>7s} {'ΔBase':>7s} {'ΔFull':>7s}")
print(f"  {'─'*50}")
for i,sn in enumerate(ALL_SHORT):
    b=baseline_f1.get(sn,0); curr=f1s[i]; f=full_f1.get(sn,0)
    d_base = curr - b; d_full = curr - f
    print(f"  {sn:5s} {gm[i]['N']:3d} {b:7.3f} {curr:7.3f} {f:7.3f} {d_base:+7.3f} {d_full:+7.3f}")
macro=np.mean(f1s)
b_macro = np.mean(list(baseline_f1.values()))
f_macro = np.mean(list(full_f1.values()))
print(f"  {'─'*50}")
print(f"  {'MACRO':5s} {'':3s} {b_macro:7.3f} {macro:7.3f} {f_macro:7.3f} "
      f"{macro-b_macro:+7.3f} {macro-f_macro:+7.3f}")

# Save
res={'gt_ratio':GT_RATIO,
     'experiment_type':'gt_reduction',
     'pseudo_n':len(pseudo_df),
     'gt_full_n':len(meta),
     'macro_f1':float(macro),
     'macro_auc':float(gmacro['AUC']),
     'per_class':{r['label']:{k:(float(v) if isinstance(v,(float,np.floating)) else v) for k,v in r.items()} for r in gm},
     'thresholds':gth.tolist(),
     'folds':all_fold_results}
with open(os.path.join(SAVE_DIR,'results.json'),'w') as f:
    json.dump(res,f,indent=2,default=str)
np.save(os.path.join(SAVE_DIR,'all_probs.npy'),gp)
np.save(os.path.join(SAVE_DIR,'all_trues.npy'),gt)

print(f"\n✅ Saved to: {SAVE_DIR}")
print(f"   GT used per fold: avg ~{np.mean([r['gt_used'] for r in all_fold_results]):.0f}")
print(f"   Pseudo: {len(pseudo_df)} (FULL)")

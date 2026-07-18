# ============================================================================
# CELL 10 v6: MULTI-SCALE ATTENTION + DUAL TEACHER + CONTRASTIVE LOSS
# ============================================================================
# Major architectural changes from v3:
#   1. Multi-scale features from EfficientNet (3 levels)
#   2. Class-specific spatial attention (each label learns where to look)
#   3. Dual teacher embedding (ref + noref → fusion)
#   4. Label-aware contrastive loss (same-label close, diff-label far)
#   5. Separate rare heads preserved
#   6. Sample-level balanced folds preserved
# ============================================================================

import os, time, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import timm

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_ROOT = r'C:\Users\gtu\Documents\cerrahpasa\files\files'
EMB_REF_PATH = os.path.join(DATA_ROOT, 'teacher_embeddings_ft', 'teacher_embeddings.npy')
EMB_NOREF_PATH = os.path.join(DATA_ROOT, 'teacher_embeddings', 'teacher_embeddings.npy')
META_PATH = os.path.join(DATA_ROOT, 'teacher_embeddings_ref', 'teacher_metadata.csv')
SAVE_DIR = os.path.join(DATA_ROOT, 'results_distillation_v6_ft')
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

MIN_THRESHOLDS = {'DDB': 0.15, 'RI': 0.15, 'HEM': 0.25, 'PVK': 0.15}

CHECKPOINT_WEIGHTS = {
    'DKS': 1.0, 'ODB': 1.0, 'VI': 1.0, 'MÖ': 1.0,
    'DDB': 2.0, 'RI': 2.0, 'HEM': 2.0, 'PVK': 2.0,
}

print(f"📂 Config: Device={DEVICE}, IMG={IMG_SIZE}, BS={BATCH_SIZE}, Ep={NUM_EPOCHS}")
print(f"   Architecture: Multi-Scale Attention + Dual Teacher + Contrastive")

# ─────────────────────────────────────────────
# LOAD DATA + DUAL EMBEDDINGS
# ─────────────────────────────────────────────
meta = pd.read_csv(META_PATH)
emb_ref = np.load(EMB_REF_PATH)    # (561, 2048)
emb_noref = np.load(EMB_NOREF_PATH)  # (561, 2048)

# Dual embedding: concatenate + L2 normalize
embeddings_dual = np.concatenate([emb_ref, emb_noref], axis=1)  # (561, 4096)
norms = np.linalg.norm(embeddings_dual, axis=1, keepdims=True)
embeddings_dual = embeddings_dual / (norms + 1e-8)
EMB_DIM = embeddings_dual.shape[1]

print(f"\n📂 Data: {len(meta)} images")
print(f"   Dual embedding: ref{emb_ref.shape} + noref{emb_noref.shape} → {embeddings_dual.shape}")
for col, sn in zip(ALL_LABELS, ALL_SHORT):
    n = int(meta[col].sum())
    tag = " ⚠️" if sn in RARE_SHORT else ""
    print(f"   {sn:6s}: {n:4d}{tag}")

# ─────────────────────────────────────────────
# STRATIFIED SPLIT
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
print(f"\n📊 Fold distribution:")
for i, (tr, va) in enumerate(splits):
    vd = meta.iloc[va]
    counts = " ".join([f"{sn}={int(vd[c].sum())}" for sn, c in zip(ALL_SHORT, ALL_LABELS)])
    print(f"  F{i+1}: Tr={len(tr)} Va={len(va)} | {counts}")

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


class LabelAwareContrastiveLoss(nn.Module):
    """Pull same-label embeddings together, push different-label apart"""
    def __init__(self, temperature=0.1, num_classes=8):
        super().__init__()
        self.temp = temperature
        self.num_classes = num_classes

    def forward(self, embeddings, labels):
        """
        embeddings: (B, D) L2-normalized
        labels: (B, C) multi-hot
        """
        B = embeddings.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=embeddings.device)

        # Cosine similarity matrix
        sim = torch.mm(embeddings, embeddings.t()) / self.temp  # (B, B)

        # Label overlap matrix: how many labels shared
        label_overlap = torch.mm(labels, labels.t())  # (B, B)
        label_counts = labels.sum(dim=1, keepdim=True)  # (B, 1)
        # Normalize: Jaccard-like similarity
        union = label_counts + label_counts.t() - label_overlap
        label_sim = label_overlap / (union + 1e-8)  # (B, B) in [0, 1]

        # Mask diagonal
        mask = torch.eye(B, device=embeddings.device).bool()
        sim = sim.masked_fill(mask, -1e9)

        # Weighted contrastive: positive pairs (high label_sim) should have high embedding sim
        # Use label_sim as soft positive weight
        log_softmax = F.log_softmax(sim, dim=1)  # (B, B)

        # Positive weight: higher label overlap → stronger pull
        pos_weight = label_sim.masked_fill(mask, 0)
        pos_weight_sum = pos_weight.sum(dim=1)

        # Avoid division by zero
        valid = pos_weight_sum > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        loss = -(pos_weight * log_softmax).sum(dim=1)
        loss = loss[valid] / pos_weight_sum[valid]
        return loss.mean()

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class FundusDatasetV6(Dataset):
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
        return img, ml, rl, all_l, emb

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
# MODEL: Multi-Scale Attention + Class-Specific Pooling
# ─────────────────────────────────────────────
class SpatialAttention(nn.Module):
    """Learn a spatial attention map for a specific purpose"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        # x: (B, C, H, W)
        attn = self.conv(x)  # (B, 1, H, W)
        return (x * attn).mean(dim=[2, 3])  # (B, C) — attention-weighted global pooling


class StudentModelV6(nn.Module):
    def __init__(self, num_main=NUM_MAIN, num_rare=NUM_RARE, emb_dim=EMB_DIM):
        super().__init__()

        # EfficientNet-B4 backbone — extract multi-scale features
        self.backbone = timm.create_model('efficientnet_b4', pretrained=True,
                                           features_only=True, out_indices=[2, 3, 4])
        # out_indices=[2,3,4] gives 3 feature levels:
        #   level 2: (B, 56, 48, 48)   — low-level texture
        #   level 3: (B, 160, 24, 24)  — mid-level structure
        #   level 4: (B, 448, 12, 12)  — high-level semantics

        # Channel dimensions for EfficientNet-B4
        self.ch2, self.ch3, self.ch4 = 56, 160, 448

        # Spatial attention per level
        self.attn2 = SpatialAttention(self.ch2)
        self.attn3 = SpatialAttention(self.ch3)
        self.attn4 = SpatialAttention(self.ch4)

        # Fused feature dimension
        fused_dim = self.ch2 + self.ch3 + self.ch4  # 56+160+448 = 664

        # Class-specific attention for rare labels (each gets own attention)
        self.rare_attn = nn.ModuleList([
            SpatialAttention(self.ch4) for _ in range(num_rare)
        ])

        # Main classification head (uses fused multi-scale features)
        self.main_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(fused_dim, 512),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, num_main),
        )

        # Rare heads (each uses class-specific attention on level 4)
        self.rare_heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(0.4), nn.Linear(self.ch4, 128),
                nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1),
            ) for _ in range(num_rare)
        ])

        # Embedding head (uses fused features → dual teacher dim)
        self.emb_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(fused_dim, 1024),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, emb_dim),
        )

    def forward(self, x):
        # Multi-scale features
        feats = self.backbone(x)  # list of 3 feature maps
        f2, f3, f4 = feats  # (B,56,48,48), (B,160,24,24), (B,448,12,12)

        # Attention-weighted pooling per level
        v2 = self.attn2(f2)  # (B, 56)
        v3 = self.attn3(f3)  # (B, 160)
        v4 = self.attn4(f4)  # (B, 448)

        # Fused multi-scale feature
        fused = torch.cat([v2, v3, v4], dim=1)  # (B, 664)

        # Main head
        main_logits = self.main_head(fused)

        # Rare heads — each with own spatial attention on deepest features
        rare_outs = []
        for i, (attn, head) in enumerate(zip(self.rare_attn, self.rare_heads)):
            rare_feat = attn(f4)  # (B, 448) — class-specific attention
            rare_outs.append(head(rare_feat))
        rare_logits = torch.cat(rare_outs, dim=1)  # (B, num_rare)

        # Embedding
        emb = F.normalize(self.emb_head(fused), p=2, dim=1)

        return main_logits, rare_logits, emb

# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────
def get_sampler_weights(df):
    mm = df[MAIN_LABELS].values; rm = df[RARE_LABELS].values
    mw = 1.0 / (mm.sum(axis=0) + 1); rw = 1.0 / (rm.sum(axis=0) + 1) * 5
    sw = (mm * mw).sum(axis=1) + (rm * rw).sum(axis=1)
    sw = np.maximum(sw, 0.1); return torch.DoubleTensor(sw / sw.sum() * len(df))

def weighted_f1(f1_list, label_names, weights):
    tw=tf=0
    for i, sn in enumerate(label_names):
        w=weights.get(sn,1.0); tf+=f1_list[i]*w; tw+=w
    return tf/tw

def find_thresholds(y_true, y_prob, label_names, min_t=None):
    th = np.zeros(len(label_names))
    for i, sn in enumerate(label_names):
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
print(f"🚀 STARTING {NUM_FOLDS}-FOLD TRAINING (v6 — Multi-Scale Attention)")
print(f"{'='*80}")

all_fold_results = []
all_probs = np.zeros((len(meta), NUM_CLASSES))
all_trues = np.zeros((len(meta), NUM_CLASSES))

for fold, (train_idx, val_idx) in enumerate(splits):
    print(f"\n{'─'*80}")
    print(f"  FOLD {fold+1}/{NUM_FOLDS} — Train: {len(train_idx)}, Val: {len(val_idx)}")
    vdf = meta.iloc[val_idx]
    for sn, col in zip(RARE_SHORT, RARE_LABELS):
        print(f"    {sn}: {int(vdf[col].sum())}")
    print(f"{'─'*80}")

    tdf = meta.iloc[train_idx]; vdf = meta.iloc[val_idx]
    te = embeddings_dual[train_idx]; ve = embeddings_dual[val_idx]

    tds = FundusDatasetV6(tdf, te, train_transform)
    vds = FundusDatasetV6(vdf, ve, val_transform)

    sw = get_sampler_weights(tdf)
    sampler = WeightedRandomSampler(sw, len(tds)*3, replacement=True)

    tl = DataLoader(tds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
    vl = DataLoader(vds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = StudentModelV6().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    mp = tdf[MAIN_LABELS].sum().values; mn = len(tdf) - mp
    main_loss_fn = FocalLoss(0.75, 2.0, torch.tensor(mn/(mp+1), dtype=torch.float32).to(DEVICE))
    rp = tdf[RARE_LABELS].sum().values; rn = len(tdf) - rp
    rare_loss_fn = FocalLoss(0.85, 2.5, torch.tensor(rn/(rp+1), dtype=torch.float32).to(DEVICE))
    mse_loss_fn = nn.MSELoss()
    contrastive_loss_fn = LabelAwareContrastiveLoss(temperature=0.1, num_classes=NUM_CLASSES)

    best_wf1 = 0; patience = 12; no_imp = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        lm=lr_=le=lc=0; nb=0

        # Curriculum: contrastive loss ramps up, emb loss ramps down
        progress = epoch / NUM_EPOCHS
        lamb_emb = LAMBDA_EMB * (1.0 - progress * 0.7)  # 1.0 → 0.3
        lamb_con = LAMBDA_CONTRASTIVE * min(1.0, progress * 2)  # 0 → 0.5

        for imgs, ml, rl, all_l, temb in tl:
            imgs=imgs.to(DEVICE); ml=ml.to(DEVICE); rl=rl.to(DEVICE)
            all_l=all_l.to(DEVICE); temb=temb.to(DEVICE)

            mlog, rlog, pemb = model(imgs)

            l1 = main_loss_fn(mlog, ml)
            l2 = rare_loss_fn(rlog, rl)
            l3 = mse_loss_fn(pemb, temb)
            l4 = contrastive_loss_fn(pemb, all_l)

            loss = LAMBDA_LABEL*l1 + LAMBDA_RARE*l2 + lamb_emb*l3 + lamb_con*l4

            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            lm+=l1.item(); lr_+=l2.item(); le+=l3.item(); lc+=l4.item(); nb+=1
            if nb%25==0:
                print(f"    Ep {epoch+1:2d} b{nb:4d} | main={lm/nb:.3f} rare={lr_/nb:.3f} "
                      f"emb={le/nb:.4f} con={lc/nb:.4f}", end='\r')

        scheduler.step()

        # Validate
        model.eval()
        vmp=[]; vrp=[]; vml=[]; vrl=[]
        with torch.no_grad():
            for imgs,ml,rl,_,_ in vl:
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
        print(f"\n  Ep {epoch+1:2d} | main={lm/nb:.3f} rare={lr_/nb:.3f} emb={le/nb:.4f} "
              f"con={lc/nb:.4f} | F1m={vf1m:.3f} wF1={vf1w:.3f}")
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

    # Best model eval
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, f'best_fold{fold}.pt'), weights_only=True))
    model.eval()
    vmp2=[]; vrp2=[]; vml2=[]; vrl2=[]
    with torch.no_grad():
        for imgs,ml,rl,_,_ in vl:
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
    print(f"  Thresholds: {' '.join([f'{ALL_SHORT[i]}={th[i]:.2f}' for i in range(NUM_CLASSES)])}")

    all_fold_results.append({'fold':fold+1, 'f1':fmacro['F1'], 'auc':fmacro['AUC']})
    all_probs[val_idx,:NUM_CLASSES] = vap_f
    all_trues[val_idx,:NUM_CLASSES] = val_f

# ═══════════════════════════════════════════════════════════════
# GLOBAL
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"📊 GLOBAL RESULTS — v6 (Multi-Scale Attention)")
print(f"{'='*80}")

gp=all_probs[:,:NUM_CLASSES]; gt=all_trues[:,:NUM_CLASSES]
gth = find_thresholds(gt, gp, ALL_SHORT, MIN_THRESHOLDS)
gd=np.zeros_like(gp)
for i in range(NUM_CLASSES): gd[:,i]=(gp[:,i]>=gth[i]).astype(int)

gm = compute_metrics(gt, gd, gp, ALL_SHORT)
gmacro = print_table(gm, title="GLOBAL METRICS (v6)")
print(f"  Thresholds: {' '.join([f'{ALL_SHORT[i]}={gth[i]:.2f}' for i in range(NUM_CLASSES)])}")

# Confusion
print(f"\n  Confusion Matrix:")
print(f"  {'GT\\Pred':8s} "+" ".join([f"{sn:>6s}" for sn in ALL_SHORT]))
print(f"  {'─'*(8+7*NUM_CLASSES)}")
for i,sn in enumerate(ALL_SHORT):
    mask=gt[:,i]==1
    counts=[int(gd[mask,j].sum()) for j in range(NUM_CLASSES)]
    print(f"  {sn:8s} "+" ".join([f"{c:6d}" for c in counts])+f"  (N={int(mask.sum())})")

print(f"\n  Per-fold:")
for r in all_fold_results:
    print(f"    F{r['fold']}: F1={r['f1']:.4f} AUC={r['auc']:.4f}")

# Compare
print(f"\n{'='*80}")
print(f"📊 COMPARISON")
print(f"{'='*80}")

prev={'DKS':.887,'ODB':.917,'VI':.746,'MÖ':.769,'DDB':.673,'RI':.824,'HEM':.759,'PVK':.788}
f1s=[r['F1'] for r in gm]
print(f"\n  {'Lab':5s} {'N':>3s} {'v3':>6s} {'v6':>6s} {'Δ':>7s}")
print(f"  {'─'*28}")
for i,sn in enumerate(ALL_SHORT):
    d=prev.get(sn,0); v6=f1s[i]; delta=v6-d
    print(f"  {sn:5s} {gm[i]['N']:3d} {d:6.3f} {v6:6.3f} {delta:+7.3f}")
macro=np.mean(f1s)
print(f"  {'─'*28}")
print(f"  {'MACRO':5s} {'':3s} {'.795':>6s} {macro:6.3f} {macro-.795:+7.3f}")

# Save
res={'macro_f1':float(macro),'macro_auc':float(gmacro['AUC']),
     'per_class':{r['label']:{k:(float(v) if isinstance(v,(float,np.floating)) else v) for k,v in r.items()} for r in gm},
     'thresholds':gth.tolist(),'folds':all_fold_results,
     'architecture':'multi_scale_attention_dual_teacher_contrastive'}
with open(os.path.join(SAVE_DIR,'results.json'),'w') as f:
    json.dump(res,f,indent=2,default=str)
np.save(os.path.join(SAVE_DIR,'all_probs.npy'),gp)
np.save(os.path.join(SAVE_DIR,'all_trues.npy'),gt)
print(f"\n✅ Saved to: {SAVE_DIR}")

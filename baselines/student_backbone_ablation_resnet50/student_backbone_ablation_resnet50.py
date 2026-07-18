# ============================================================================
# KD STUDENT BACKBONE ABLATION — resnet50 student
# ============================================================================
# v3 KD pipeline ile BİREBİR AYNI:
#   - 4 main + 4 rare head + embedding distillation head
#   - Aynı focal loss (main α=0.75 γ=2.0 / rare α=0.85 γ=2.5)
#   - Aynı λ (label=1.0, rare=2.0, emb 2.0→0.3)
#   - Aynı stratified group k-fold, sampler, thresholds
#   - Aynı teacher embeddings (2048-dim), MSE distillation
# TEK FARK: backbone efficientnet_b4 → resnet50
# Amaç: KD'nin student-agnostic olduğunu göstermek (ablation)
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
EMB_PATH = os.path.join(DATA_ROOT, 'teacher_embeddings', 'teacher_embeddings.npy')
META_PATH = os.path.join(DATA_ROOT, 'teacher_embeddings', 'teacher_metadata.csv')
SAVE_DIR = os.path.join(DATA_ROOT, 'results_distillation_resnet50')
os.makedirs(SAVE_DIR, exist_ok=True)

STUDENT_BACKBONE = 'resnet50'   # ← v3'te efficientnet_b4 idi

IMG_SIZE = 380
BATCH_SIZE = 4
NUM_EPOCHS = 35
LR = 1e-4
NUM_FOLDS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DDB moved to rare — 4 main + 4 rare = 8 labels
MAIN_LABELS = ['gt_DKS', 'gt_ODB', 'gt_VI', 'gt_MÖ']
RARE_LABELS = ['gt_DDB', 'gt_RI', 'gt_HEM', 'gt_PVK']
ALL_LABELS = MAIN_LABELS + RARE_LABELS
MAIN_SHORT = ['DKS', 'ODB', 'VI', 'MÖ']
RARE_SHORT = ['DDB', 'RI', 'HEM', 'PVK']
ALL_SHORT = MAIN_SHORT + RARE_SHORT
NUM_MAIN = len(MAIN_LABELS)
NUM_RARE = len(RARE_LABELS)
NUM_CLASSES = NUM_MAIN + NUM_RARE
EMB_DIM = 2048

LAMBDA_LABEL = 1.0
LAMBDA_RARE = 2.0
LAMBDA_EMB_START = 2.0
LAMBDA_EMB_END = 0.3

MIN_THRESHOLDS = {'DDB': 0.15, 'RI': 0.15, 'HEM': 0.25, 'PVK': 0.15}

print(f"📂 Config: Device={DEVICE}, IMG={IMG_SIZE}, BS={BATCH_SIZE}, Ep={NUM_EPOCHS}")
print(f"   ★ STUDENT BACKBONE: {STUDENT_BACKBONE} (ablation)")
print(f"   Main: {MAIN_SHORT}")
print(f"   Rare: {RARE_SHORT}")

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
meta = pd.read_csv(META_PATH)
embeddings = np.load(EMB_PATH)

print(f"\n📂 Data: {len(meta)} images, {meta['patient_id'].nunique()} patients")
for col, sn in zip(ALL_LABELS, ALL_SHORT):
    n = int(meta[col].sum())
    tag = " ⚠️ RARE" if sn in RARE_SHORT else ""
    print(f"   {sn:6s}: {n:4d} ({n/len(meta)*100:.1f}%){tag}")

# ─────────────────────────────────────────────
# STRATIFIED GROUP K-FOLD (v3 ile AYNI)
# ─────────────────────────────────────────────
def stratified_group_kfold(df, group_col, rare_cols, n_splits=5, random_state=42):
    np.random.seed(random_state)
    all_label_cols = MAIN_LABELS + RARE_LABELS
    n_samples = len(df)
    fold_assign = np.full(n_samples, -1, dtype=int)

    label_counts = [(col, int(df[col].sum())) for col in all_label_cols]
    label_counts.sort(key=lambda x: x[1])

    for col, _ in label_counts:
        pos_indices = df.index[df[col] == 1].values
        unassigned = pos_indices[fold_assign[pos_indices] == -1]
        if len(unassigned) == 0:
            continue
        np.random.shuffle(unassigned)
        already = np.zeros(n_splits, dtype=int)
        assigned_pos = pos_indices[fold_assign[pos_indices] != -1]
        for idx in assigned_pos:
            already[fold_assign[idx]] += 1
        for idx in unassigned:
            target = int(np.argmin(already))
            fold_assign[idx] = target
            already[target] += 1

    unassigned = np.where(fold_assign == -1)[0]
    np.random.shuffle(unassigned)
    fold_sizes = np.array([np.sum(fold_assign == f) for f in range(n_splits)])
    for idx in unassigned:
        target = int(np.argmin(fold_sizes))
        fold_assign[idx] = target
        fold_sizes[target] += 1

    splits = []
    for fold in range(n_splits):
        val_mask = fold_assign == fold
        splits.append((np.where(~val_mask)[0], np.where(val_mask)[0]))
    return splits

splits = stratified_group_kfold(meta, 'patient_id', RARE_LABELS, n_splits=NUM_FOLDS)

print(f"\n📊 Fold distribution:")
print(f"  {'Fold':6s} {'Train':>6s} {'Val':>5s} " +
      " ".join([f"{sn:>5s}" for sn in ALL_SHORT]))
print(f"  {'─' * (18 + 6*NUM_CLASSES)}")
for i, (tr, va) in enumerate(splits):
    val_df = meta.iloc[va]
    counts = [int(val_df[c].sum()) for c in ALL_LABELS]
    pcts = [f"{c:3d}({c/int(meta[col].sum())*100:.0f}%)" if int(meta[col].sum())>0 else f"{c:3d}"
            for c, col in zip(counts, ALL_LABELS)]
    print(f"  Fold {i+1}  {len(tr):6d} {len(va):5d} " + " ".join(pcts))

# ─────────────────────────────────────────────
# FOCAL LOSS (v3 ile AYNI)
# ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.pos_weight = pos_weight
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none')
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()

# ─────────────────────────────────────────────
# DATASET (v3 ile AYNI)
# ─────────────────────────────────────────────
class FundusDistillDataset(Dataset):
    def __init__(self, df, emb_array, transform=None):
        self.df = df.reset_index(drop=True)
        self.emb_array = emb_array
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        main_labels = torch.tensor([row[c] for c in MAIN_LABELS], dtype=torch.float32)
        rare_labels = torch.tensor([row[c] for c in RARE_LABELS], dtype=torch.float32)
        emb = torch.tensor(self.emb_array[idx], dtype=torch.float32)
        return img, main_labels, rare_labels, emb

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
# MODEL — DenseNet-121 student (v3 yapısı, sadece backbone değişti)
# ─────────────────────────────────────────────
class StudentModelResNet(nn.Module):
    def __init__(self, num_main=NUM_MAIN, num_rare=NUM_RARE, emb_dim=EMB_DIM):
        super().__init__()
        self.backbone = timm.create_model(STUDENT_BACKBONE, pretrained=True, num_classes=0)
        backbone_dim = self.backbone.num_features  # resnet50 → 2048

        self.main_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(backbone_dim, 512),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, num_main),
        )
        self.rare_heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(0.4), nn.Linear(backbone_dim, 128),
                nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1),
            ) for _ in range(num_rare)
        ])
        self.emb_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(backbone_dim, 1024),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, emb_dim),
        )

    def forward(self, x):
        features = self.backbone(x)
        main_logits = self.main_head(features)
        rare_logits = torch.cat([h(features) for h in self.rare_heads], dim=1)
        emb = F.normalize(self.emb_head(features), p=2, dim=1)
        return main_logits, rare_logits, emb

# ─────────────────────────────────────────────
# UTILITIES (v3 ile AYNI)
# ─────────────────────────────────────────────
def get_sampler_weights(df):
    main_matrix = df[MAIN_LABELS].values
    rare_matrix = df[RARE_LABELS].values
    main_counts = main_matrix.sum(axis=0)
    rare_counts = rare_matrix.sum(axis=0)
    main_weights = 1.0 / (main_counts + 1)
    rare_weights = 1.0 / (rare_counts + 1) * 5
    sample_w = (main_matrix * main_weights).sum(axis=1) + (rare_matrix * rare_weights).sum(axis=1)
    sample_w = np.maximum(sample_w, 0.1)
    sample_w = sample_w / sample_w.sum() * len(df)
    return torch.DoubleTensor(sample_w)

def get_lambda_emb(epoch, total_epochs):
    return LAMBDA_EMB_START + (LAMBDA_EMB_END - LAMBDA_EMB_START) * (epoch / total_epochs)

def find_optimal_thresholds(y_true, y_prob, label_names, min_thresholds=None):
    thresholds = np.zeros(len(label_names))
    for i, sn in enumerate(label_names):
        best_f1, best_t = 0, 0.5
        min_t = min_thresholds.get(sn, 0.10) if min_thresholds else 0.10
        for t in np.arange(min_t, 0.90, 0.025):
            pred = (y_prob[:, i] >= t).astype(int)
            f1 = f1_score(y_true[:, i], pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1; best_t = t
        thresholds[i] = best_t
    return thresholds

def compute_detailed_metrics(y_true, y_pred, y_prob, label_names):
    results = []
    for i, sn in enumerate(label_names):
        gt = y_true[:, i]; pr = y_pred[:, i]; pb = y_prob[:, i]
        tp = int(((gt==1)&(pr==1)).sum()); fp = int(((gt==0)&(pr==1)).sum())
        fn = int(((gt==1)&(pr==0)).sum()); tn = int(((gt==0)&(pr==0)).sum())
        f1 = f1_score(gt, pr, zero_division=0)
        sens = tp/(tp+fn) if (tp+fn)>0 else 0
        spec = tn/(tn+fp) if (tn+fp)>0 else 0
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        try: auc = roc_auc_score(gt, pb) if 0 < gt.sum() < len(gt) else 0
        except: auc = 0
        try: ap = average_precision_score(gt, pb) if gt.sum()>0 else 0
        except: ap = 0
        results.append({'label': sn, 'N': int(gt.sum()), 'TP': tp, 'FP': fp,
                        'FN': fn, 'TN': tn, 'F1': f1, 'Sens': sens, 'Spec': spec,
                        'Prec': prec, 'AUC': auc, 'AP': ap})
    return results

def print_table(results, title=""):
    if title: print(f"\n  {title}")
    print(f"  {'Label':6s} {'N':>5s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'TN':>5s} "
          f"{'F1':>7s} {'Sens':>7s} {'Spec':>7s} {'Prec':>7s} {'AUC':>7s} {'AP':>7s}")
    print(f"  {'─' * 82}")
    for r in results:
        tag = " ⚠️" if r['label'] in RARE_SHORT else ""
        print(f"  {r['label']:6s} {r['N']:5d} {r['TP']:5d} {r['FP']:5d} {r['FN']:5d} {r['TN']:5d} "
              f"{r['F1']:7.4f} {r['Sens']:7.4f} {r['Spec']:7.4f} "
              f"{r['Prec']:7.4f} {r['AUC']:7.4f} {r['AP']:7.4f}{tag}")
    macro = {k: np.mean([r[k] for r in results]) for k in ['F1','Sens','Spec','Prec','AUC','AP']}
    print(f"  {'─' * 82}")
    print(f"  {'MACRO':6s} {'':5s} {'':5s} {'':5s} {'':5s} {'':5s} "
          f"{macro['F1']:7.4f} {macro['Sens']:7.4f} {macro['Spec']:7.4f} "
          f"{macro['Prec']:7.4f} {macro['AUC']:7.4f} {macro['AP']:7.4f}")
    return macro

# ─────────────────────────────────────────────
# TRAINING (v3 ile AYNI)
# ─────────────────────────────────────────────
print(f"\n{'=' * 80}")
print(f"🚀 STARTING {NUM_FOLDS}-FOLD KD TRAINING — Student: {STUDENT_BACKBONE}")
print(f"{'=' * 80}")

all_fold_results = []
all_probs = np.zeros((len(meta), NUM_CLASSES))
all_trues = np.zeros((len(meta), NUM_CLASSES))

for fold, (train_idx, val_idx) in enumerate(splits):
    print(f"\n{'─' * 80}")
    print(f"  FOLD {fold+1}/{NUM_FOLDS} — Train: {len(train_idx)}, Val: {len(val_idx)}")
    val_df_fold = meta.iloc[val_idx]
    for sn, col in zip(RARE_SHORT, RARE_LABELS):
        print(f"    {sn} in val: {int(val_df_fold[col].sum())}")
    print(f"{'─' * 80}")

    train_df = meta.iloc[train_idx]
    val_df = meta.iloc[val_idx]
    train_embs = embeddings[train_idx]
    val_embs = embeddings[val_idx]

    train_ds = FundusDistillDataset(train_df, train_embs, train_transform)
    val_ds = FundusDistillDataset(val_df, val_embs, val_transform)

    sampler_weights = get_sampler_weights(train_df)
    sampler = WeightedRandomSampler(sampler_weights, len(train_ds) * 3, replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                               num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True)

    model = StudentModelResNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    main_pos = train_df[MAIN_LABELS].sum().values
    main_neg = len(train_df) - main_pos
    main_pw = torch.tensor(main_neg / (main_pos + 1), dtype=torch.float32).to(DEVICE)
    main_loss_fn = FocalLoss(alpha=0.75, gamma=2.0, pos_weight=main_pw)

    rare_pos = train_df[RARE_LABELS].sum().values
    rare_neg = len(train_df) - rare_pos
    rare_pw = torch.tensor(rare_neg / (rare_pos + 1), dtype=torch.float32).to(DEVICE)
    rare_loss_fn = FocalLoss(alpha=0.85, gamma=2.5, pos_weight=rare_pw)

    mse_loss_fn = nn.MSELoss()

    best_val_f1 = 0
    patience = 10
    no_improve = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss_main = 0; train_loss_rare = 0; train_loss_emb = 0
        n_batches = 0
        lambda_emb = get_lambda_emb(epoch, NUM_EPOCHS)

        for imgs, main_labels, rare_labels, teacher_embs in train_loader:
            imgs = imgs.to(DEVICE)
            main_labels = main_labels.to(DEVICE)
            rare_labels = rare_labels.to(DEVICE)
            teacher_embs = teacher_embs.to(DEVICE)

            main_logits, rare_logits, pred_embs = model(imgs)
            loss_main = main_loss_fn(main_logits, main_labels)
            loss_rare = rare_loss_fn(rare_logits, rare_labels)
            loss_emb = mse_loss_fn(pred_embs, teacher_embs)
            loss = LAMBDA_LABEL * loss_main + LAMBDA_RARE * loss_rare + lambda_emb * loss_emb

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_main += loss_main.item()
            train_loss_rare += loss_rare.item()
            train_loss_emb += loss_emb.item()
            n_batches += 1

            if n_batches % 25 == 0:
                print(f"    Ep {epoch+1:2d} b{n_batches:4d} | "
                      f"main={train_loss_main/n_batches:.3f} "
                      f"rare={train_loss_rare/n_batches:.3f}", end='\r')

        scheduler.step()

        # ── Validate ──
        model.eval()
        vm_probs = []; vr_probs = []; vm_labels = []; vr_labels = []
        with torch.no_grad():
            for imgs, ml, rl, _ in val_loader:
                imgs = imgs.to(DEVICE)
                m_log, r_log, _ = model(imgs)
                vm_probs.append(torch.sigmoid(m_log).cpu().numpy())
                vr_probs.append(torch.sigmoid(r_log).cpu().numpy())
                vm_labels.append(ml.numpy())
                vr_labels.append(rl.numpy())

        vap = np.hstack([np.vstack(vm_probs), np.vstack(vr_probs)])
        val = np.hstack([np.vstack(vm_labels), np.vstack(vr_labels)])

        vad = np.zeros_like(vap)
        for i, sn in enumerate(ALL_SHORT):
            t = MIN_THRESHOLDS.get(sn, 0.3) if sn in RARE_SHORT else 0.5
            vad[:, i] = (vap[:, i] >= t).astype(int)

        vf1 = [f1_score(val[:, i], vad[:, i], zero_division=0) for i in range(NUM_CLASSES)]
        vf1m = np.mean(vf1)

        f1_main = " ".join([f"{ALL_SHORT[i]}={vf1[i]:.2f}" for i in range(NUM_MAIN)])
        f1_rare = " ".join([f"{ALL_SHORT[NUM_MAIN+i]}={vf1[NUM_MAIN+i]:.2f}" for i in range(NUM_RARE)])
        print(f"\n  Ep {epoch+1:2d} | main={train_loss_main/n_batches:.3f} "
              f"rare={train_loss_rare/n_batches:.3f} λe={lambda_emb:.2f} | F1m={vf1m:.3f}")
        print(f"         Main: {f1_main}")
        print(f"         Rare: {f1_rare}")

        if vf1m > best_val_f1:
            best_val_f1 = vf1m
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'best_fold{fold}.pt'))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  ⏹ Early stop at epoch {epoch+1}")
                break

    # ── Best model eval ──
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, f'best_fold{fold}.pt'),
                                      weights_only=True))
    model.eval()
    vm_p = []; vr_p = []; vm_l = []; vr_l = []
    with torch.no_grad():
        for imgs, ml, rl, _ in val_loader:
            imgs = imgs.to(DEVICE)
            m_log, r_log, _ = model(imgs)
            vm_p.append(torch.sigmoid(m_log).cpu().numpy())
            vr_p.append(torch.sigmoid(r_log).cpu().numpy())
            vm_l.append(ml.numpy())
            vr_l.append(rl.numpy())

    vap_f = np.hstack([np.vstack(vm_p), np.vstack(vr_p)])
    val_f = np.hstack([np.vstack(vm_l), np.vstack(vr_l)])

    thresholds = find_optimal_thresholds(val_f, vap_f, ALL_SHORT, MIN_THRESHOLDS)
    vad_f = np.zeros_like(vap_f)
    for i in range(NUM_CLASSES):
        vad_f[:, i] = (vap_f[:, i] >= thresholds[i]).astype(int)

    fm = compute_detailed_metrics(val_f, vad_f, vap_f, ALL_SHORT)
    fmacro = print_table(fm, title=f"✅ FOLD {fold+1} RESULTS")
    print(f"  Thresholds: {' '.join([f'{ALL_SHORT[i]}={thresholds[i]:.3f}' for i in range(NUM_CLASSES)])}")

    all_fold_results.append({'fold': fold+1, 'f1': fmacro['F1'], 'auc': fmacro['AUC']})
    all_probs[val_idx, :NUM_CLASSES] = vap_f
    all_trues[val_idx, :NUM_CLASSES] = val_f

# ═══════════════════════════════════════════════════════════════
# GLOBAL
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print(f"📊 GLOBAL RESULTS — {NUM_FOLDS}-Fold CV (Student: {STUDENT_BACKBONE})")
print(f"{'=' * 80}")

gp = all_probs[:, :NUM_CLASSES]; gt = all_trues[:, :NUM_CLASSES]
g_thresh = find_optimal_thresholds(gt, gp, ALL_SHORT, MIN_THRESHOLDS)
gd = np.zeros_like(gp)
for i in range(NUM_CLASSES):
    gd[:, i] = (gp[:, i] >= g_thresh[i]).astype(int)

gm = compute_detailed_metrics(gt, gd, gp, ALL_SHORT)
gmacro = print_table(gm, title=f"GLOBAL METRICS ({STUDENT_BACKBONE})")
print(f"  Thresholds: {' '.join([f'{ALL_SHORT[i]}={g_thresh[i]:.3f}' for i in range(NUM_CLASSES)])}")

# Per-fold
print(f"\n  Per-fold:")
for r in all_fold_results:
    print(f"    Fold {r['fold']}: F1={r['f1']:.4f} AUC={r['auc']:.4f}")

# ── STUDENT BACKBONE COMPARISON ──
macro = np.mean([r['F1'] for r in gm])
rare_f1 = np.mean([gm[i]['F1'] for i in range(NUM_MAIN, NUM_CLASSES)])
print(f"\n{'=' * 80}")
print(f"📊 STUDENT BACKBONE ABLATION (KD only, no pseudo)")
print(f"{'=' * 80}")
print(f"  {'Student':22s} {'Macro F1':>10s} {'Rare F1':>10s}")
print(f"  {'─' * 44}")
print(f"  {'EfficientNet-B4 (v3)':22s} {'0.7950':>10s} {'—':>10s}")
print(f"  {'DenseNet-121':22s} {'0.8006':>10s} {'0.7459':>10s}")
print(f"  {STUDENT_BACKBONE + ' (this)':22s} {macro:>10.4f} {rare_f1:>10.4f}")

# Save
results = {
    'student_backbone': STUDENT_BACKBONE,
    'global_macro_f1': float(macro), 'global_macro_auc': float(gmacro['AUC']),
    'global_rare_f1': float(rare_f1),
    'per_class': {r['label']: {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                                for k, v in r.items()} for r in gm},
    'thresholds': g_thresh.tolist(), 'fold_results': all_fold_results,
}
with open(os.path.join(SAVE_DIR, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)
np.save(os.path.join(SAVE_DIR, 'all_probs.npy'), gp)
np.save(os.path.join(SAVE_DIR, 'all_trues.npy'), gt)
print(f"\n✅ Saved to: {SAVE_DIR}")

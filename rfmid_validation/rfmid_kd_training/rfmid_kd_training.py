# ============================================================================
# RFMiD KD TRAINING — Aşama 3
# Cerrahpaşa "Cell 10 v3" (StudentModelV3) yapısına SADIK, RFMiD'e uyarlanmış.
#
# Değişiklikler (FA -> RFMiD):
#   - 14 RFMiD label (extraction kodundaki sıra = frekans sırası)
#       main (>~100):  DR, MH, ODC, TSLN, DN, ARMD, MYA
#       rare (<~100):  BRVO, ODP, ODE, LS, RS, CSR, CRS
#   - Embeddings: teacher_embeddings_rfmid/  (Qwen2.5-VL-3B, GT-guided prompt)
#   - patient_id YOK -> sample-level stratified 5-fold (fonksiyon zaten öyle)
#   - USE_KD flag: True = tam KD (emb distillation), False = no-KD baseline
#       => aynı scriptten KONTROLLÜ KD vs no-KD (tek fark embedding loss)
#   - EMB_DIM otomatik (embeddings.shape[1])
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
EMB_PATH  = os.path.join(DATA_ROOT, 'teacher_embeddings_rfmid', 'teacher_embeddings.npy')
META_PATH = os.path.join(DATA_ROOT, 'teacher_embeddings_rfmid', 'teacher_metadata.csv')

# >>> TEK ANAHTAR: KD mı, no-KD baseline mı? <<<
USE_KD = True       # True = KD (emb distillation açık) | False = no-KD baseline

SAVE_DIR = os.path.join(DATA_ROOT,
                        'results_rfmid_kd' if USE_KD else 'results_rfmid_nokd')
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE = 380
BATCH_SIZE = 4
NUM_EPOCHS = 35
LR = 1e-4
NUM_FOLDS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 14 RFMiD label — frekans sırasına göre main/rare (extraction kodundaki sıra)
MAIN_SHORT = ['DR', 'MH', 'ODC', 'TSLN', 'DN', 'ARMD', 'MYA']
RARE_SHORT = ['BRVO', 'ODP', 'ODE', 'LS', 'RS', 'CSR', 'CRS']
ALL_SHORT  = MAIN_SHORT + RARE_SHORT
MAIN_LABELS = [f'gt_{s}' for s in MAIN_SHORT]
RARE_LABELS = [f'gt_{s}' for s in RARE_SHORT]
ALL_LABELS  = MAIN_LABELS + RARE_LABELS
NUM_MAIN = len(MAIN_LABELS)
NUM_RARE = len(RARE_LABELS)
NUM_CLASSES = NUM_MAIN + NUM_RARE

LAMBDA_LABEL = 1.0
LAMBDA_RARE = 2.0
LAMBDA_EMB_START = 2.0
LAMBDA_EMB_END = 0.3

# rare label'lar için minimum threshold (FP patlamasını önler)
MIN_THRESHOLDS = {s: 0.15 for s in RARE_SHORT}

print(f"📂 Config: Device={DEVICE}, IMG={IMG_SIZE}, BS={BATCH_SIZE}, Ep={NUM_EPOCHS}")
print(f"   MODE: {'KD (embedding distillation ON)' if USE_KD else 'no-KD BASELINE'}")
print(f"   SAVE: {SAVE_DIR}")
print(f"   Main: {MAIN_SHORT}")
print(f"   Rare: {RARE_SHORT}")

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
meta = pd.read_csv(META_PATH)
embeddings = np.load(EMB_PATH)
EMB_DIM = embeddings.shape[1]      # Qwen2.5-VL-3B -> 2048 (otomatik)

assert len(meta) == len(embeddings), \
    f"meta ({len(meta)}) ile embeddings ({len(embeddings)}) hizalı değil!"

# eksik gt_ kolonu varsa hata ver (label seti uyumu kontrolü)
missing = [c for c in ALL_LABELS if c not in meta.columns]
assert not missing, f"Metadata'da eksik kolon(lar): {missing}"
for c in ALL_LABELS:
    meta[c] = meta[c].astype(int)

print(f"\n📂 Data: {len(meta)} images | EMB_DIM={EMB_DIM}")
print(f"   Label dağılımı (frekans sırası kontrolü):")
for col, sn in zip(ALL_LABELS, ALL_SHORT):
    n = int(meta[col].sum())
    tag = " ⚠️ RARE" if sn in RARE_SHORT else ""
    print(f"   {sn:6s}: {n:4d} ({n/len(meta)*100:4.1f}%){tag}")
print("   (Not: main/rare bölünmesi yukarıdaki sayılarla uyumsuzsa "
      "MAIN_SHORT/RARE_SHORT listelerini düzenle.)")

# ─────────────────────────────────────────────
# STRATIFIED K-FOLD (sample-level, label-balanced — patient_id kullanmaz)
# ─────────────────────────────────────────────
def stratified_group_kfold(df, rare_cols, n_splits=5, random_state=42):
    """Her label ~%(100/n_splits) her fold'da; sıfır pozitifli fold yok."""
    np.random.seed(random_state)
    all_label_cols = MAIN_LABELS + RARE_LABELS
    n_samples = len(df)
    fold_assign = np.full(n_samples, -1, dtype=int)

    # en nadirden en sığa doğru işle
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

    # tamamı negatif olan örnekleri en küçük fold'lara dağıt
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

splits = stratified_group_kfold(meta, RARE_LABELS, n_splits=NUM_FOLDS)

print(f"\n📊 Fold dağılımı (sample-level balanced):")
print(f"  {'Fold':6s} {'Train':>6s} {'Val':>5s} " + " ".join([f"{sn:>5s}" for sn in ALL_SHORT]))
print(f"  {'─' * (18 + 6*NUM_CLASSES)}")
for i, (tr, va) in enumerate(splits):
    val_df = meta.iloc[va]
    counts = [int(val_df[c].sum()) for c in ALL_LABELS]
    print(f"  Fold {i+1}  {len(tr):6d} {len(va):5d} " + " ".join([f"{c:5d}" for c in counts]))

# ─────────────────────────────────────────────
# FOCAL LOSS
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
# DATASET
# ─────────────────────────────────────────────
class FundusDistillDataset(Dataset):
    def __init__(self, df, emb_array, transform=None):
        self.df = df.reset_index(drop=True)
        self.emb_array = emb_array
        self.transform = transform
    def __len__(self): return len(self.df)
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
# MODEL — NUM_MAIN main + NUM_RARE ayrı rare head (StudentModelV3 ile aynı)
# ─────────────────────────────────────────────
class StudentModelV3(nn.Module):
    def __init__(self, num_main=NUM_MAIN, num_rare=NUM_RARE, emb_dim=EMB_DIM):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)
        backbone_dim = self.backbone.num_features  # 1792
        self.main_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(backbone_dim, 512),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, num_main))
        self.rare_heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(0.4), nn.Linear(backbone_dim, 128),
                nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
            for _ in range(num_rare)])
        self.emb_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(backbone_dim, 1024),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, emb_dim))
    def forward(self, x):
        features = self.backbone(x)
        main_logits = self.main_head(features)
        rare_logits = torch.cat([h(features) for h in self.rare_heads], dim=1)
        emb = F.normalize(self.emb_head(features), p=2, dim=1)
        return main_logits, rare_logits, emb

# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────
def get_sampler_weights(df):
    main_matrix = df[MAIN_LABELS].values
    rare_matrix = df[RARE_LABELS].values
    main_weights = 1.0 / (main_matrix.sum(axis=0) + 1)
    rare_weights = 1.0 / (rare_matrix.sum(axis=0) + 1) * 5
    sample_w = (main_matrix * main_weights).sum(axis=1) + (rare_matrix * rare_weights).sum(axis=1)
    sample_w = np.maximum(sample_w, 0.1)
    sample_w = sample_w / sample_w.sum() * len(df)
    return torch.DoubleTensor(sample_w)

def get_lambda_emb(epoch, total_epochs):
    if not USE_KD:
        return 0.0     # no-KD: embedding distillation kapalı
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
    rare_macro = {k: np.mean([r[k] for r in results if r['label'] in RARE_SHORT])
                  for k in ['F1','AUC','AP']}
    print(f"  {'─' * 82}")
    print(f"  {'MACRO':6s} {'':5s} {'':5s} {'':5s} {'':5s} {'':5s} "
          f"{macro['F1']:7.4f} {macro['Sens']:7.4f} {macro['Spec']:7.4f} "
          f"{macro['Prec']:7.4f} {macro['AUC']:7.4f} {macro['AP']:7.4f}")
    print(f"  {'RARE':6s} {'':5s} {'':5s} {'':5s} {'':5s} {'':5s} "
          f"{rare_macro['F1']:7.4f} {'':7s} {'':7s} {'':7s} "
          f"{rare_macro['AUC']:7.4f} {rare_macro['AP']:7.4f}")
    macro['rare_f1'] = rare_macro['F1']; macro['rare_auc'] = rare_macro['AUC']
    return macro

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
print(f"\n{'=' * 80}")
print(f"🚀 RFMiD {NUM_FOLDS}-FOLD TRAINING — {'KD' if USE_KD else 'no-KD BASELINE'}")
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

    train_df = meta.iloc[train_idx]; val_df = meta.iloc[val_idx]
    train_embs = embeddings[train_idx]; val_embs = embeddings[val_idx]

    train_ds = FundusDistillDataset(train_df, train_embs, train_transform)
    val_ds = FundusDistillDataset(val_df, val_embs, val_transform)

    sampler_weights = get_sampler_weights(train_df)
    sampler = WeightedRandomSampler(sampler_weights, len(train_ds) * 3, replacement=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    model = StudentModelV3().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    main_pos = train_df[MAIN_LABELS].sum().values
    main_pw = torch.tensor((len(train_df) - main_pos) / (main_pos + 1), dtype=torch.float32).to(DEVICE)
    main_loss_fn = FocalLoss(alpha=0.75, gamma=2.0, pos_weight=main_pw)

    rare_pos = train_df[RARE_LABELS].sum().values
    rare_pw = torch.tensor((len(train_df) - rare_pos) / (rare_pos + 1), dtype=torch.float32).to(DEVICE)
    rare_loss_fn = FocalLoss(alpha=0.85, gamma=2.5, pos_weight=rare_pw)

    mse_loss_fn = nn.MSELoss()   # L2-normalize embedding'lerde MSE ≈ cosine loss

    best_val_f1 = 0; patience = 10; no_improve = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        tl_main = tl_rare = tl_emb = 0; n_batches = 0
        lambda_emb = get_lambda_emb(epoch, NUM_EPOCHS)

        for imgs, main_labels, rare_labels, teacher_embs in train_loader:
            imgs = imgs.to(DEVICE)
            main_labels = main_labels.to(DEVICE)
            rare_labels = rare_labels.to(DEVICE)
            teacher_embs = teacher_embs.to(DEVICE)

            main_logits, rare_logits, pred_embs = model(imgs)
            loss_main = main_loss_fn(main_logits, main_labels)
            loss_rare = rare_loss_fn(rare_logits, rare_labels)
            loss_emb = mse_loss_fn(pred_embs, teacher_embs)   # USE_KD=False ise ağırlığı 0
            loss = LAMBDA_LABEL * loss_main + LAMBDA_RARE * loss_rare + lambda_emb * loss_emb

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            tl_main += loss_main.item(); tl_rare += loss_rare.item(); tl_emb += loss_emb.item()
            n_batches += 1
            if n_batches % 25 == 0:
                print(f"    Ep {epoch+1:2d} b{n_batches:4d} | "
                      f"main={tl_main/n_batches:.3f} rare={tl_rare/n_batches:.3f}", end='\r')

        scheduler.step()

        # ── Validate ──
        model.eval()
        vm_probs=[]; vr_probs=[]; vm_labels=[]; vr_labels=[]
        with torch.no_grad():
            for imgs, ml, rl, _ in val_loader:
                imgs = imgs.to(DEVICE)
                m_log, r_log, _ = model(imgs)
                vm_probs.append(torch.sigmoid(m_log).cpu().numpy())
                vr_probs.append(torch.sigmoid(r_log).cpu().numpy())
                vm_labels.append(ml.numpy()); vr_labels.append(rl.numpy())

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
        print(f"\n  Ep {epoch+1:2d} | main={tl_main/n_batches:.3f} "
              f"rare={tl_rare/n_batches:.3f} emb={tl_emb/n_batches:.4f} "
              f"λe={lambda_emb:.2f} | F1m={vf1m:.3f}")
        print(f"         Main: {f1_main}")
        print(f"         Rare: {f1_rare}")

        if vf1m > best_val_f1:
            best_val_f1 = vf1m; no_improve = 0
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
    vm_p=[]; vr_p=[]; vm_l=[]; vr_l=[]
    with torch.no_grad():
        for imgs, ml, rl, _ in val_loader:
            imgs = imgs.to(DEVICE)
            m_log, r_log, _ = model(imgs)
            vm_p.append(torch.sigmoid(m_log).cpu().numpy())
            vr_p.append(torch.sigmoid(r_log).cpu().numpy())
            vm_l.append(ml.numpy()); vr_l.append(rl.numpy())

    vap_f = np.hstack([np.vstack(vm_p), np.vstack(vr_p)])
    val_f = np.hstack([np.vstack(vm_l), np.vstack(vr_l)])
    thresholds = find_optimal_thresholds(val_f, vap_f, ALL_SHORT, MIN_THRESHOLDS)
    vad_f = np.zeros_like(vap_f)
    for i in range(NUM_CLASSES):
        vad_f[:, i] = (vap_f[:, i] >= thresholds[i]).astype(int)

    fm = compute_detailed_metrics(val_f, vad_f, vap_f, ALL_SHORT)
    fmacro = print_table(fm, title=f"✅ FOLD {fold+1} RESULTS")

    all_fold_results.append({'fold': fold+1, 'f1': fmacro['F1'], 'auc': fmacro['AUC'],
                             'rare_f1': fmacro['rare_f1']})
    all_probs[val_idx, :] = vap_f
    all_trues[val_idx, :] = val_f

# ═══════════════════════════════════════════════════════════════
# GLOBAL
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print(f"📊 RFMiD GLOBAL RESULTS — {NUM_FOLDS}-Fold CV — {'KD' if USE_KD else 'no-KD'}")
print(f"{'=' * 80}")

gp = all_probs; gt = all_trues
g_thresh = find_optimal_thresholds(gt, gp, ALL_SHORT, MIN_THRESHOLDS)
gd = np.zeros_like(gp)
for i in range(NUM_CLASSES):
    gd[:, i] = (gp[:, i] >= g_thresh[i]).astype(int)

gm = compute_detailed_metrics(gt, gd, gp, ALL_SHORT)
gmacro = print_table(gm, title="GLOBAL METRICS")

print(f"\n  Per-fold:")
for r in all_fold_results:
    print(f"    Fold {r['fold']}: F1={r['f1']:.4f} AUC={r['auc']:.4f} RareF1={r['rare_f1']:.4f}")

fold_f1 = np.array([r['f1'] for r in all_fold_results])
print(f"\n  Fold-mean macro F1: {fold_f1.mean():.4f} ± {fold_f1.std(ddof=1):.4f}")
print(f"  Pooled  macro F1:   {gmacro['F1']:.4f}  | Rare F1: {gmacro['rare_f1']:.4f}")

# Save
results = {
    'mode': 'KD' if USE_KD else 'no-KD',
    'global_macro_f1': float(gmacro['F1']),
    'global_macro_auc': float(gmacro['AUC']),
    'global_rare_f1': float(gmacro['rare_f1']),
    'fold_mean_f1': float(fold_f1.mean()),
    'fold_std_f1': float(fold_f1.std(ddof=1)),
    'per_class': {r['label']: {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                               for k, v in r.items()} for r in gm},
    'thresholds': g_thresh.tolist(),
    'fold_results': all_fold_results,
}
with open(os.path.join(SAVE_DIR, 'results.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, default=str)
np.save(os.path.join(SAVE_DIR, 'all_probs.npy'), gp)
np.save(os.path.join(SAVE_DIR, 'all_trues.npy'), gt)
print(f"\n✅ Saved to: {SAVE_DIR}")
print(f"   (KD vs no-KD karşılaştırması için USE_KD'yi değiştirip tekrar çalıştır.)")

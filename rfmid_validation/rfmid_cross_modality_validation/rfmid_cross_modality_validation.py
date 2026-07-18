#!/usr/bin/env python3
"""
🌐 RFMiD CROSS-MODALITY VALIDATION
   Senin FA-eğitimli modelin (v8c, KD+Pseudo) → RFMiD CFP görüntülerinde test
   = FA→CFP cross-modality generalization

   Model: StudentModelV8, 5-fold ENSEMBLE (olasılık ortalaması)
   Test:  3200 RFMiD CFP görüntüsü
   Metrik: AUC (threshold-free, az örnekte güvenilir) + AP

   5 overlapping label (dürüst raporlama, az örneklem işaretli):
     ODB (pred) → ODE   (n=96)   ✅ güvenilir
     RI  (pred) → RS+CRS (n=125) ✅ güvenilir
     HEM (pred) → HR+VH+PRH (n=10) ⚠️ az örneklem
     MÖ  (pred) → CME   (n=7)    ⚠️ az örneklem
     DDB (pred) → VS    (n=4)    ⚠️ az örneklem
"""

import os, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
import timm

try:
    from tqdm import tqdm
except ImportError:
    os.system('pip install tqdm --break-system-packages -q')
    from tqdm import tqdm

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
# v8c model checkpoints
CERR_ROOT = r'C:\Users\gtu\Documents\cerrahpasa\files\files'
V8C_DIR = os.path.join(CERR_ROOT, 'results_distillation_v8c')
CHECKPOINTS = [os.path.join(V8C_DIR, f'best_fold{i}.pt') for i in range(5)]

# RFMiD data
RFMID_ROOT = r'C:\Users\gtu\Downloads\A. RFMiD_All_Classes_Dataset'
RFMID_IMG = os.path.join(RFMID_ROOT, '1. Original Images')
RFMID_GT = os.path.join(RFMID_ROOT, '2. Groundtruths')
RFMID_SETS = {
    'Training':   (os.path.join(RFMID_IMG, 'a. Training Set'),
                   os.path.join(RFMID_GT, 'a. RFMiD_Training_Labels.csv')),
    'Validation': (os.path.join(RFMID_IMG, 'b. Validation Set'),
                   os.path.join(RFMID_GT, 'b. RFMiD_Validation_Labels.csv')),
    'Testing':    (os.path.join(RFMID_IMG, 'c. Testing Set'),
                   os.path.join(RFMID_GT, 'c. RFMiD_Testing_Labels.csv')),
}

SAVE_DIR = os.path.join(CERR_ROOT, 'results_rfmid_crossmodality')
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE = 380
BATCH_SIZE = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model output order (v8c): main [DKS,ODB,VI,MÖ] + rare [DDB,RI,HEM,PVK]
MODEL_LABELS = ['DKS', 'ODB', 'VI', 'MÖ', 'DDB', 'RI', 'HEM', 'PVK']
EMB_DIM = 4096  # v8c dual embedding (model init için gerekli)

# Senin label (model çıktısı) → RFMiD ground-truth kolonları
# (model'in pred'i → RFMiD'de hangi kolon(lar)la kıyaslanacak)
OVERLAP = {
    'ODB': (['ODE'],            96, 'güvenilir'),   # optik disk
    'RI':  (['RS', 'CRS'],      125, 'güvenilir'),  # retinit/koryoretinit
    'HEM': (['HR', 'VH', 'PRH'], 10, 'az örneklem'),# hemoraji
    'MÖ':  (['CME'],             7, 'az örneklem'), # makula ödemi
    'DDB': (['VS'],              4, 'az örneklem'), # vaskülit
}

print("=" * 80)
print("🌐 RFMiD CROSS-MODALITY VALIDATION (FA→CFP)")
print(f"   Model: v8c 5-fold ensemble | Device: {DEVICE}")
print("=" * 80)

# ─────────────────────────────────────────────
# MODEL (v8c — birebir aynı tanım, checkpoint yüklenebilsin)
# ─────────────────────────────────────────────
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 1), nn.ReLU(), nn.Conv2d(64, 1, 1), nn.Sigmoid())
    def forward(self, x):
        return (x * self.conv(x)).mean(dim=[2, 3])

class StudentModelV8(nn.Module):
    def __init__(self, num_main=4, num_rare=4, emb_dim=EMB_DIM):
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
# RFMiD DATASET
# ─────────────────────────────────────────────
class RFMiDDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{int(row['ID'])}.png")
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, idx

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# 1. RFMiD verisini yükle (3 set birleşik)
# ─────────────────────────────────────────────
print(f"\n  RFMiD yükleniyor...")
all_dfs = []
for split, (img_dir, csv_path) in RFMID_SETS.items():
    df = pd.read_csv(csv_path)
    df['__img_dir__'] = img_dir
    # sadece dosyası var olanları tut
    df = df[df['ID'].apply(lambda i: os.path.exists(os.path.join(img_dir, f"{int(i)}.png")))]
    all_dfs.append(df)
    print(f"    {split:12s}: {len(df)} görüntü")
rfmid = pd.concat(all_dfs, ignore_index=True)
print(f"    TOPLAM: {len(rfmid)} görüntü")


# ─────────────────────────────────────────────
# 2. 5-fold ensemble ile RFMiD tahmin
# ─────────────────────────────────────────────
print(f"\n  5-fold ensemble tahmin...")

# Her set kendi img_dir'inden okunacağı için set bazında işliyoruz
ensemble_probs = np.zeros((len(rfmid), 8), dtype=np.float32)  # 8 model label

# checkpoint kontrolü
missing = [c for c in CHECKPOINTS if not os.path.exists(c)]
if missing:
    print(f"  ❌ Eksik checkpoint: {missing}")
    raise SystemExit

# Her fold için model yükle, tüm RFMiD'i tahmin et, olasılıkları topla
for fold_i, ckpt in enumerate(CHECKPOINTS):
    model = StudentModelV8().to(DEVICE)
    state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # set bazında (her set farklı klasör)
    fold_probs = np.zeros((len(rfmid), 8), dtype=np.float32)
    offset = 0
    for split, (img_dir, _) in RFMID_SETS.items():
        sub = rfmid[rfmid['__img_dir__'] == img_dir]
        if len(sub) == 0: continue
        ds = RFMiDDataset(sub, img_dir, val_transform)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        sub_global_idx = sub.index.values  # rfmid içindeki global index
        with torch.no_grad():
            for imgs, local_idx in tqdm(loader, desc=f"  Fold{fold_i+1} {split[:4]}",
                                        bar_format='{l_bar}{bar:20}{r_bar}', leave=False):
                imgs = imgs.to(DEVICE)
                m_log, r_log, _ = model(imgs)
                probs = torch.cat([torch.sigmoid(m_log), torch.sigmoid(r_log)], dim=1).cpu().numpy()
                gi = sub_global_idx[local_idx.numpy()]
                fold_probs[gi] = probs
    ensemble_probs += fold_probs
    print(f"    Fold {fold_i+1}/5 tahmin tamamlandı")

ensemble_probs /= len(CHECKPOINTS)  # ortalama
print(f"  ✓ Ensemble olasılıkları hazır: {ensemble_probs.shape}")


# ─────────────────────────────────────────────
# 3. Overlapping label'larda AUC/AP hesapla
# ─────────────────────────────────────────────
print(f"\n{'='*80}\n📊 CROSS-MODALITY SONUÇLARI (FA→CFP)\n{'='*80}")
print(f"\n  {'Model':5s} {'→':1s} {'RFMiD GT':14s} {'N+':>4s} {'AUC':>7s} {'AP':>7s}  {'Güven'}")
print(f"  {'─'*60}")

results = {}
for model_label, (rfmid_cols, expected_n, reliability) in OVERLAP.items():
    # model çıktısındaki index
    mi = MODEL_LABELS.index(model_label)
    pred_prob = ensemble_probs[:, mi]

    # RFMiD ground-truth: OR of mapped columns (herhangi biri 1 ise pozitif)
    gt = np.zeros(len(rfmid), dtype=int)
    valid_cols = []
    for col in rfmid_cols:
        if col in rfmid.columns:
            gt = gt | rfmid[col].values.astype(int)
            valid_cols.append(col)
    n_pos = int(gt.sum())

    if n_pos == 0 or n_pos == len(gt):
        print(f"  {model_label:5s} → {'+'.join(valid_cols):14s} {n_pos:>4d} {'N/A':>7s} {'N/A':>7s}  {reliability}")
        continue

    auc = roc_auc_score(gt, pred_prob)
    ap = average_precision_score(gt, pred_prob)
    flag = "✅" if reliability == 'güvenilir' else "⚠️"
    print(f"  {model_label:5s} → {'+'.join(valid_cols):14s} {n_pos:>4d} {auc:>7.4f} {ap:>7.4f}  {flag} {reliability}")

    results[model_label] = {
        'rfmid_cols': valid_cols, 'n_pos': n_pos,
        'auc': float(auc), 'ap': float(ap), 'reliability': reliability,
    }

# ─────────────────────────────────────────────
# 4. Özet + yorum
# ─────────────────────────────────────────────
print(f"\n{'='*80}\n📋 ÖZET\n{'='*80}")

reliable = {k: v for k, v in results.items() if v['reliability'] == 'güvenilir'}
if reliable:
    mean_auc = np.mean([v['auc'] for v in reliable.values()])
    print(f"\n  Güvenilir label'lar (n≥90):")
    for k, v in reliable.items():
        print(f"    {k} → {'+'.join(v['rfmid_cols'])}: AUC={v['auc']:.4f} (n={v['n_pos']})")
    print(f"  → Ortalama AUC (güvenilir): {mean_auc:.4f}")

low_n = {k: v for k, v in results.items() if v['reliability'] == 'az örneklem'}
if low_n:
    print(f"\n  Az örneklemli label'lar (n<15, gösterge niteliğinde):")
    for k, v in low_n.items():
        print(f"    {k} → {'+'.join(v['rfmid_cols'])}: AUC={v['auc']:.4f} (n={v['n_pos']}) ⚠️")

print(f"\n  YORUM:")
print(f"  FA'da eğitilen model, hiç CFP görmeden, RFMiD renkli fundus")
print(f"  görüntülerinde overlapping bulguları AUC>0.5 ile ayırt edebiliyorsa")
print(f"  → cross-modality generalization kanıtı.")

# ─────────────────────────────────────────────
# 5. Kaydet
# ─────────────────────────────────────────────
save = {
    'model': 'v8c (KD+Pseudo) 5-fold ensemble',
    'rfmid_total_images': len(rfmid),
    'modality': 'FA(train) → CFP(test)',
    'results': results,
    'mean_auc_reliable': float(np.mean([v['auc'] for v in reliable.values()])) if reliable else None,
}
with open(os.path.join(SAVE_DIR, 'rfmid_crossmodality_results.json'), 'w', encoding='utf-8') as f:
    json.dump(save, f, indent=2, ensure_ascii=False)

# olasılıkları da kaydet (sonra ROC eğrisi vs için)
np.save(os.path.join(SAVE_DIR, 'ensemble_probs.npy'), ensemble_probs)
rfmid[['ID', '__img_dir__']].to_csv(os.path.join(SAVE_DIR, 'rfmid_image_index.csv'), index=False)

print(f"\n  💾 {SAVE_DIR}/")
print(f"\n{'='*80}\n✅ RFMiD CROSS-MODALITY VALIDATION TAMAMLANDI\n{'='*80}")

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import timm

DATA_ROOT = r'C:\Users\gtu\Documents\cerrahpasa\files\files'
V6_DIR = os.path.join(DATA_ROOT, 'results_distillation_v6')
CANDIDATES_CSV = os.path.join(DATA_ROOT, 'pseudo_abnormal_candidates.csv')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAIN_LABELS = ['gt_DKS', 'gt_ODB', 'gt_VI', 'gt_MÖ']
RARE_LABELS = ['gt_DDB', 'gt_RI', 'gt_HEM', 'gt_PVK']
ALL_SHORT = ['DKS', 'ODB', 'VI', 'MÖ', 'DDB', 'RI', 'HEM', 'PVK']
NUM_MAIN = 4; NUM_RARE = 4; IMG_SIZE = 380

# v6 thresholds (from training)
THRESHOLDS = {'DKS': 0.58, 'ODB': 0.63, 'VI': 0.79, 'MÖ': 0.57,
              'DDB': 0.65, 'RI': 0.33, 'HEM': 0.25, 'PVK': 0.33}

# v6 model architecture (must match training)
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 1), nn.ReLU(), nn.Conv2d(64, 1, 1), nn.Sigmoid())
    def forward(self, x):
        return (x * self.conv(x)).mean(dim=[2, 3])

class StudentModelV6(nn.Module):
    def __init__(self, num_main=NUM_MAIN, num_rare=NUM_RARE, emb_dim=4096):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=False,
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
        self.rare_heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(0.4), nn.Linear(self.ch4, 128),
                nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
            for _ in range(num_rare)])
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

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load all 5 fold v6 models
print("🧠 Loading 5 v6 models...")
models = []
for fold in range(5):
    model_path = os.path.join(V6_DIR, f'best_fold{fold}.pt')
    if not os.path.exists(model_path):
        print(f"   ⚠️ Fold {fold} not found")
        continue
    model = StudentModelV6().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    models.append(model)
    print(f"   ✅ Fold {fold}")
print(f"   {len(models)} models loaded")

# Load candidates
cand_df = pd.read_csv(CANDIDATES_CSV)
print(f"\n📂 Candidates: {len(cand_df)} high-conf abnormal images")

# Predict multi-label
print(f"\n🔍 Multi-label prediction...")
all_results = []

with torch.no_grad():
    for idx, row in cand_df.iterrows():
        img_path = row['image_path']
        if not os.path.exists(img_path):
            continue
        try:
            img = Image.open(img_path).convert('RGB')
            img_t = val_transform(img).unsqueeze(0).to(DEVICE)

            # Ensemble: average probabilities from all folds
            fold_main_probs = []
            fold_rare_probs = []
            for model in models:
                m_log, r_log, _ = model(img_t)
                fold_main_probs.append(torch.sigmoid(m_log).cpu().numpy())
                fold_rare_probs.append(torch.sigmoid(r_log).cpu().numpy())

            avg_main = np.mean(fold_main_probs, axis=0)[0]  # (4,)
            avg_rare = np.mean(fold_rare_probs, axis=0)[0]  # (4,)
            avg_all = np.concatenate([avg_main, avg_rare])    # (8,)

            result = {
                'image_path': img_path,
                'image_name': row['filename'],
                'folder': row['folder'],
                'binary_prob': row['prob_abnormal'],
            }
            for i, sn in enumerate(ALL_SHORT):
                result[f'prob_{sn}'] = float(avg_all[i])
                result[f'pred_{sn}'] = int(avg_all[i] >= THRESHOLDS[sn])

            # Count predicted labels
            result['n_pred_labels'] = sum(result[f'pred_{sn}'] for sn in ALL_SHORT)
            result['pred_labels'] = '+'.join([sn for sn in ALL_SHORT if result[f'pred_{sn}'] == 1])

            all_results.append(result)
        except:
            pass

        if (idx + 1) % 200 == 0:
            print(f"   [{idx+1}/{len(cand_df)}]...")

results_df = pd.DataFrame(all_results)

# Stats
print(f"\n{'='*60}")
print(f"📊 MULTI-LABEL PREDICTION RESULTS")
print(f"{'='*60}")
print(f"  Total predicted: {len(results_df)}")
print(f"  With ≥1 label: {(results_df['n_pred_labels'] > 0).sum()}")
print(f"  With 0 labels: {(results_df['n_pred_labels'] == 0).sum()}")

print(f"\n  Per-label counts:")
for sn in ALL_SHORT:
    n = results_df[f'pred_{sn}'].sum()
    avg_p = results_df[f'prob_{sn}'].mean()
    high_conf = (results_df[f'prob_{sn}'] >= 0.8).sum()
    print(f"    {sn:5s}: {n:5d} predicted | avg_prob={avg_p:.3f} | high_conf(≥0.8)={high_conf}")

print(f"\n  Label combination distribution (top 15):")
combo_counts = results_df['pred_labels'].value_counts().head(15)
for combo, count in combo_counts.items():
    print(f"    {combo if combo else '(none)':30s}: {count}")

# Save
save_path = os.path.join(DATA_ROOT, 'pseudo_labels_v6.csv')
results_df.to_csv(save_path, index=False)
print(f"\n💾 Saved: {save_path}")

# High confidence pseudo-labels (all probs above threshold + max prob ≥ 0.7)
high_conf = results_df[results_df['n_pred_labels'] > 0].copy()
max_probs = high_conf[[f'prob_{sn}' for sn in ALL_SHORT]].max(axis=1)
high_conf = high_conf[max_probs >= 0.7]
hc_path = os.path.join(DATA_ROOT, 'pseudo_labels_high_conf.csv')
high_conf.to_csv(hc_path, index=False)
print(f"💾 High-conf pseudo-labels (max_prob≥0.7): {len(high_conf)} → {hc_path}")

# Per-label high confidence counts
print(f"\n  High-conf per label (prob≥0.7):")
for sn in ALL_SHORT:
    n = (high_conf[f'prob_{sn}'] >= 0.7).sum() if len(high_conf) > 0 else 0
    print(f"    {sn:5s}: {n}")

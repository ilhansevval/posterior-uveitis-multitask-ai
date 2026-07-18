import os, glob
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import timm
import torch.nn as nn

DATA_ROOT = r'C:\Users\gtu\Documents\cerrahpasa\files\files'
BINARY_MODEL_DIR = os.path.join(DATA_ROOT, 'results_binary_v3')
UNLABELED_CSV = os.path.join(DATA_ROOT, 'unlabeled_images.csv')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load unlabeled images list
unlabeled_df = pd.read_csv(UNLABELED_CSV)
print(f"📂 Unlabeled images: {len(unlabeled_df)}")

# Binary model
class FundusBinaryModel(nn.Module):
    def __init__(self, model_name='efficientnet_b4', dropout=0.4):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        n_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(n_features, 256),
            nn.ReLU(), nn.Dropout(dropout * 0.5), nn.Linear(256, 1))
    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(-1)

val_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load all 5 fold models for ensemble prediction
print("\n🧠 Loading 5 binary models...")
models = []
for fold in range(5):
    model_path = os.path.join(BINARY_MODEL_DIR, f'model_fold{fold}.pth')
    if not os.path.exists(model_path):
        print(f"   ⚠️ Fold {fold} model not found: {model_path}")
        continue
    model = FundusBinaryModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    models.append(model)
    print(f"   ✅ Fold {fold} loaded")

print(f"   {len(models)} models loaded")

# Predict all unlabeled images
print(f"\n🔍 Predicting {len(unlabeled_df)} images...")
all_probs = []

with torch.no_grad():
    for idx, row in unlabeled_df.iterrows():
        img_path = row['image_path']
        if not os.path.exists(img_path):
            all_probs.append(-1)  # missing
            continue
        
        try:
            img = Image.open(img_path).convert('RGB')
            img_t = val_transform(img).unsqueeze(0).to(DEVICE)
            
            # Ensemble: average probability from all folds
            fold_probs = []
            for model in models:
                logit = model(img_t)
                prob = torch.sigmoid(logit).item()
                fold_probs.append(prob)
            
            avg_prob = np.mean(fold_probs)
            all_probs.append(avg_prob)
        except Exception as e:
            all_probs.append(-1)
        
        if (idx + 1) % 500 == 0:
            print(f"   [{idx+1}/{len(unlabeled_df)}] done...")

unlabeled_df['prob_abnormal'] = all_probs
unlabeled_df['pred_abnormal'] = (unlabeled_df['prob_abnormal'] >= 0.5).astype(int)

# Filter out failed images
valid = unlabeled_df['prob_abnormal'] >= 0
unlabeled_df = unlabeled_df[valid].reset_index(drop=True)

# Stats
n_normal = (unlabeled_df['pred_abnormal'] == 0).sum()
n_abnormal = (unlabeled_df['pred_abnormal'] == 1).sum()
n_high_conf_abnormal = (unlabeled_df['prob_abnormal'] >= 0.8).sum()
n_high_conf_normal = (unlabeled_df['prob_abnormal'] <= 0.2).sum()

print(f"\n{'='*60}")
print(f"📊 BINARY PREDICTION RESULTS")
print(f"{'='*60}")
print(f"  Total: {len(unlabeled_df)}")
print(f"  Normal (prob<0.5): {n_normal} ({n_normal/len(unlabeled_df)*100:.1f}%)")
print(f"  Abnormal (prob≥0.5): {n_abnormal} ({n_abnormal/len(unlabeled_df)*100:.1f}%)")
print(f"  High confidence abnormal (prob≥0.8): {n_high_conf_abnormal}")
print(f"  High confidence normal (prob≤0.2): {n_high_conf_normal}")

# Probability distribution
print(f"\n  Probability distribution:")
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    n = (unlabeled_df['prob_abnormal'] >= t).sum()
    print(f"    prob ≥ {t:.1f}: {n:5d}")

# Save
save_path = os.path.join(DATA_ROOT, 'unlabeled_predictions.csv')
unlabeled_df.to_csv(save_path, index=False)
print(f"\n💾 Saved: {save_path}")

# Save high confidence abnormals separately
abnormal_df = unlabeled_df[unlabeled_df['prob_abnormal'] >= 0.7].copy()
abnormal_path = os.path.join(DATA_ROOT, 'pseudo_abnormal_candidates.csv')
abnormal_df.to_csv(abnormal_path, index=False)
print(f"💾 High-conf abnormal candidates (prob≥0.7): {len(abnormal_df)} → {abnormal_path}")

# ============================================================================
# CELL 8b-RETFOUND-CFP: SETUP + BATCH EXTRACTION
# ============================================================================
# RETFound MAE pre-trained on 904K Color Fundus Photos
# Encoder-only ViT-Large/16 — outputs 1024-dim CLS embedding
# No prompts, no references — just raw image → embedding
# ============================================================================

import os, glob, time
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from huggingface_hub import hf_hub_download

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_ROOT = r'C:\Users\gtu\Documents\cerrahpasa\files\files'
DATASET_CSV = os.path.join(DATA_ROOT, 'Resimler_birlesik_TF_plusYorum_clean_numeric.csv')

# RETFound CFP weights from HuggingFace
MODEL_REPO = "YukunZhou/RETFound_mae_natureCFP"

ALL_PAT_SHORT = ['DKS', 'ODB', 'VI', 'MÖ', 'DDB', 'RI', 'HEM', 'PVK', 'RSLD', 'GV']
ALL_PAT_COLS = [
    'Diffüz kapiller sızıntı', 'Optik disk boyanması', 'Vitreus inflamasyonu',
    'Makula ödemi', 'Damar duvar boyanması', 'Retinal infiltrat',
    'Hemoraji', 'Perivasküler kılıflanma', 'Retina sinir lif defekti', 'Ghost vessel'
]

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
print("📂 Loading dataset...")
df = pd.read_csv(DATASET_CSV, encoding='utf-8')
df.columns = [c.strip().replace('\xa0', '') for c in df.columns]
df['patient_id'] = df['Klasör'].astype(str)
df['image_name'] = df['Dosya ismi'].astype(str)
df['image_path'] = df.apply(
    lambda r: os.path.join(DATA_ROOT, r['patient_id'], r['image_name']), axis=1)
for c in ALL_PAT_COLS:
    df[c] = df[c].astype(int)
df['n_pathology'] = df[ALL_PAT_COLS].sum(axis=1)
df_pat = df[df['n_pathology'] > 0].reset_index(drop=True)
df_pat = df_pat[df_pat['image_path'].apply(os.path.exists)].reset_index(drop=True)
print(f"   Total: {len(df)} images, Pathological: {len(df_pat)}, Normal: {len(df)-len(df_pat)}")

# ─────────────────────────────────────────────
# DOWNLOAD RETFound checkpoint
# ─────────────────────────────────────────────
print(f"\n📥 Downloading RETFound-MAE-CFP weights...")
# HF_TOKEN koda gömülmez! Çalıştırmadan önce ortam değişkeni olarak ayarla:
#   Linux/Mac:  export HF_TOKEN=hf_YENI_TOKEN
#   Windows:    setx HF_TOKEN hf_YENI_TOKEN   (sonra yeni terminal aç)
assert os.environ.get("HF_TOKEN"), "HF_TOKEN ortam degiskeni ayarli degil!"

# RETFound stores weights as RETFound_mae_natureCFP.pth
try:
    ckpt_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="RETFound_mae_natureCFP.pth",
    )
    print(f"   ✅ Checkpoint: {ckpt_path}")
except Exception as e:
    # Fallback: try other common filenames
    print(f"   ⚠️ Default name failed: {e}")
    print(f"   Trying alternative filenames...")
    for fn in ["RETFound_cfp_weights.pth", "RETFound_mae_meh_natureCFP.pth", 
               "checkpoint.pth", "model.pth"]:
        try:
            ckpt_path = hf_hub_download(repo_id=MODEL_REPO, filename=fn)
            print(f"   ✅ Found: {fn}")
            break
        except Exception:
            continue
    else:
        raise RuntimeError("Could not find RETFound checkpoint file in repo")

# ─────────────────────────────────────────────
# BUILD ViT-Large/16 (RETFound architecture)
# ─────────────────────────────────────────────
print(f"\n🧠 Building ViT-Large/16 (RETFound architecture)...")

# RETFound uses standard MAE ViT-Large/16 with image_size=224
# We'll use timm to construct it
try:
    import timm
    model = timm.create_model(
        'vit_large_patch16_224',
        pretrained=False,
        num_classes=0,  # Remove classification head, return embedding
        global_pool='token',  # Use CLS token
    )
    print(f"   ✅ Built ViT-Large/16 via timm")
except ImportError:
    raise ImportError("Please install timm: pip install timm")

# ─────────────────────────────────────────────
# LOAD RETFound weights
# ─────────────────────────────────────────────
print(f"\n📦 Loading RETFound weights...")
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

# RETFound checkpoint format: typically {'model': state_dict} or just state_dict
if isinstance(ckpt, dict):
    if 'model' in ckpt:
        state_dict = ckpt['model']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
else:
    state_dict = ckpt

# Remove MAE-specific keys (decoder, mask_token) that timm ViT doesn't have
keys_to_remove = []
for k in state_dict.keys():
    if any(prefix in k for prefix in ['decoder', 'mask_token', 'norm_pix_loss']):
        keys_to_remove.append(k)
for k in keys_to_remove:
    del state_dict[k]
print(f"   Removed {len(keys_to_remove)} MAE decoder keys")

# Try loading with strict=False to see what matches
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"   Missing keys (in model, not in ckpt): {len(missing)}")
print(f"   Unexpected keys (in ckpt, not in model): {len(unexpected)}")
if len(missing) > 0:
    print(f"   First few missing: {missing[:5]}")
if len(unexpected) > 0:
    print(f"   First few unexpected: {unexpected[:5]}")

# Critical check: should have transferred most of encoder weights
n_loaded = len(state_dict) - len(unexpected)
n_total = sum(1 for _ in model.state_dict().keys())
load_ratio = n_loaded / n_total
print(f"   Loaded {n_loaded}/{n_total} weights ({load_ratio*100:.1f}%)")

if load_ratio < 0.5:
    print(f"   ⚠️ WARNING: Less than 50% weights loaded — RETFound may not be functional!")
else:
    print(f"   ✅ Weights loaded successfully")

model = model.cuda().eval()
print(f"   ✅ Model on GPU. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# Test forward pass
with torch.no_grad():
    test_input = torch.randn(1, 3, 224, 224).cuda()
    test_out = model(test_input)
    print(f"   Test forward: input={tuple(test_input.shape)} → output={tuple(test_out.shape)}")

EMBEDDING_DIM = test_out.shape[-1]
print(f"   Embedding dim: {EMBEDDING_DIM}")

# ─────────────────────────────────────────────
# IMAGE PREPROCESSING (RETFound standard: 224×224, ImageNet norm)
# ─────────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

transform = T.Compose([
    T.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ─────────────────────────────────────────────
# CORE EMBEDDING FUNCTION
# ─────────────────────────────────────────────
def extract_retfound_embedding(img_path):
    """Extract single embedding from RETFound encoder."""
    img = Image.open(img_path).convert("RGB")
    pixel_values = transform(img).unsqueeze(0).cuda()  # (1, 3, 224, 224)

    with torch.no_grad():
        emb = model(pixel_values)  # (1, 1024)

    emb = emb[0].cpu().float().numpy()
    norm = np.linalg.norm(emb)
    if norm > 1e-8:
        emb = emb / norm
    return emb


# ═══════════════════════════════════════════════════════════════════
# CELL 9: BATCH EMBEDDING EXTRACTION — 561 PAT
# ═══════════════════════════════════════════════════════════════════
SAVE_DIR = os.path.join(DATA_ROOT, 'teacher_embeddings_retfound_cfp')
os.makedirs(SAVE_DIR, exist_ok=True)

EMB_PATH  = os.path.join(SAVE_DIR, 'teacher_embeddings.npy')
META_PATH = os.path.join(SAVE_DIR, 'teacher_metadata.csv')

df_all = df_pat.reset_index(drop=True)
print(f"\n{'='*60}")
print(f"📂 Total images: {len(df_all)} (pathological only)")
print(f"📂 Save dir: {SAVE_DIR}")
print(f"   Estimated time: ~{len(df_all) * 0.3 / 60:.0f} min (encoder-only is fast)")
print(f"{'='*60}\n")

# Test on first image
print("🧪 Testing extraction on first image...")
test_row = df_all.iloc[0]
test_emb = extract_retfound_embedding(test_row['image_path'])
print(f"   ✅ Test passed! emb={test_emb.shape}, norm={np.linalg.norm(test_emb):.4f}")
print(f"   First 5 values: {test_emb[:5]}\n")

# Batch extraction
new_metas, new_embs = [], []
errors = 0
t0 = time.time()

for i, row in df_all.iterrows():
    img_path = row['image_path']
    img_name = os.path.basename(img_path)
    patient  = row['patient_id']

    gt_dict = {sn: int(row[col]) if col in row.index else 0
               for col, sn in zip(ALL_PAT_COLS, ALL_PAT_SHORT)}
    gt_str  = "+".join([s for s in ALL_PAT_SHORT if gt_dict[s] == 1])

    try:
        emb = extract_retfound_embedding(img_path)
        new_embs.append(emb)
        meta = {
            'image_path': img_path, 'image_name': img_name,
            'patient_id': patient,  'gt_labels': gt_str,
        }
        for sn in ALL_PAT_SHORT:
            meta[f'gt_{sn}'] = gt_dict[sn]
        new_metas.append(meta)

        if (i+1) % 50 == 0:
            elapsed = (time.time() - t0) / 60
            eta = elapsed / (i+1) * (len(df_all) - i - 1)
            print(f"  [{i+1:4d}/{len(df_all)}] {img_name:30s} [{gt_str:15s}] "
                  f"({elapsed:.1f}m, ETA {eta:.1f}m)")

    except Exception as e:
        errors += 1
        print(f"  [{i+1:4d}/{len(df_all)}] {img_name:30s} ⚠️ {str(e)[:80]}")

# Save final
np.save(EMB_PATH, np.array(new_embs))
pd.DataFrame(new_metas).to_csv(META_PATH, index=False, encoding='utf-8-sig')

elapsed = (time.time() - t0) / 60
print(f"\n{'='*60}")
print(f"✅ EMBEDDING EXTRACTION COMPLETE (RETFound-CFP)")
print(f"   Total: {len(new_embs)}/{len(df_all)} images")
print(f"   Errors: {errors}")
print(f"   Time: {elapsed:.1f} min")
print(f"   Embeddings: {EMB_PATH}")
print(f"   Metadata:   {META_PATH}")

if os.path.exists(EMB_PATH):
    embs = np.load(EMB_PATH)
    meta = pd.read_csv(META_PATH)
    print(f"\n   Verify: {embs.shape[0]} embeddings × {embs.shape[1]} dim")
    print(f"   Label distribution:")
    for sn in ALL_PAT_SHORT:
        n = int(meta[f'gt_{sn}'].sum())
        if n > 0:
            print(f"     {sn:6s}: {n:4d}")
print(f"{'='*60}")

# ============================================================================
# CELL 8b-BIOMEDCLIP: SETUP + BATCH EXTRACTION
# ============================================================================
# Microsoft BiomedCLIP — Vision-Language Contrastive (encoder-only)
# Vision: ViT-Base/16, 512-dim image embedding
# No prompts, no references — just raw image → embedding (vision tower)
# ============================================================================

import os, time
import torch
import numpy as np
import pandas as pd
from PIL import Image

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_ROOT = r'C:\Users\gtu\Documents\cerrahpasa\files\files'
DATASET_CSV = os.path.join(DATA_ROOT, 'Resimler_birlesik_TF_plusYorum_clean_numeric.csv')

# BiomedCLIP from HuggingFace (Microsoft)
MODEL_NAME = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

ALL_PAT_SHORT = ['DKS', 'ODB', 'VI', 'MÖ', 'DDB', 'RI', 'HEM', 'PVK', 'RSLD', 'GV']
ALL_PAT_COLS = [
    'Diffüz kapiller sızıntı', 'Optik disk boyanması', 'Vitreus inflamasyonu',
    'Makula ödemi', 'Damar duvar boyanması', 'Retinal infiltrat',
    'Hemoraji', 'Perivasküler kılıflanma', 'Retina sinir lif defekti', 'Ghost vessel'
]
# HF_TOKEN koda gömülmez! Çalıştırmadan önce ortam değişkeni olarak ayarla:
#   Linux/Mac:  export HF_TOKEN=hf_YENI_TOKEN
#   Windows:    setx HF_TOKEN hf_YENI_TOKEN   (sonra yeni terminal aç)
assert os.environ.get("HF_TOKEN"), "HF_TOKEN ortam degiskeni ayarli degil!"

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
# LOAD BIOMEDCLIP — try open_clip first, then transformers fallback
# ─────────────────────────────────────────────
print(f"\n🧠 Loading BiomedCLIP...")

model = None
preprocess = None
USE_OPEN_CLIP = False

try:
    # Method 1: open_clip (BiomedCLIP'in resmi yöntemi)
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        f"hf-hub:{MODEL_NAME}"
    )
    model = model.cuda().eval()
    USE_OPEN_CLIP = True
    print(f"   ✅ Loaded via open_clip")
except ImportError:
    print(f"   ⚠️ open_clip not installed, trying transformers...")
    try:
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(MODEL_NAME).cuda().eval()
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        print(f"   ✅ Loaded via transformers")
    except Exception as e:
        print(f"   ❌ transformers also failed: {e}")
        raise RuntimeError(
            "BiomedCLIP loading failed. Install open_clip:\n"
            "  pip install open_clip_torch\n"
            "Then retry."
        )
except Exception as e:
    print(f"   ⚠️ open_clip error: {e}")
    print(f"   Trying transformers...")
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(MODEL_NAME).cuda().eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    print(f"   ✅ Loaded via transformers")

print(f"   VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# Test forward to determine embedding dim
with torch.no_grad():
    test_img = Image.new("RGB", (224, 224), color=128)
    if USE_OPEN_CLIP:
        test_input = preprocess(test_img).unsqueeze(0).cuda()
        test_emb = model.encode_image(test_input)
    else:
        test_inputs = processor(images=test_img, return_tensors="pt").to("cuda")
        test_emb = model.get_image_features(**test_inputs)

EMBEDDING_DIM = test_emb.shape[-1]
print(f"   Test forward: output={tuple(test_emb.shape)}, embedding dim={EMBEDDING_DIM}")

# ─────────────────────────────────────────────
# CORE EMBEDDING FUNCTION
# ─────────────────────────────────────────────
def extract_biomedclip_embedding(img_path):
    """Extract image embedding from BiomedCLIP vision tower."""
    img = Image.open(img_path).convert("RGB")

    with torch.no_grad():
        if USE_OPEN_CLIP:
            pixel_values = preprocess(img).unsqueeze(0).cuda()
            emb = model.encode_image(pixel_values)
        else:
            inputs = processor(images=img, return_tensors="pt").to("cuda")
            emb = model.get_image_features(**inputs)

    emb = emb[0].cpu().float().numpy()
    norm = np.linalg.norm(emb)
    if norm > 1e-8:
        emb = emb / norm
    return emb


# ═══════════════════════════════════════════════════════════════════
# CELL 9: BATCH EMBEDDING EXTRACTION — 561 PAT
# ═══════════════════════════════════════════════════════════════════
SAVE_DIR = os.path.join(DATA_ROOT, 'teacher_embeddings_biomedclip')
os.makedirs(SAVE_DIR, exist_ok=True)

EMB_PATH  = os.path.join(SAVE_DIR, 'teacher_embeddings.npy')
META_PATH = os.path.join(SAVE_DIR, 'teacher_metadata.csv')

df_all = df_pat.reset_index(drop=True)
print(f"\n{'='*60}")
print(f"📂 Total images: {len(df_all)} (pathological only)")
print(f"📂 Save dir: {SAVE_DIR}")
print(f"   Estimated time: ~{len(df_all) * 0.3 / 60:.0f} min (encoder-only)")
print(f"{'='*60}\n")

# Sanity test
print("🧪 Testing extraction on first image...")
test_row = df_all.iloc[0]
test_emb = extract_biomedclip_embedding(test_row['image_path'])
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
        emb = extract_biomedclip_embedding(img_path)
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
print(f"✅ EMBEDDING EXTRACTION COMPLETE (BiomedCLIP)")
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

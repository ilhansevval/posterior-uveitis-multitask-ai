# ============================================================================
# CELL 8b-MG: MEDGEMMA TEACHER EMBEDDING (NO GT GUIDANCE) + REFERENCE IMAGES
# ============================================================================
# MedGemma 4B-it with few-shot reference gallery from fundus_choosen/
# 6 labels with 2 reference images, 2 labels text-only (DKS, VI)
# The model sees a reference gallery of ALL labeled findings but is NOT told
# which are present — it decides on its own. GT is used ONLY for evaluation.
# Embedding: last 30 tokens, L2 normalized
# ============================================================================

import os, glob
import torch
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_ROOT = r'C:\Users\gtu\Documents\cerrahpasa\files\files'
DATASET_CSV = os.path.join(DATA_ROOT, 'Resimler_birlesik_TF_plusYorum_clean_numeric.csv')
REF_ROOT = os.path.join(DATA_ROOT, 'fundus_choosen')
MODEL_PATH = r'C:\Users\gtu\models\medgemma-4b-it'

# ─────────────────────────────────────────────
# LABEL MAPPINGS
# ─────────────────────────────────────────────
ALL_PAT_SHORT = ['DKS', 'ODB', 'VI', 'MÖ', 'DDB', 'RI', 'HEM', 'PVK', 'RSLD', 'GV']
ALL_PAT_COLS = [
    'Diffüz kapiller sızıntı', 'Optik disk boyanması', 'Vitreus inflamasyonu',
    'Makula ödemi', 'Damar duvar boyanması', 'Retinal infiltrat',
    'Hemoraji', 'Perivasküler kılıflanma', 'Retina sinir lif defekti', 'Ghost vessel'
]

# Reference image folder names → label mapping
REF_FOLDER_MAP = {
    'ODB': 'Optik disk boyanması',
    'MÖ': 'makula ödemi',
    'DDB': 'damar duvar',
    'RI': 'Retinal infiltrat',
    'HEM': 'hemoraji',
    'PVK': 'Perivasküler kılıflanma',
}
# DKS and VI have no reference images (global findings)

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
print(f"   Total: {len(df)} images, Pathological: {len(df_pat)}, Normal: {len(df) - len(df_pat)}")

# ─────────────────────────────────────────────
# LOAD REFERENCE IMAGES
# ─────────────────────────────────────────────
print("\n📸 Loading reference images...")
ref_images = {}
for label_short, folder_name in REF_FOLDER_MAP.items():
    folder_path = os.path.join(REF_ROOT, folder_name)
    if not os.path.exists(folder_path):
        print(f"   ⚠️ {label_short}: folder not found: {folder_path}")
        continue
    imgs = sorted(glob.glob(os.path.join(folder_path, '*.jpg')) +
                  glob.glob(os.path.join(folder_path, '*.png')) +
                  glob.glob(os.path.join(folder_path, '*.jpeg')))
    # Take first 2
    ref_images[label_short] = [Image.open(p).convert("RGB") for p in imgs[:2]]
    print(f"   {label_short}: {len(ref_images[label_short])} reference images from {folder_name}/")

# ─────────────────────────────────────────────
# LOAD MEDGEMMA
# ─────────────────────────────────────────────
print(f"\n🧠 Loading MedGemma 4B-it (4-bit)...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
vl_model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
vl_processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"   ✅ Model loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

# ─────────────────────────────────────────────
# LABEL DEFINITIONS (concise for MedGemma)
# ─────────────────────────────────────────────
LABEL_DEFS = {
    "DKS": "Diffuse capillary leakage: widespread foggy brightness BETWEEN vessels across >1/3 of image. NOT normal background variation.",
    "ODB": "Optic disc staining: disc ABNORMALLY bright with blurred/washed-out edges. Normal disc is bright but with sharp margins.",
    "VI": "Vitreous inflammation: ENTIRE image is hazy/foggy with globally reduced clarity. NOT regional — must be uniform across whole image. Look at overall image clarity, not local areas.",
    "MÖ": "Macular edema: flower-petal or star-shaped brightness at image CENTER (macula). Normal fovea is dark.",
    "DDB": (
        "Vessel wall staining: bright glow ALONG vessel walls (outside the lumen), making vessels appear "
        "outlined, double-lined, or thicker than normal with a halo/sheathing effect. "
        "CRITICAL DISTINCTION: Normal vessels are bright INSIDE from blood flow — this is NOT DDB. "
        "DDB means the WALL ITSELF glows — look for vessels that appear wider than normal with "
        "brightness extending BEYOND the vessel edge, creating a 'railroad track' or 'double contour' appearance. "
        "Often seen in medium-to-small vessels, not just major arcades."
    ),
    "RI": "Retinal infiltrate: dark patch with irregular/fuzzy edges surrounded by a bright halo. NOT the normal dark fovea. Usually near vessels.",
    "HEM": "Hemorrhage: distinctly BLACK areas with irregular borders that block underlying pattern completely. Flame-shaped or blot-shaped, OUT OF PLACE.",
    "PVK": "Perivascular sheathing: white opaque coating/sleeve around vessel segments. Railroad-track appearance with opaque white on both sides.",
    "RSLD": "Nerve fiber layer defect: wedge/arc-shaped dark gap near optic disc.",
    "GV": "Ghost vessel: vessel traces visible but EMPTY — no dye filling. Like abandoned roads.",
}

COOCCURRENCE_INFO = """
CO-OCCURRENCE PATTERNS:
Group A (tend together): DKS ↔ ODB ↔ MÖ ↔ DDB
Group B (tend together): VI ↔ RI ↔ HEM
Groups A and B rarely overlap.
"""

# ─────────────────────────────────────────────
# ANALYSIS PROMPT WITH REFERENCE IMAGES (NO GT GUIDANCE)
# ─────────────────────────────────────────────
def make_prompt_with_refs():
    """Build prompt text (images added separately in message content).
    The ground-truth labels are NOT included — the model decides on its own."""
    defs_text = "\n".join(
        [f"  - {sn}: {LABEL_DEFS.get(sn, sn)}" for sn in ALL_PAT_SHORT])

    prompt = f"""Analyze this retinal fluorescein angiography (FA) PATIENT image.
Identify any pathological findings present, using ONLY the visible image evidence.
Reference example images (shown above) illustrate what some findings look like —
use them as a visual guide, NOT as an indication of what is present in this patient.

POSSIBLE FINDINGS (look for visual evidence of each):
{defs_text}

{COOCCURRENCE_INFO}

For EACH finding you judge PRESENT: describe specific abnormal visual evidence, location, why pathological.
For EACH finding you judge ABSENT: explain what would be visible if present, confirm normal anatomy.
State for each: "This is NORMAL/PATHOLOGICAL because..."
Conclude with a short list of the findings you judge PRESENT (or "Normal" if none)."""

    return prompt


def build_message_content(patient_img, image_size=512):
    """Build message content with reference gallery + patient image.
    Reference images for ALL labeled findings are shown — NOT only the
    ground-truth-positive ones — so the gallery does not leak the answer."""
    content = []

    # Reference gallery: every label that has reference images
    for sn in ALL_PAT_SHORT:
        if sn in ref_images:
            content.append({"type": "text", "text": f"\n[REFERENCE: {sn} — {LABEL_DEFS[sn]}]\nThe following annotated images show example {sn} findings (arrows/marks indicate the pathology):"})
            for ref_img in ref_images[sn]:
                ref_resized = ref_img.resize((image_size, image_size))
                content.append({"type": "image", "image": ref_resized})

    # Add patient image
    content.append({"type": "text", "text": "\n[PATIENT IMAGE — analyze this image and identify which findings are present]:"})
    patient_resized = Image.open(patient_img).convert("RGB").resize((image_size, image_size))
    content.append({"type": "image", "image": patient_resized})

    # Add prompt text
    prompt_text = make_prompt_with_refs()
    content.append({"type": "text", "text": prompt_text})

    return content


SYSTEM_PROMPT = (
    "You are a medical ophthalmology AI specialized in retinal fluorescein "
    "angiography (FA) interpretation. You are shown a gallery of reference images "
    "illustrating annotated pathological findings, followed by a patient image to "
    "analyze. Identify which findings are present in the patient image and explain "
    "the visual evidence thoroughly."
)

# ─────────────────────────────────────────────
# EMBEDDING EXTRACTION
# ─────────────────────────────────────────────
def medgemma_embedding(img_path, image_size=512, max_new_tokens=1500):
    """Extract embedding from MedGemma with reference gallery (no GT guidance)"""
    content = build_message_content(img_path, image_size)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]

    # Collect all images for processor
    all_images = [item["image"] for item in content if item.get("type") == "image"]

    inputs = vl_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(vl_model.device, dtype=torch.bfloat16)

    torch.cuda.empty_cache()
    input_len = inputs['input_ids'].shape[-1]

    with torch.inference_mode():
        gen_output = vl_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    # Last 30 generated tokens, last layer
    gen_embeddings = []
    for step_hidden in gen_output.hidden_states:
        last_layer = step_hidden[-1]
        last_token = last_layer[0, -1, :]
        gen_embeddings.append(last_token.cpu().float())

    gen_embeddings = torch.stack(gen_embeddings)
    n_use = min(30, len(gen_embeddings))
    emb = gen_embeddings[-n_use:].mean(dim=0).numpy()
    emb = emb / (np.linalg.norm(emb) + 1e-8)

    # Explanation
    generated_ids = gen_output.sequences[0][input_len:]
    explanation = vl_processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return emb, explanation, len(gen_embeddings)


# ═══════════════════════════════════════════════════════════════════
# TEST: 10 pathological + 5 normal (same as Qwen test)
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("🔍 MEDGEMMA TEACHER EMBEDDING (NO GT GUIDANCE) — TEST (10 PAT + 5 NOR)")
print(f"{'='*80}")

test_gt = []
used_paths = set()

# 2 per main label (DKS, ODB, VI, MÖ, DDB)
for col, sn in zip(ALL_PAT_COLS[:5], ALL_PAT_SHORT[:5]):
    count = 0
    for _, row in df_pat[df_pat[col] == 1].iterrows():
        if row['image_path'] not in used_paths and count < 2:
            test_gt.append(("PAT", row, sn))
            used_paths.add(row['image_path'])
            count += 1

# 5 normal
for _, row in df[df["n_pathology"] == 0].sample(5, random_state=42).iterrows():
    test_gt.append(("NOR", row, "Normal"))

embeddings = []
labels = []
primary_labels = []
img_names = []
all_gt_dicts = []

for idx, (group, row, primary) in enumerate(test_gt):
    # gt_dict is kept ONLY for display / evaluation — it is NOT fed to the model
    gt_dict = {sn: int(row[col]) if col in row.index else 0
               for col, sn in zip(ALL_PAT_COLS, ALL_PAT_SHORT)}
    gt_str = "+".join([s for s in ALL_PAT_SHORT if gt_dict[s] == 1]) or "Normal"
    img_name = os.path.basename(row["image_path"])

    # Reference gallery is the same for every image (all labeled findings shown)
    has_refs = len(ref_images)

    print(f"\n{'█'*80}")
    print(f"  [{idx+1:2d}/{len(test_gt)}] {img_name}")
    print(f"  {group} | Primary: {primary} | GT: [{gt_str}] | Gallery: {has_refs} labels with images")
    print(f"{'█'*80}")

    emb, expl, n_gen = medgemma_embedding(row["image_path"])

    embeddings.append(emb)
    labels.append(group)
    primary_labels.append(primary)
    img_names.append(img_name)
    all_gt_dicts.append(gt_dict)

    print(f"  Embedding: {emb.shape}, norm={np.linalg.norm(emb):.4f}, gen={n_gen} tokens")
    print(f"  Explanation ({len(expl)} chars):")
    for line in expl[:500].split('\n'):
        if line.strip():
            print(f"    {line.strip()[:120]}")

emb_array = np.array(embeddings)
n_pat = sum(1 for l in labels if l == "PAT")
n_nor = len(labels) - n_pat

# ═══════════════════════════════════════════════════════════════
# ANALYSIS (same as Qwen version)
# ═══════════════════════════════════════════════════════════════
sim = cosine_similarity(emb_array)

print(f"\n{'='*80}")
print("📊 PAT vs NOR SEPARATION")
print(f"{'='*80}")

pp = sim[:n_pat, :n_pat].copy(); np.fill_diagonal(pp, np.nan)
nn = sim[n_pat:, n_pat:].copy(); np.fill_diagonal(nn, np.nan)
pn = sim[:n_pat, n_pat:]
mean_pp = np.nanmean(pp); mean_nn = np.nanmean(nn); mean_pn = np.nanmean(pn)

print(f"  PAT↔PAT: {mean_pp:.4f} | NOR↔NOR: {mean_nn:.4f} | PAT↔NOR: {mean_pn:.4f}")
print(f"  Gap PP: {mean_pp - mean_pn:.4f} | Gap NN: {mean_nn - mean_pn:.4f}")

# Centroid
pat_c = emb_array[:n_pat].mean(axis=0); pat_c /= (np.linalg.norm(pat_c) + 1e-8)
nor_c = emb_array[n_pat:].mean(axis=0); nor_c /= (np.linalg.norm(nor_c) + 1e-8)

print(f"\n{'='*80}")
print("📊 CENTROID ACCURACY")
print(f"{'='*80}")

n_correct = 0
for i in range(len(embeddings)):
    sp = cosine_similarity(emb_array[i:i+1], pat_c.reshape(1,-1))[0,0]
    sn = cosine_similarity(emb_array[i:i+1], nor_c.reshape(1,-1))[0,0]
    closer = "PAT" if sp > sn else "NOR"
    ok = closer == labels[i]
    if ok: n_correct += 1
    gt_str = "+".join([s for s in ALL_PAT_SHORT if all_gt_dicts[i].get(s,0)==1]) or "Normal"
    print(f"  {img_names[i]:30s} {labels[i]:4s} [{gt_str:15s}] →PAT={sp:.4f} →NOR={sn:.4f} {closer} {'✓' if ok else '✗'}")

print(f"\n  Centroid: {n_correct}/{len(embeddings)} ({n_correct/len(embeddings)*100:.0f}%)")

# Label overlap
print(f"\n{'='*80}")
print("📊 LABEL OVERLAP")
print(f"{'='*80}")
ol, nol = [], []
for i, j in combinations(range(len(embeddings)), 2):
    shared = sum(1 for s in ALL_PAT_SHORT if all_gt_dicts[i].get(s,0)==1 and all_gt_dicts[j].get(s,0)==1)
    s_val = sim[i,j]
    if shared > 0: ol.append(s_val)
    elif sum(all_gt_dicts[i].values()) > 0 or sum(all_gt_dicts[j].values()) > 0: nol.append(s_val)
if ol and nol:
    print(f"  Overlap: {np.mean(ol):.4f} (n={len(ol)}) | No overlap: {np.mean(nol):.4f} (n={len(nol)})")
    print(f"  Gap: {np.mean(ol)-np.mean(nol):.4f}")

# Summary
print(f"\n{'='*80}")
print("📋 SUMMARY — MedGemma vs Qwen comparison")
print(f"{'='*80}")
print(f"  MedGemma: dim={emb_array.shape[1]}, gap_PP={mean_pp-mean_pn:.4f}, centroid={n_correct}/{len(embeddings)}")
print(f"  Qwen ref: dim=2048, gap_PP=0.0396, centroid=10/10")
print(f"  {'✅ MedGemma BETTER' if (mean_pp-mean_pn) > 0.0396 else '🔄 Compare embedding dims'}")

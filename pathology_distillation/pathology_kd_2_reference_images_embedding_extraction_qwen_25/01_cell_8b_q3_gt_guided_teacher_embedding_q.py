# ============================================================================
# CELL 8b-Q3: GT-GUIDED TEACHER EMBEDDING — Qwen3-VL-8B + Reference Images
# ============================================================================
# Same structure as original Cell 8b (5 PAT + 5 NOR, Part 1-4 analysis)
# Changes: Qwen3-VL-8B model, reference images from fundus_choosen/
# Prompts identical to original
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
MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"
# ─────────────────────────────────────────────
# LABEL MAPPINGS
# ─────────────────────────────────────────────
ALL_PAT_SHORT = ['DKS', 'ODB', 'VI', 'MÖ', 'DDB', 'RI', 'HEM', 'PVK', 'RSLD', 'GV']
ALL_PAT_COLS = [
    'Diffüz kapiller sızıntı', 'Optik disk boyanması', 'Vitreus inflamasyonu',
    'Makula ödemi', 'Damar duvar boyanması', 'Retinal infiltrat',
    'Hemoraji', 'Perivasküler kılıflanma', 'Retina sinir lif defekti', 'Ghost vessel'
]

# Reference folders (DKS and VI have no refs — global findings)
REF_FOLDER_MAP = {
    'ODB': 'Optik disk boyanması',
    'MÖ': 'makula ödemi',
    'DDB': 'damar duvar',
    'RI': 'Retinal infiltrat',
    'HEM': 'hemoraji',
    'PVK': 'Perivasküler kılıflanma',
}

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
    ref_images[label_short] = [Image.open(p).convert("RGB") for p in imgs[:2]]
    print(f"   {label_short}: {len(ref_images[label_short])} refs from {folder_name}/")

# ─────────────────────────────────────────────
# LOAD QWEN3-VL-8B
# ─────────────────────────────────────────────
print(f"\n🧠 Loading Qwen3-VL-8B-Instruct (4-bit)...")
# HF_TOKEN koda gömülmez! Çalıştırmadan önce ortam değişkeni olarak ayarla:
#   Linux/Mac:  export HF_TOKEN=hf_YENI_TOKEN
#   Windows:    setx HF_TOKEN hf_YENI_TOKEN   (sonra yeni terminal aç)
assert os.environ.get("HF_TOKEN"), "HF_TOKEN ortam degiskeni ayarli degil!"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
vl_model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
vl_processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"   ✅ Model loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

# ─────────────────────────────────────────────
# LABEL DEFINITIONS (identical to original)
# ─────────────────────────────────────────────
LABEL_DEFS = {
    "DKS": (
        "Diffuse capillary leakage — widespread foggy brightness BETWEEN vessels across a large area. "
        "NORMAL ANATOMY NOTE: Some background brightness is normal. PATHOLOGICAL only if the brightness "
        "is widespread (covers >1/3 of image), has fuzzy borders, and reduces contrast significantly. "
        "Compare peripheral background to areas near disc — if both are hazy, DKS is likely."
    ),
    "ODB": (
        "Optic disc staining — the disc is ABNORMALLY bright with blurred/washed-out edges. "
        "NORMAL ANATOMY NOTE: The optic disc is ALWAYS somewhat bright in FA — this is NORMAL. "
        "A normal disc has clear sharp margins and moderate brightness. PATHOLOGICAL only if the disc "
        "is dramatically brighter than surrounding retina AND its edges are obscured by the brightness. "
        "Do NOT call normal disc brightness as ODB."
    ),
    "VI": (
        "Vitreous inflammation — the ENTIRE image is hazy/foggy with globally reduced clarity. "
        "NORMAL ANATOMY NOTE: Some FA images have slightly lower contrast due to technique. "
        "PATHOLOGICAL only if vessel edges and disc borders that should be sharp appear blurry, "
        "AND the haziness is uniform across the whole image, not just one region."
    ),
    "MÖ": (
        "Macular edema — flower-petal or star-shaped brightness at the image CENTER (macula). "
        "NORMAL ANATOMY NOTE: The macula center (fovea) is normally DARK due to avascular zone. "
        "PATHOLOGICAL only if you see abnormal brightness specifically at the macula — petal pattern, "
        "central glow, or bright cysts clustered at the center."
    ),
    "DDB": (
        "Vessel wall staining — bright glow along vessel WALLS (outside the lumen). "
        "NORMAL ANATOMY NOTE: Vessels are normally bright INSIDE because of blood flow. "
        "PATHOLOGICAL only if you see brightness OUTSIDE/ALONG the vessel walls, making vessels "
        "appear outlined, double-lined, or thicker than normal with a halo effect."
    ),
    "RI": (
        "Retinal infiltrate — a dark patch with irregular/fuzzy edges surrounded by a bright halo. "
        "NORMAL ANATOMY NOTE: The fovea is normally dark — do NOT confuse with infiltrate. "
        "PATHOLOGICAL only if a dark area has IRREGULAR borders AND a surrounding bright ring/halo. "
        "Usually found near vessels, not at the foveal center."
    ),
    "HEM": (
        "Hemorrhage — distinctly BLACK areas that block the underlying pattern completely. "
        "NORMAL ANATOMY NOTE: The fovea and some peripheral areas can be dark normally. "
        "PATHOLOGICAL only if dark areas have IRREGULAR borders, are flame-shaped or blot-shaped, "
        "and appear OUT OF PLACE — interrupting the normal vessel/background pattern."
    ),
    "PVK": (
        "Perivascular sheathing — white opaque coating/sleeve around vessel segments. "
        "NORMAL ANATOMY NOTE: Vessels have some natural wall visibility. "
        "PATHOLOGICAL only if you see a distinct white COATING wrapping around a vessel segment, "
        "creating a railroad-track appearance with opaque white on both sides."
    ),
    "RSLD": (
        "Nerve fiber layer defect — wedge/arc-shaped dark gap near the optic disc. "
        "NORMAL ANATOMY NOTE: Area around disc can have variable brightness. "
        "PATHOLOGICAL only if one sector around the disc is distinctly darker than adjacent sectors, "
        "forming a wedge or arc shape pointing away from the disc."
    ),
    "GV": (
        "Ghost vessel — vessel traces visible but EMPTY (no dye filling). "
        "NORMAL ANATOMY NOTE: All normal vessels should be bright (filled with dye). "
        "PATHOLOGICAL only if you see vessel outlines that are faint/transparent with no bright "
        "dye inside — like abandoned roads. Nearby vessels may be abnormally dilated."
    ),
}

COOCCURRENCE_INFO = """
CLINICAL CO-OCCURRENCE PATTERNS (use to validate your observations):
Group A (tend to appear together): DKS ↔ ODB ↔ MÖ ↔ DDB
  - When ODB present → MÖ also present 58% of the time
  - When ODB present → DDB also present 34% of the time
  - When DKS present → ODB also present 14% of the time
Group B (tend to appear together): VI ↔ RI ↔ HEM
  - When RI present → HEM also present 31% of the time (54x more likely)
  - When HEM present → RI also present 57% of the time (86x more likely)
  - When VI present → RI 4x more likely, HEM 6x more likely
Groups A and B RARELY overlap.
"""

GT_SYSTEM = (
    "You are a medical vision-language assistant specialized in retinal fluorescein "
    "angiography interpretation. The diagnoses are already confirmed by an expert. "
    "Your job is to explain the visual evidence that supports present findings and "
    "the missing visual evidence for absent findings. Be thorough and specific."
)

# ─────────────────────────────────────────────
# GT-GUIDED PROMPT (identical to original)
# ─────────────────────────────────────────────
def make_gt_prompt(gt_dict, max_neg=7):
    pos = [sn for sn, v in gt_dict.items() if v == 1]
    neg = [sn for sn, v in gt_dict.items() if v == 0]
    neg = neg[:max_neg]

    pos_text = "\n".join([f"  - {sn}: {LABEL_DEFS.get(sn, sn)}" for sn in pos]) if pos else "  - None (normal image)"
    neg_text = "\n".join([f"  - {sn}: {LABEL_DEFS.get(sn, sn)}" for sn in neg]) if neg else "  - None"

    prompt = f"""You are analyzing a retinal fluorescein angiography (FA) image.
The diagnosis is ALREADY CONFIRMED by an expert ophthalmologist.
Your task is NOT to diagnose — it is to explain the visible evidence.

CRITICAL RULE — NORMAL vs PATHOLOGICAL:
Many structures in the eye are NORMALLY bright or dark in FA images:
- The OPTIC DISC is naturally bright — this alone is NOT pathological
- The FOVEA (macula center) is naturally dark — this alone is NOT hemorrhage
- Vessels are naturally bright inside — this alone is NOT vessel wall staining
- Some background brightness variation is normal — this alone is NOT leakage
ONLY mark something as pathological if it is ABNORMAL compared to what 
a healthy eye would show. Each label definition includes what is normal 
vs what is pathological — follow those guidelines strictly.

Image characteristics:
- Grayscale or green-tinted medical image
- Bright areas = dye leaking or accumulating
- Dark areas = something blocking the dye
- Blood vessels = bright branching lines
- Optic disc = round bright structure where vessels converge (NORMALLY bright)
- Macula = center area (NORMALLY darker)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIRMED PRESENT findings:
{pos_text}

CONFIRMED ABSENT findings:
{neg_text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{COOCCURRENCE_INFO}

INSTRUCTIONS:
1. For EACH confirmed PRESENT finding:
   - Describe the specific ABNORMAL visual evidence (not normal anatomy)
   - Mention WHERE in the image
   - Explain why this is PATHOLOGICAL and not normal anatomy

2. For EACH confirmed ABSENT finding:
   - State what ABNORMAL feature would be visible if it were present
   - Explain that you see only NORMAL anatomy in that region, not pathology

3. For each finding, explicitly state: "This is [NORMAL ANATOMY / PATHOLOGICAL] because..."

Be detailed and clinically focused. Use ONLY visible image evidence.
Distinguish normal anatomical appearances from pathological findings."""

    return prompt

# ─────────────────────────────────────────────
# EMBEDDING + EXPLANATION EXTRACTION (with refs)
# ─────────────────────────────────────────────
def qwen3_gt_embedding(img_path, gt_dict, image_size=384, max_new_tokens=1500):
    img = Image.open(img_path).convert("RGB").resize((image_size, image_size))
    prompt = make_gt_prompt(gt_dict)
    pos_labels = [sn for sn, v in gt_dict.items() if v == 1]

    # Build content: ref images + patient image + prompt
    content = []

    # Add reference images for present labels
    for sn in pos_labels:
        if sn in ref_images:
            content.append({"type": "text", "text": f"[REFERENCE — {sn} examples with annotations (arrows show pathology)]:"})
            for ref_img in ref_images[sn]:
                content.append({"type": "image", "image": ref_img.resize((image_size, image_size))})

    # Patient image
    content.append({"type": "text", "text": "[PATIENT IMAGE to analyze]:"})
    content.append({"type": "image", "image": img})

    # Prompt
    content.append({"type": "text", "text": prompt})

    messages = [
        {"role": "system", "content": GT_SYSTEM},
        {"role": "user", "content": content},
    ]

    text_input = vl_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    # Collect all PIL images
    all_images = [item["image"] for item in content if item.get("type") == "image"]

    inputs = vl_processor(
        text=[text_input], images=[all_images], padding=True, return_tensors="pt",
    ).to(vl_model.device)

    torch.cuda.empty_cache()
    input_len = inputs['input_ids'].shape[1]

    with torch.no_grad():
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

    generated_ids = gen_output.sequences[0][input_len:]
    explanation = vl_processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    if len(explanation) < 50:
        print(f"    ⚠️ Short explanation ({len(explanation)} chars)")

    return emb, explanation


# ═══════════════════════════════════════════════════════════════════
# TEST: 5 pathological + 5 normal (same as original)
# ═══════════════════════════════════════════════════════════════════
print("=" * 80)
print("🔍 QWEN2.5-VL-4B GT-GUIDED TEACHER EMBEDDING — FULL ANALYSIS")
print("=" * 80)

test_gt = []

# 5 pathological — one per label (diverse)
used_paths = set()
for col, sn in zip(ALL_PAT_COLS[:5], ALL_PAT_SHORT[:5]):
    pos = df_pat[df_pat[col] == 1]
    for _, row in pos.iterrows():
        if row['image_path'] not in used_paths:
            test_gt.append(("PAT", row, sn))
            used_paths.add(row['image_path'])
            break

# 5 normal
for _, row in df[df["n_pathology"] == 0].sample(5, random_state=42).iterrows():
    test_gt.append(("NOR", row, "Normal"))

embeddings = []
labels = []
primary_labels = []
explanations = []
img_names = []

for group, row, primary in test_gt:
    gt_dict = {sn: int(row[col]) if col in row.index else 0
               for col, sn in zip(ALL_PAT_COLS, ALL_PAT_SHORT)}
    gt_str = "+".join([s for s in ALL_PAT_SHORT if gt_dict[s] == 1]) or "Normal"
    img_name = os.path.basename(row["image_path"])

    pos_labels = [s for s in ALL_PAT_SHORT if gt_dict[s] == 1]
    n_refs = sum(1 for s in pos_labels if s in ref_images)

    print(f"\n{'─' * 80}")
    print(f"  [{group}] {img_name} — Primary: {primary} — GT: [{gt_str}] — Refs: {n_refs}")

    emb, expl = qwen3_gt_embedding(row["image_path"], gt_dict)

    embeddings.append(emb)
    labels.append(group)
    primary_labels.append(primary)
    explanations.append(expl)
    img_names.append(img_name)

    print(f"  Embedding: shape={emb.shape}, norm={np.linalg.norm(emb):.4f}")
    print(f"  Explanation ({len(expl)} chars):")
    for line in expl[:300].split('\n'):
        if line.strip():
            print(f"    {line.strip()[:100]}")

emb_array = np.array(embeddings)

# ═══════════════════════════════════════════════════════════════════
# PART 1: PAT vs NOR SEPARATION
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("📊 PART 1: PAT vs NOR EMBEDDING SEPARATION")
print(f"{'=' * 80}")

n_pat = sum(1 for l in labels if l == "PAT")
n_nor = len(labels) - n_pat

sim = cosine_similarity(emb_array)

pat_pat = sim[:n_pat, :n_pat].copy()
nor_nor = sim[n_pat:, n_pat:].copy()
pat_nor = sim[:n_pat, n_pat:]

np.fill_diagonal(pat_pat, np.nan)
np.fill_diagonal(nor_nor, np.nan)

mean_pp = np.nanmean(pat_pat)
mean_nn = np.nanmean(nor_nor)
mean_pn = np.nanmean(pat_nor)

print(f"\n  PAT ↔ PAT (intra):  {mean_pp:.4f}")
print(f"  NOR ↔ NOR (intra):  {mean_nn:.4f}")
print(f"  PAT ↔ NOR (cross):  {mean_pn:.4f}")
print(f"  Gap (PP - PN):      {mean_pp - mean_pn:.4f}")
print(f"  Gap (NN - PN):      {mean_nn - mean_pn:.4f}")

if mean_pp > mean_pn + 0.02 or mean_nn > mean_pn + 0.02:
    print(f"\n  ✅ PAT vs NOR separation detected!")
else:
    print(f"\n  ⚠️ Weak PAT vs NOR separation")

# ═══════════════════════════════════════════════════════════════════
# PART 2: NEAREST NEIGHBOR ANALYSIS
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("📊 PART 2: NEAREST NEIGHBOR — Her görüntünün en yakın/uzak komşusu")
print(f"{'=' * 80}")

for i in range(len(embeddings)):
    sims = []
    for j in range(len(embeddings)):
        if i != j:
            s = cosine_similarity(emb_array[i].reshape(1, -1), emb_array[j].reshape(1, -1))[0, 0]
            sims.append((j, s))
    sims.sort(key=lambda x: -x[1])

    gt_i = "+".join([s for col, s in zip(ALL_PAT_COLS, ALL_PAT_SHORT)
                     if col in test_gt[i][1].index and test_gt[i][1][col] == 1]) or "Normal"

    print(f"\n  {img_names[i]:30s} [{labels[i]}] GT=[{gt_i}]")
    print(f"  En yakın 3:")
    for j, s in sims[:3]:
        gt_j = "+".join([sn for col, sn in zip(ALL_PAT_COLS, ALL_PAT_SHORT)
                         if col in test_gt[j][1].index and test_gt[j][1][col] == 1]) or "Normal"
        match = "✓ same" if labels[i] == labels[j] else "✗ diff"
        print(f"    → {img_names[j]:30s} [{labels[j]}] GT=[{gt_j:15s}] sim={s:.4f} {match}")
    j_far, s_far = sims[-1]
    gt_far = "+".join([sn for col, sn in zip(ALL_PAT_COLS, ALL_PAT_SHORT)
                       if col in test_gt[j_far][1].index and test_gt[j_far][1][col] == 1]) or "Normal"
    print(f"  En uzak:")
    print(f"    → {img_names[j_far]:30s} [{labels[j_far]}] GT=[{gt_far:15s}] sim={s_far:.4f}")

# ═══════════════════════════════════════════════════════════════════
# PART 3: LABEL OVERLAP vs SIMILARITY
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("📊 PART 3: LABEL OVERLAP vs COSINE SIMILARITY")
print(f"{'=' * 80}")

overlap_sims = []
no_overlap_sims = []
both_normal_sims = []

for i, j in combinations(range(len(embeddings)), 2):
    gt_i = {sn: int(test_gt[i][1][col]) if col in test_gt[i][1].index else 0
            for col, sn in zip(ALL_PAT_COLS, ALL_PAT_SHORT)}
    gt_j = {sn: int(test_gt[j][1][col]) if col in test_gt[j][1].index else 0
            for col, sn in zip(ALL_PAT_COLS, ALL_PAT_SHORT)}

    shared = sum(1 for sn in ALL_PAT_SHORT if gt_i[sn] == 1 and gt_j[sn] == 1)
    any_pos_i = sum(gt_i.values())
    any_pos_j = sum(gt_j.values())
    s = cosine_similarity(emb_array[i].reshape(1, -1), emb_array[j].reshape(1, -1))[0, 0]

    if any_pos_i == 0 and any_pos_j == 0:
        both_normal_sims.append(s)
    elif shared > 0:
        overlap_sims.append(s)
    else:
        no_overlap_sims.append(s)

print(f"\n  Ortak label olan çiftler:     mean={np.mean(overlap_sims):.4f} (n={len(overlap_sims)})" if overlap_sims else "  Ortak label olan çiftler: yok")
print(f"  Ortak label olmayan çiftler:  mean={np.mean(no_overlap_sims):.4f} (n={len(no_overlap_sims)})" if no_overlap_sims else "  Ortak label olmayan çiftler: yok")
print(f"  Her ikisi de normal:          mean={np.mean(both_normal_sims):.4f} (n={len(both_normal_sims)})" if both_normal_sims else "  Her ikisi de normal: yok")

if overlap_sims and no_overlap_sims:
    gap = np.mean(overlap_sims) - np.mean(no_overlap_sims)
    print(f"\n  Gap (overlap - no_overlap): {gap:.4f}")
    if gap > 0.02:
        print(f"  ✅ Aynı label'a sahip görüntüler birbirine daha yakın!")
    elif gap > 0.005:
        print(f"  🟡 Zayıf ama pozitif ayrışma")
    else:
        print(f"  ⚠️ Label bazlı ayrışma yok")

# ═══════════════════════════════════════════════════════════════════
# PART 4: PER-LABEL CENTROID ANALYSIS
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("📊 PART 4: PER-LABEL CENTROID — Her label'ın ortalamasına uzaklık")
print(f"{'=' * 80}")

nor_indices = [i for i, l in enumerate(labels) if l == "NOR"]
nor_centroid = emb_array[nor_indices].mean(axis=0)
nor_centroid = nor_centroid / (np.linalg.norm(nor_centroid) + 1e-8)

pat_indices = [i for i, l in enumerate(labels) if l == "PAT"]
pat_centroid = emb_array[pat_indices].mean(axis=0)
pat_centroid = pat_centroid / (np.linalg.norm(pat_centroid) + 1e-8)

print(f"\n  {'Image':30s} {'Group':5s} {'Primary':8s} {'→PAT':>8s} {'→NOR':>8s} {'Closer':>8s}")
print(f"  {'─' * 70}")

n_correct = 0
for i in range(len(embeddings)):
    sim_pat = cosine_similarity(emb_array[i].reshape(1, -1), pat_centroid.reshape(1, -1))[0, 0]
    sim_nor = cosine_similarity(emb_array[i].reshape(1, -1), nor_centroid.reshape(1, -1))[0, 0]
    closer = "PAT" if sim_pat > sim_nor else "NOR"
    correct = closer == labels[i]
    if correct: n_correct += 1
    mark = "✓" if correct else "✗"
    print(f"  {img_names[i]:30s} {labels[i]:5s} {primary_labels[i]:8s} "
          f"{sim_pat:8.4f} {sim_nor:8.4f} {closer:>5s} {mark}")

print(f"\n  Centroid accuracy: {n_correct}/{len(embeddings)} ({n_correct/len(embeddings)*100:.0f}%)")

# ═══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("📋 FINAL SUMMARY — Qwen2.5-VL-REF vs Qwen2.5-VL-3B vs MedGemma")
print(f"{'=' * 80}")
print(f"  Images tested: {len(embeddings)} ({n_pat} PAT + {n_nor} NOR)")
print(f"  Embedding dim: {emb_array.shape[1]}")
print(f"  PAT↔PAT: {mean_pp:.4f} | NOR↔NOR: {mean_nn:.4f} | PAT↔NOR: {mean_pn:.4f}")
print(f"  Separation gap: {mean_pp - mean_pn:.4f} / {mean_nn - mean_pn:.4f}")
print(f"  Centroid accuracy: {n_correct}/{len(embeddings)}")
if overlap_sims and no_overlap_sims:
    print(f"  Label overlap gap: {np.mean(overlap_sims) - np.mean(no_overlap_sims):.4f}")

gap_pp = mean_pp - mean_pn
print(f"\n  Qwen2.5-VL-8B:  dim={emb_array.shape[1]}, gap_PP={gap_pp:.4f}, centroid={n_correct}/{len(embeddings)}")
print(f"  Qwen2.5-3B:   dim=2048, gap_PP=0.0396, centroid=10/10")
print(f"  MedGemma 4B:  dim=2560, gap_PP=-0.0123, centroid=13/15")

if gap_pp > 0.0396:
    print(f"\n  ✅ Qwen2.5-VL-REF — proceed with this model for full extraction")
elif gap_pp > 0:
    print(f"\n  🟡 Positive gap but Qwen2.5-3B still better — consider sticking with 3B")
else:
    print(f"\n  ❌ Negative gap — stick with Qwen2.5-VL-3B")

viable = (mean_pp > mean_pn + 0.01) or (mean_nn > mean_pn + 0.01) or (n_correct >= len(embeddings) * 0.7)
if viable:
    print(f"  ✅ GT-guided embedding VIABLE — proceed to full extraction")
else:
    print(f"  ⚠️ Weak signal — may need larger test or different pooling strategy")

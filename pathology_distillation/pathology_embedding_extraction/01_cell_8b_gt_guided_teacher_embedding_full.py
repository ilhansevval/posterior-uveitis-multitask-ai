# ============================================================================
# CELL 8b: TEACHER EMBEDDING (NO GT GUIDANCE) — FULL CLASS-LEVEL ANALYSIS
# ============================================================================
# Self-contained: loads data, model, everything needed.
# The model analyzes each image on its own — ground-truth labels are NOT
# fed into the prompt. GT is used ONLY for evaluation / metadata.
# Just run this single cell.
# ============================================================================

import os
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
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# ─────────────────────────────────────────────
# LABEL MAPPINGS
# ─────────────────────────────────────────────
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
print(f"   Total: {len(df)} images, Pathological: {len(df_pat)}, Normal: {len(df) - len(df_pat)}")

# ─────────────────────────────────────────────
# LOAD QWEN MODEL
# ─────────────────────────────────────────────
print(f"\n🧠 Loading {MODEL_ID} (4-bit)...")
# HF_TOKEN koda gomulmez! Ortam degiskeni ile ver: export HF_TOKEN=hf_...
assert os.environ.get("HF_TOKEN"), "HF_TOKEN ortam degiskeni ayarli degil!"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
vl_model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
vl_processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
print(f"   ✅ Model loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

# ─────────────────────────────────────────────
# LABEL DEFINITIONS
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

# ─────────────────────────────────────────────
# CO-OCCURRENCE INFO FOR PROMPT
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# ANALYSIS PROMPT (NO GT GUIDANCE)
# ─────────────────────────────────────────────
# The model does NOT receive the ground-truth labels. It is asked to
# examine the image and decide, on its own, which findings are present.
def make_prompt():
    defs_text = "\n".join(
        [f"  - {sn}: {LABEL_DEFS.get(sn, sn)}" for sn in ALL_PAT_SHORT])

    prompt = f"""You are analyzing a retinal fluorescein angiography (FA) image.
Your task is to examine the image and identify any pathological findings
present, using ONLY the visible image evidence.

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
POSSIBLE PATHOLOGICAL FINDINGS (look for visual evidence of each):
{defs_text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{COOCCURRENCE_INFO}

INSTRUCTIONS:
1. Systematically examine the image for EACH possible finding listed above.

2. For each finding you judge to be PRESENT:
   - Describe the specific ABNORMAL visual evidence (not normal anatomy)
   - Mention WHERE in the image
   - Explain why this is PATHOLOGICAL and not normal anatomy

3. For each finding you judge to be ABSENT:
   - State what ABNORMAL feature would be visible if it were present
   - Explain that you see only NORMAL anatomy in that region, not pathology

4. For each finding, explicitly state: "This is [NORMAL ANATOMY / PATHOLOGICAL] because..."

5. Conclude with a short list of the findings you judge PRESENT
   (or state "Normal — no pathological findings" if none).

Be detailed and clinically focused. Use ONLY visible image evidence.
Distinguish normal anatomical appearances from pathological findings."""

    return prompt


SYSTEM_PROMPT = (
    "You are a medical vision-language assistant specialized in retinal fluorescein "
    "angiography interpretation. Examine each image carefully and identify any "
    "pathological findings, explaining the visual evidence for each present finding "
    "and the missing evidence for absent ones. Be thorough and specific."
)

# ─────────────────────────────────────────────
# EMBEDDING + EXPLANATION EXTRACTION
# ─────────────────────────────────────────────
def qwen_embedding(img_path, image_size=384, max_new_tokens=600):
    img = Image.open(img_path).convert("RGB").resize((image_size, image_size))
    prompt = make_prompt()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ]},
    ]

    text_input = vl_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    inputs = vl_processor(
        text=[text_input], images=[img], padding=True, return_tensors="pt",
    ).to(vl_model.device)

    torch.cuda.empty_cache()
    input_len = inputs['input_ids'].shape[1]

    # Generate WITH hidden states — get reasoning embedding from generated tokens
    with torch.no_grad():
        gen_output = vl_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    # ── EMBEDDING: from generated tokens (reasoning state) ──
    # Each step in hidden_states has layer outputs for that generated token
    # We take the LAST layer's last token from each generation step
    gen_embeddings = []
    for step_hidden in gen_output.hidden_states:
        # step_hidden is tuple of (n_layers,) each (batch, seq, hidden)
        last_layer = step_hidden[-1]  # last layer
        last_token = last_layer[0, -1, :]  # last token position
        gen_embeddings.append(last_token.cpu().float())

    gen_embeddings = torch.stack(gen_embeddings)  # (n_generated, hidden_dim)

    # Pool: use last 30 tokens (where the model has finished reasoning)
    n_use = min(30, len(gen_embeddings))
    emb = gen_embeddings[-n_use:].mean(dim=0).numpy()

    # L2 normalize
    emb = emb / (np.linalg.norm(emb) + 1e-8)

    # ── EXPLANATION: decode generated text ──
    generated_ids = gen_output.sequences[0][input_len:]
    explanation = vl_processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Quality check
    if len(explanation) < 50:
        print(f"    ⚠️ Short explanation ({len(explanation)} chars)")

    return emb, explanation


# ═══════════════════════════════════════════════════════════════════
# TEST: 5 pathological + 5 normal
# ═══════════════════════════════════════════════════════════════════
print("=" * 80)
print("🔍 TEACHER EMBEDDING (NO GT GUIDANCE) — FULL ANALYSIS")
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

    print(f"\n{'─' * 80}")
    print(f"  [{group}] {img_name} — Primary: {primary} — GT: [{gt_str}]")

    emb, expl = qwen_embedding(row["image_path"])

    embeddings.append(emb)
    labels.append(group)
    primary_labels.append(primary)
    explanations.append(expl)
    img_names.append(img_name)

    print(f"  Embedding: shape={emb.shape}, norm={np.linalg.norm(emb):.4f}")
    print(f"  Explanation ({len(expl)} chars):")
    # Print first 300 chars wrapped
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

# Normal centroid
nor_indices = [i for i, l in enumerate(labels) if l == "NOR"]
nor_centroid = emb_array[nor_indices].mean(axis=0)
nor_centroid = nor_centroid / (np.linalg.norm(nor_centroid) + 1e-8)

# Pat centroid
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
print("📋 FINAL SUMMARY")
print(f"{'=' * 80}")
print(f"  Images tested: {len(embeddings)} ({n_pat} PAT + {n_nor} NOR)")
print(f"  Embedding dim: {emb_array.shape[1]}")
print(f"  PAT↔PAT: {mean_pp:.4f} | NOR↔NOR: {mean_nn:.4f} | PAT↔NOR: {mean_pn:.4f}")
print(f"  Separation gap: {mean_pp - mean_pn:.4f} / {mean_nn - mean_pn:.4f}")
print(f"  Centroid accuracy: {n_correct}/{len(embeddings)}")
if overlap_sims and no_overlap_sims:
    print(f"  Label overlap gap: {np.mean(overlap_sims) - np.mean(no_overlap_sims):.4f}")

viable = (mean_pp > mean_pn + 0.01) or (mean_nn > mean_pn + 0.01) or (n_correct >= len(embeddings) * 0.7)
if viable:
    print(f"\n  ✅ Embedding VIABLE — proceed to full extraction")
else:
    print(f"\n  ⚠️ Weak signal — may need larger test or different pooling strategy")

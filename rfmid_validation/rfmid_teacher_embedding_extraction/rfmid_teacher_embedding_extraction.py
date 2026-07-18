# ============================================================================
# RFMiD TEACHER EMBEDDING EXTRACTION — Aşama 2
# ============================================================================
# Cerrahpaşa Cell 8b + Cell 9 yapısına SADIK, RFMiD'e uyarlanmış:
#   - GT-guided prompt (her label NORMAL vs PATHOLOGICAL ayrımıyla)
#   - last_30 token pooling, max_tokens=1000, normalize
#   - incremental save (her 20'de), auto-resume
#   - Self-contained: model + prompt + extraction hepsi burada
#
# FARK: FA → CFP (renkli fundus), uveitis label'ları → RFMiD 14 label
# Görüntüler: 14 label'dan ≥1 pozitif olan train+val+test görüntüleri
# ============================================================================

import os, time
import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
RFMID_ROOT = r'C:\Users\gtu\Downloads\A. RFMiD_All_Classes_Dataset'
IMG_DIR = os.path.join(RFMID_ROOT, '1. Original Images')
GT_DIR = os.path.join(RFMID_ROOT, '2. Groundtruths')
SETS = {
    'train': (os.path.join(IMG_DIR, 'a. Training Set'),
              os.path.join(GT_DIR, 'a. RFMiD_Training_Labels.csv')),
    'val':   (os.path.join(IMG_DIR, 'b. Validation Set'),
              os.path.join(GT_DIR, 'b. RFMiD_Validation_Labels.csv')),
    'test':  (os.path.join(IMG_DIR, 'c. Testing Set'),
              os.path.join(GT_DIR, 'c. RFMiD_Testing_Labels.csv')),
}

# embedding'ler Cerrahpaşa klasörüne kaydedilsin (KD eğitimi orada)
SAVE_ROOT = r'C:\Users\gtu\Documents\cerrahpasa\files\files'
SAVE_DIR = os.path.join(SAVE_ROOT, 'teacher_embeddings_rfmid')
os.makedirs(SAVE_DIR, exist_ok=True)
EMB_PATH = os.path.join(SAVE_DIR, 'teacher_embeddings.npy')
META_PATH = os.path.join(SAVE_DIR, 'teacher_metadata.csv')
EXPL_DIR = os.path.join(SAVE_DIR, 'explanations')
os.makedirs(EXPL_DIR, exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# 14 eğitilebilir RFMiD label (Aşama 1 ile aynı)
ALL_SHORT = ['DR', 'MH', 'ODC', 'TSLN', 'DN', 'ARMD', 'MYA',
             'BRVO', 'ODP', 'ODE', 'LS', 'RS', 'CSR', 'CRS']
ALL_COLS = ALL_SHORT  # RFMiD'de kolon adı = kısaltma

# ─────────────────────────────────────────────
# LABEL DEFINITIONS — RFMiD CFP (renkli fundus)
# Senin FA tanımlarındaki NORMAL vs PATHOLOGICAL mantığı korunuyor
# ─────────────────────────────────────────────
LABEL_DEFS = {
    "DR": (
        "Diabetic retinopathy — microaneurysms (tiny red dots), dot/blot hemorrhages, hard exudates "
        "(yellow waxy deposits), and cotton wool spots scattered across the retina. "
        "NORMAL ANATOMY NOTE: A few isolated vessel crossings are normal. PATHOLOGICAL only if there "
        "are multiple red dots/hemorrhages and/or yellow exudates distributed in the posterior pole."
    ),
    "MH": (
        "Media haze — overall blurred/cloudy image with reduced clarity due to opacity in the optical "
        "media (cornea, lens, vitreous). NORMAL ANATOMY NOTE: Some images are slightly soft-focus. "
        "PATHOLOGICAL only if the WHOLE image is uniformly hazy/foggy so that vessel and disc detail "
        "are globally washed out, not just one region."
    ),
    "ODC": (
        "Optic disc cupping — enlarged central cup of the optic disc with increased cup-to-disc ratio "
        "(thin neuroretinal rim), suggestive of glaucoma. NORMAL ANATOMY NOTE: Every disc has a small "
        "physiological cup. PATHOLOGICAL only if the cup is LARGE relative to the disc (>0.6 ratio), "
        "with a thinned or notched rim."
    ),
    "TSLN": (
        "Tessellation — a tigroid/checkerboard fundus where large choroidal vessels are visible through "
        "a thin retinal pigment epithelium. NORMAL ANATOMY NOTE: Mild tessellation near the periphery "
        "is common. PATHOLOGICAL/notable only if the tigroid pattern is prominent across the posterior pole."
    ),
    "DN": (
        "Drusen — discrete yellow-white round deposits beneath the retina, typically clustered at the "
        "macula. NORMAL ANATOMY NOTE: Normal macula is uniform. PATHOLOGICAL only if multiple pale "
        "yellow round spots are seen, usually around the fovea."
    ),
    "ARMD": (
        "Age-related macular degeneration — macular drusen, pigment changes, geographic atrophy, or "
        "neovascular scarring at the macula. NORMAL ANATOMY NOTE: A healthy macula is smooth and slightly "
        "darker centrally. PATHOLOGICAL only if there is clustered drusen, pigment mottling, or a macular "
        "scar/atrophic patch."
    ),
    "MYA": (
        "Myopia (myopic fundus) — peripapillary atrophy (pale crescent beside the disc), tessellated "
        "fundus, and a tilted disc from axial elongation. NORMAL ANATOMY NOTE: A small scleral crescent "
        "can be normal. PATHOLOGICAL/notable only if there is a clear peripapillary atrophic crescent "
        "with diffuse tessellation."
    ),
    "BRVO": (
        "Branch retinal vein occlusion — sectoral (wedge-shaped) flame hemorrhages and dilated tortuous "
        "veins confined to ONE quadrant along a vein. NORMAL ANATOMY NOTE: Vessels are normally smooth. "
        "PATHOLOGICAL only if hemorrhages and venous dilation are localized to one sector draining a vein."
    ),
    "ODP": (
        "Optic disc pallor — an abnormally pale/white optic disc indicating optic atrophy. "
        "NORMAL ANATOMY NOTE: The disc is normally pinkish-orange with a paler central cup. "
        "PATHOLOGICAL only if the whole neuroretinal rim is distinctly pale/white compared to a healthy "
        "pink disc."
    ),
    "ODE": (
        "Optic disc edema — a swollen optic disc with blurred/elevated margins and obscured vessels at "
        "the disc edge. NORMAL ANATOMY NOTE: A normal disc has sharp, flat margins. PATHOLOGICAL only if "
        "the disc margins are blurred, the disc looks elevated/hyperemic, and crossing vessels are obscured."
    ),
    "LS": (
        "Laser scars — multiple discrete round/oval pigmented scars (panretinal photocoagulation) in a "
        "regular pattern in the mid-periphery. NORMAL ANATOMY NOTE: The peripheral retina is normally "
        "uniform. PATHOLOGICAL only if there are many regularly-spaced round pigmented/atrophic spots."
    ),
    "RS": (
        "Retinitis — a focal area of retinal inflammation appearing as a white/creamy fluffy patch with "
        "indistinct borders, often near vessels. NORMAL ANATOMY NOTE: The retina is normally a uniform "
        "orange-red. PATHOLOGICAL only if there is a discrete pale/white fluffy lesion with blurred edges."
    ),
    "CSR": (
        "Central serous retinopathy — a round area of serous retinal detachment at the macula appearing "
        "as a subtle dome/blister of subretinal fluid. NORMAL ANATOMY NOTE: A normal macula is flat. "
        "PATHOLOGICAL only if there is a circular slightly elevated zone of fluid at the posterior pole."
    ),
    "CRS": (
        "Chorioretinitis — inflammatory chorioretinal lesion with a pale/atrophic center and pigmented "
        "borders, often punched-out. NORMAL ANATOMY NOTE: The fundus is normally uniform. PATHOLOGICAL "
        "only if there is a focal scar with pale center and pigmented margin, or active creamy infiltrate "
        "involving choroid and retina."
    ),
}

# ─────────────────────────────────────────────
# CO-OCCURRENCE INFO (RFMiD klinik örüntüleri)
# ─────────────────────────────────────────────
COOCCURRENCE_INFO = """
CLINICAL CO-OCCURRENCE PATTERNS (use to validate your observations):
- Myopia (MYA) frequently co-occurs with Tessellation (TSLN) and peripapillary atrophy.
- Age-related macular degeneration (ARMD) frequently co-occurs with Drusen (DN) at the macula.
- Diabetic retinopathy (DR) may co-occur with vein occlusion (BRVO) and laser scars (LS) from prior treatment.
- Optic disc edema (ODE), optic disc pallor (ODP), and optic disc cupping (ODC) are MUTUALLY DISTINCT
  disc appearances — a swollen disc (ODE) is not pale (ODP) and not cupped (ODC).
- Retinitis (RS) and Chorioretinitis (CRS) are both inflammatory and may appear similar; CRS involves
  the choroid with pigmented punched-out borders, RS is more superficial retinal whitening.
"""

# ─────────────────────────────────────────────
# GT-GUIDED PROMPT (Cerrahpaşa yapısına sadık, CFP'ye uyarlı)
# ─────────────────────────────────────────────
def make_gt_prompt(gt_dict, max_neg=7):
    pos = [sn for sn, v in gt_dict.items() if v == 1]
    neg = [sn for sn, v in gt_dict.items() if v == 0]
    neg = neg[:max_neg]

    pos_text = "\n".join([f"  - {sn}: {LABEL_DEFS.get(sn, sn)}" for sn in pos]) if pos else "  - None (normal image)"
    neg_text = "\n".join([f"  - {sn}: {LABEL_DEFS.get(sn, sn)}" for sn in neg]) if neg else "  - None"

    prompt = f"""You are analyzing a COLOR FUNDUS PHOTOGRAPH (CFP) of the retina.
The diagnosis is ALREADY CONFIRMED by an expert ophthalmologist.
Your task is NOT to diagnose — it is to explain the visible evidence.

CRITICAL RULE — NORMAL vs PATHOLOGICAL:
Many structures in the eye are NORMALLY present in a color fundus photo:
- The OPTIC DISC is a normal round pinkish-orange structure with a small central cup — this alone is NOT pathological
- The MACULA (center) is normally slightly darker and uniform — this alone is NOT a lesion
- Retinal VESSELS radiate from the disc and are normally smooth — this alone is NOT occlusion
- Mild peripheral tessellation/color variation can be normal — this alone is NOT disease
ONLY mark something as pathological if it is ABNORMAL compared to a healthy eye.
Each label definition includes what is normal vs pathological — follow those strictly.

Image characteristics (color fundus):
- Orange-red background = normal retina/choroid
- Bright yellow-white spots = exudates, drusen, or inflammatory lesions
- Dark red blots/flames = hemorrhages
- Pale/white round disc = optic nerve head (NORMALLY present)
- Darker central zone = macula/fovea (NORMALLY present)

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
   - Mention WHERE in the image (macula, disc, which quadrant)
   - Explain why this is PATHOLOGICAL and not normal anatomy

2. For EACH confirmed ABSENT finding:
   - State what ABNORMAL feature would be visible if it were present
   - Explain that you see only NORMAL anatomy in that region, not pathology

3. For each finding, explicitly state: "This is [NORMAL ANATOMY / PATHOLOGICAL] because..."

Be detailed and clinically focused. Use ONLY visible image evidence.
Distinguish normal anatomical appearances from pathological findings."""
    return prompt


GT_SYSTEM = (
    "You are a medical vision-language assistant specialized in COLOR FUNDUS PHOTOGRAPH "
    "interpretation. The diagnoses are already confirmed by an expert. "
    "Your job is to explain the visual evidence that supports present findings and "
    "the missing visual evidence for absent findings. Be thorough and specific."
)

# ─────────────────────────────────────────────
# LOAD RFMiD DATA (3 set birleşik, ≥1 pozitif görüntüler)
# ─────────────────────────────────────────────
print("📂 Loading RFMiD...")
parts = []
for split, (img_dir, csv_path) in SETS.items():
    d = pd.read_csv(csv_path)
    d['image_path'] = d['ID'].apply(lambda i: os.path.join(img_dir, f"{int(i)}.png"))
    d['split'] = split
    d = d[d['image_path'].apply(os.path.exists)]
    parts.append(d)
df = pd.concat(parts, ignore_index=True)
for c in ALL_COLS:
    df[c] = df[c].astype(int)

# 14 label'dan en az biri pozitif olanlar
df['n_pathology'] = df[ALL_COLS].sum(axis=1)
df_pat = df[df['n_pathology'] > 0].reset_index(drop=True)
print(f"   Total: {len(df)} | With ≥1 of 14 labels: {len(df_pat)}")
print(f"   Label dağılımı (df_pat):")
for sn in ALL_SHORT:
    print(f"     {sn:6s}: {int(df_pat[sn].sum()):4d}")

# ─────────────────────────────────────────────
# LOAD QWEN MODEL (Cerrahpaşa ile aynı — 4-bit)
# ─────────────────────────────────────────────
print(f"\n🧠 Loading {MODEL_ID} (4-bit)...")
# HF_TOKEN koda gomulmez! Ortam degiskeni ile ver: export HF_TOKEN=hf_...
assert os.environ.get("HF_TOKEN"), "HF_TOKEN ortam degiskeni ayarli degil!"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
vl_model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
vl_processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
print(f"   ✅ Model loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

# ─────────────────────────────────────────────
# EXTRACTION (last_30 strategy, max_tokens=1000 — Cerrahpaşa ile AYNI)
# ─────────────────────────────────────────────
def extract_teacher_embedding(img_path, gt_dict):
    img = Image.open(img_path).convert("RGB").resize((384, 384))
    prompt = make_gt_prompt(gt_dict)
    messages = [
        {"role": "system", "content": GT_SYSTEM},
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

    with torch.no_grad():
        gen_output = vl_model.generate(
            **inputs, max_new_tokens=1000, do_sample=False,
            output_hidden_states=True, return_dict_in_generate=True)

    gen_embs = []
    for step_hidden in gen_output.hidden_states:
        gen_embs.append(step_hidden[-1][0, -1, :].cpu().float())
    gen_embs = torch.stack(gen_embs)
    n_use = min(30, len(gen_embs))
    emb = gen_embs[-n_use:].mean(dim=0).numpy()
    emb = emb / (np.linalg.norm(emb) + 1e-8)

    generated_ids = gen_output.sequences[0][input_len:]
    explanation = vl_processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return emb, explanation


# ─────────────────────────────────────────────
# AUTO-RESUME
# ─────────────────────────────────────────────
df_all = df_pat.reset_index(drop=True)
print(f"\n📂 Total to process: {len(df_all)}")
if os.path.exists(META_PATH):
    done_df = pd.read_csv(META_PATH)
    done_paths = set(done_df['image_path'].values)
    done_embs = list(np.load(EMB_PATH)) if os.path.exists(EMB_PATH) else []
    print(f"   Resuming: {len(done_paths)} already done")
else:
    done_df = pd.DataFrame(); done_paths = set(); done_embs = []
    print("   Starting fresh")

remaining = df_all[~df_all['image_path'].isin(done_paths)].reset_index(drop=True)
print(f"   Remaining: {len(remaining)} | Est: ~{len(remaining)*15/60:.0f} min\n")

# ─────────────────────────────────────────────
# BATCH LOOP
# ─────────────────────────────────────────────
new_metas = []; new_embs = []; errors = 0
t0 = time.time()

for i, row in remaining.iterrows():
    img_path = row['image_path']
    img_name = os.path.basename(img_path)
    gt_dict = {sn: int(row[col]) for col, sn in zip(ALL_COLS, ALL_SHORT)}
    gt_str = "+".join([s for s in ALL_SHORT if gt_dict[s] == 1])

    try:
        emb, expl = extract_teacher_embedding(img_path, gt_dict)
        new_embs.append(emb)
        meta = {
            'image_path': img_path, 'image_name': img_name,
            'image_id': int(row['ID']), 'split': row['split'],
            'gt_labels': gt_str, 'expl_len': len(expl),
        }
        for sn in ALL_SHORT:
            meta[f'gt_{sn}'] = gt_dict[sn]
        new_metas.append(meta)

        expl_file = os.path.join(EXPL_DIR, img_name.replace('.png', '.txt').replace('.jpg', '.txt'))
        with open(expl_file, 'w', encoding='utf-8') as f:
            f.write(expl)

        total_done = len(done_paths) + len(new_embs)
        elapsed = (time.time() - t0) / 60
        eta = elapsed / max(len(new_embs), 1) * (len(remaining) - len(new_embs))
        print(f"  [{total_done:4d}/{len(df_all)}] {img_name:14s} [{gt_str:18s}] "
              f"expl={len(expl):5d} ({elapsed:.1f}m, ETA {eta:.0f}m)")
    except Exception as e:
        errors += 1
        print(f"  [{len(done_paths)+len(new_embs):4d}/{len(df_all)}] {img_name:14s} ⚠️ {str(e)[:50]}")

    if len(new_embs) % 20 == 0 and len(new_embs) > 0:
        np.save(EMB_PATH, np.array(done_embs + new_embs))
        pd.concat([done_df, pd.DataFrame(new_metas)], ignore_index=True).to_csv(
            META_PATH, index=False, encoding='utf-8-sig')
        print(f"    💾 Checkpoint: {len(done_embs)+len(new_embs)} embeddings")

# ─────────────────────────────────────────────
# FINAL SAVE
# ─────────────────────────────────────────────
if new_embs:
    np.save(EMB_PATH, np.array(done_embs + new_embs))
    pd.concat([done_df, pd.DataFrame(new_metas)], ignore_index=True).to_csv(
        META_PATH, index=False, encoding='utf-8-sig')

total = len(done_paths) + len(new_embs)
elapsed = (time.time() - t0) / 60
print(f"\n{'='*60}")
print(f"✅ RFMiD EMBEDDING EXTRACTION COMPLETE")
print(f"   Total: {total}/{len(df_all)} | Errors: {errors} | Time: {elapsed:.1f} min")
print(f"   Embeddings: {EMB_PATH}")
print(f"   Metadata:   {META_PATH}")
if os.path.exists(EMB_PATH) and os.path.exists(META_PATH):
    embs = np.load(EMB_PATH); meta = pd.read_csv(META_PATH)
    print(f"\n   Verify: {embs.shape[0]} embeddings × {embs.shape[1]} dim")
    for sn in ALL_SHORT:
        print(f"     {sn:6s}: {int(meta[f'gt_{sn}'].sum()):4d}")
print(f"{'='*60}")

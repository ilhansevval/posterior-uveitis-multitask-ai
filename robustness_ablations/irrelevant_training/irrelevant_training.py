# ============================================================================
# STANDALONE: WRONG PROMPT EMBEDDING EXTRACTION — 3 Variants
# ============================================================================
# 3 farklı embedding seti üretir:
#   S1: name_only    → Sadece label isimleri, açıklama yok
#   S2: wrong_anat   → Kasıtlı yanlış anatomik bilgi
#   S3: irrelevant   → Patoloji ile alakasız prompt
#
# Standalone: VLM model, df_pat hepsi bu script içinde yüklenir
# Reference images KULLANILMAZ — sadece patient image + wrong prompt
# Auto-resume: her variant için ayrı progress
# ============================================================================

import os, time, warnings
import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_ROOT = r'C:\Users\gtu\Documents\cerrahpasa\files\files'
META_PATH_IN = os.path.join(DATA_ROOT, 'teacher_embeddings_ref', 'teacher_metadata.csv')

# Label columns
ALL_PAT_COLS = ['gt_DKS', 'gt_ODB', 'gt_VI', 'gt_MÖ',
                'gt_DDB', 'gt_RI', 'gt_HEM', 'gt_PVK']
ALL_PAT_SHORT = ['DKS', 'ODB', 'VI', 'MÖ', 'DDB', 'RI', 'HEM', 'PVK']

# Model config
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device: {DEVICE}")

# ─────────────────────────────────────────────
# LOAD VL MODEL
# ─────────────────────────────────────────────
print(f"\n📦 Loading Qwen2.5-VL-3B...")
vl_processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
vl_model.eval()
print(f"   ✅ Model loaded")

# ─────────────────────────────────────────────
# LOAD DATAFRAME (561 pathological images)
# ─────────────────────────────────────────────
print(f"\n📂 Loading metadata from existing extraction...")
df_pat = pd.read_csv(META_PATH_IN)
print(f"   Total pathological images: {len(df_pat)}")
print(f"   Columns: {list(df_pat.columns)[:5]}...")

# ─────────────────────────────────────────────
# 3 YANLIŞ PROMPT VARYANTI (English)
# ─────────────────────────────────────────────

WRONG_SYSTEM = (
    "You are a medical image analysis assistant. "
    "Analyze the provided fundus image based on the user's question."
)


def prompt_name_only(gt_dict):
    """Seviye 1: Sadece label isimleri, hiçbir açıklama veya tanım yok."""
    return (
        "Analyze this fundus fluorescein angiography image. "
        "Check whether any of the following are present: "
        "DKS, ODB, VI, MO, DDB, RI, HEM, PVK. "
        "Provide a brief response."
    )


def prompt_wrong_anatomy(gt_dict):
    """Seviye 2: Kasıtlı yanlış anatomik bilgi."""
    return (
        "Analyze this fundus fluorescein angiography image for retinal pathologies. "
        "Use the following anatomical references during your analysis:\n"
        "- The optic disc normally appears DARK in FA images; brightness indicates abnormality.\n"
        "- The fovea normally appears BRIGHT and reflective; darkening suggests pathology.\n"
        "- Retinal hemorrhages appear as WHITE bright spots in FA imaging.\n"
        "- Healthy retinal vessels are wide, blurry, and indistinct in normal anatomy.\n"
        "- Macular edema causes the central retina to become smaller and darker than usual.\n"
        "- Vitreous inflammation is identified by sharp, well-defined edges in the periphery.\n"
        "Based on these criteria, describe what you observe in the image."
    )


def prompt_irrelevant(gt_dict):
    """Seviye 3: Tamamen alakasız prompt — patoloji ile hiçbir bağlantısı yok."""
    return (
        "Examine this image and describe the geometric shapes you can see. "
        "Count the number of circular regions and angular regions. "
        "Identify the dominant color in the image and estimate the overall brightness "
        "on a scale from 1 to 10. "
        "Describe the visual texture in general terms (smooth, rough, patterned)."
    )


PROMPT_VARIANTS = {
    'name_only':    prompt_name_only,
    'wrong_anat':   prompt_wrong_anatomy,
    'irrelevant':   prompt_irrelevant,
}

# ─────────────────────────────────────────────
# CORE EXTRACTION FUNCTION
# ─────────────────────────────────────────────

def extract_wrong_embedding(img_path, gt_dict, prompt_fn):
    """
    Yanlış prompt ile embedding çıkar.
    Reference images YOK — sadece patient image + wrong prompt.
    """
    img = Image.open(img_path).convert("RGB").resize((384, 384))
    prompt = prompt_fn(gt_dict)

    # Sadece patient image + prompt (ref yok!)
    content = [
        {"type": "text", "text": "[PATIENT IMAGE to analyze]:"},
        {"type": "image", "image": img},
        {"type": "text", "text": prompt},
    ]

    messages = [
        {"role": "system", "content": WRONG_SYSTEM},
        {"role": "user", "content": content},
    ]

    text_input = vl_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    all_images = [item["image"] for item in content if item.get("type") == "image"]

    inputs = vl_processor(
        text=[text_input], images=[all_images], padding=True, return_tensors="pt",
    ).to(vl_model.device)

    torch.cuda.empty_cache()
    input_len = inputs['input_ids'].shape[1]

    with torch.no_grad():
        gen_output = vl_model.generate(
            **inputs, max_new_tokens=1500, do_sample=False,
            output_hidden_states=True, return_dict_in_generate=True,
        )

    # Last 30 generated tokens, last layer
    gen_embs = []
    for step_hidden in gen_output.hidden_states:
        gen_embs.append(step_hidden[-1][0, -1, :].cpu().float())
    gen_embs = torch.stack(gen_embs)

    n_use = min(30, len(gen_embs))
    emb = gen_embs[-n_use:].mean(dim=0).numpy()
    emb = emb / (np.linalg.norm(emb) + 1e-8)

    # Explanation (generated text)
    generated_ids = gen_output.sequences[0][input_len:]
    explanation = vl_processor.tokenizer.decode(
        generated_ids, skip_special_tokens=True
    ).strip()

    return emb, explanation


# ─────────────────────────────────────────────
# DATAFRAME PREP
# ─────────────────────────────────────────────
df_all = df_pat.reset_index(drop=True)
print(f"\n📂 Total images to process: {len(df_all)} (pathological only)")
print(f"📂 Variants: {list(PROMPT_VARIANTS.keys())}")
print(f"📂 Estimated total time: ~{len(df_all) * len(PROMPT_VARIANTS) * 20 / 60:.0f} min")


# ─────────────────────────────────────────────
# PER-VARIANT EXTRACTION LOOP
# ─────────────────────────────────────────────

for variant_name, prompt_fn in PROMPT_VARIANTS.items():
    print(f"\n\n{'='*70}")
    print(f"🧪 VARIANT: {variant_name}")
    print(f"{'='*70}")

    # Per-variant save dir
    SAVE_DIR = os.path.join(DATA_ROOT, f'teacher_embeddings_wrong_{variant_name}')
    os.makedirs(SAVE_DIR, exist_ok=True)

    EMB_PATH = os.path.join(SAVE_DIR, 'teacher_embeddings.npy')
    META_PATH = os.path.join(SAVE_DIR, 'teacher_metadata.csv')
    EXPL_DIR = os.path.join(SAVE_DIR, 'explanations')
    os.makedirs(EXPL_DIR, exist_ok=True)

    print(f"📂 Save dir: {SAVE_DIR}")

    # Sample prompt for verification
    sample_gt = {sn: 0 for sn in ALL_PAT_SHORT}
    sample_prompt = prompt_fn(sample_gt)
    print(f"📝 Sample prompt (first 200 chars):")
    print(f"   {sample_prompt[:200]}...")

    # ── Check existing progress ──
    if os.path.exists(META_PATH):
        done_df = pd.read_csv(META_PATH)
        done_paths = set(done_df['image_path'].values)
        done_embs = list(np.load(EMB_PATH)) if os.path.exists(EMB_PATH) else []
        print(f"   Resuming: {len(done_paths)} already done")
    else:
        done_df = pd.DataFrame()
        done_paths = set()
        done_embs = []
        print(f"   Starting fresh")

    remaining = df_all[~df_all['image_path'].isin(done_paths)].reset_index(drop=True)
    print(f"   Remaining: {len(remaining)} images")
    print(f"   Estimated time: ~{len(remaining) * 20 / 60:.0f} min\n")

    # ── Batch loop ──
    new_metas = []
    new_embs = []
    errors = 0
    t0 = time.time()

    for i, row in remaining.iterrows():
        img_path = row['image_path']
        img_name = os.path.basename(img_path)
        patient = row['patient_id'] if 'patient_id' in row.index else 'unknown'

        gt_dict = {sn: int(row[col]) if col in row.index else 0
                   for col, sn in zip(ALL_PAT_COLS, ALL_PAT_SHORT)}
        gt_str = "+".join([s for s in ALL_PAT_SHORT if gt_dict[s] == 1])

        try:
            emb, expl = extract_wrong_embedding(img_path, gt_dict, prompt_fn)

            new_embs.append(emb)
            meta = {
                'image_path': img_path,
                'image_name': img_name,
                'patient_id': patient,
                'gt_labels': gt_str,
                'variant': variant_name,
                'expl_len': len(expl),
            }
            for sn in ALL_PAT_SHORT:
                meta[f'gt_{sn}'] = gt_dict[sn]
            new_metas.append(meta)

            # Save explanation text
            expl_file = os.path.join(
                EXPL_DIR,
                img_name.replace('.jpg', '.txt').replace('.png', '.txt')
            )
            with open(expl_file, 'w', encoding='utf-8') as f:
                f.write(expl)

            total_done = len(done_paths) + len(new_embs)
            elapsed = (time.time() - t0) / 60
            eta = elapsed / max(len(new_embs), 1) * (len(remaining) - len(new_embs))
            print(f"  [{variant_name:10s}] [{total_done:4d}/{len(df_all)}] "
                  f"{img_name:30s} [{gt_str:15s}] "
                  f"expl={len(expl):5d} ({elapsed:.1f}m, ETA {eta:.1f}m)")

        except Exception as e:
            errors += 1
            print(f"  [{variant_name:10s}] [{len(done_paths)+len(new_embs):4d}/{len(df_all)}] "
                  f"{img_name:30s} ⚠️ {str(e)[:80]}")

        # Incremental save every 20 images
        if len(new_embs) % 20 == 0 and len(new_embs) > 0:
            all_embs_save = done_embs + new_embs
            np.save(EMB_PATH, np.array(all_embs_save))

            new_meta_df = pd.DataFrame(new_metas)
            all_meta = pd.concat([done_df, new_meta_df], ignore_index=True)
            all_meta.to_csv(META_PATH, index=False, encoding='utf-8-sig')
            print(f"    💾 Saved checkpoint: {len(all_embs_save)} embeddings")

    # ── Final save for this variant ──
    if new_embs:
        all_embs_final = done_embs + new_embs
        np.save(EMB_PATH, np.array(all_embs_final))

        new_meta_df = pd.DataFrame(new_metas)
        all_meta = pd.concat([done_df, new_meta_df], ignore_index=True)
        all_meta.to_csv(META_PATH, index=False, encoding='utf-8-sig')

    total = len(done_paths) + len(new_embs)
    elapsed = (time.time() - t0) / 60

    print(f"\n  {'='*60}")
    print(f"  ✅ VARIANT '{variant_name}' COMPLETE")
    print(f"     Total: {total}/{len(df_all)} images")
    print(f"     Errors: {errors}")
    print(f"     Time: {elapsed:.1f} min")
    print(f"     Embeddings: {EMB_PATH}")

    # Verify
    if os.path.exists(EMB_PATH):
        embs = np.load(EMB_PATH)
        print(f"     Shape: {embs.shape[0]} × {embs.shape[1]}")

# ─────────────────────────────────────────────
# OVERALL SUMMARY
# ─────────────────────────────────────────────
print(f"\n\n{'='*70}")
print(f"🎯 ALL WRONG-PROMPT EMBEDDINGS COMPLETE")
print(f"{'='*70}")
for variant_name in PROMPT_VARIANTS.keys():
    SAVE_DIR = os.path.join(DATA_ROOT, f'teacher_embeddings_wrong_{variant_name}')
    EMB_PATH = os.path.join(SAVE_DIR, 'teacher_embeddings.npy')
    META_PATH = os.path.join(SAVE_DIR, 'teacher_metadata.csv')

    if os.path.exists(EMB_PATH):
        embs = np.load(EMB_PATH)
        print(f"  {variant_name:12s}: {embs.shape[0]:4d} × {embs.shape[1]:5d}  →  {EMB_PATH}")
    else:
        print(f"  {variant_name:12s}: NOT FOUND")
print(f"{'='*70}")

# ============================================================================
# CELL 9-REF: BATCH EMBEDDING EXTRACTION — Referanslı, 561 PAT → .npy
# ============================================================================
# Uses: vl_model, vl_processor, df_pat, make_gt_prompt, GT_SYSTEM,
#       ref_images, ALL_PAT_COLS, ALL_PAT_SHORT from Cell 8b (referanslı)
# Saves to: teacher_embeddings_ref/ (önceki teacher_embeddings/ üstüne yazmaz)
# Auto-resume: skips already processed images
# ============================================================================

import os, time
import torch
import numpy as np
import pandas as pd
from PIL import Image

SAVE_DIR = os.path.join(DATA_ROOT, 'teacher_embeddings_ref')
os.makedirs(SAVE_DIR, exist_ok=True)

EMB_PATH = os.path.join(SAVE_DIR, 'teacher_embeddings.npy')
META_PATH = os.path.join(SAVE_DIR, 'teacher_metadata.csv')
EXPL_DIR = os.path.join(SAVE_DIR, 'explanations')
os.makedirs(EXPL_DIR, exist_ok=True)

# ── Only pathological images ──
df_all = df_pat.reset_index(drop=True)
print(f"📂 Total images to process: {len(df_all)} (pathological only)")
print(f"📂 Save dir: {SAVE_DIR}")
print(f"📸 Reference images loaded: {list(ref_images.keys())}")

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
    print("   Starting fresh")

remaining = df_all[~df_all['image_path'].isin(done_paths)].reset_index(drop=True)
print(f"   Remaining: {len(remaining)} images")
print(f"   Estimated time: ~{len(remaining) * 20 / 60:.0f} min\n")

# ── Extraction function with reference images ──
def extract_teacher_embedding(img_path, gt_dict):
    img = Image.open(img_path).convert("RGB").resize((384, 384))
    prompt = make_gt_prompt(gt_dict)
    pos_labels = [sn for sn, v in gt_dict.items() if v == 1]

    # Build content: reference images + patient image + prompt
    content = []

    # Add reference images for present labels that have refs
    for sn in pos_labels:
        if sn in ref_images:
            content.append({
                "type": "text",
                "text": f"[REFERENCE — {sn} examples with annotations (arrows show pathology)]:"
            })
            for ref_img in ref_images[sn]:
                content.append({"type": "image", "image": ref_img.resize((384, 384))})

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

    # Explanation
    generated_ids = gen_output.sequences[0][input_len:]
    explanation = vl_processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return emb, explanation


# ── Batch loop ──
new_metas = []
new_embs = []
errors = 0
t0 = time.time()

for i, row in remaining.iterrows():
    img_path = row['image_path']
    img_name = os.path.basename(img_path)
    patient = row['patient_id']

    gt_dict = {sn: int(row[col]) if col in row.index else 0
               for col, sn in zip(ALL_PAT_COLS, ALL_PAT_SHORT)}
    gt_str = "+".join([s for s in ALL_PAT_SHORT if gt_dict[s] == 1])

    # Count how many ref labels this image has
    n_refs = sum(1 for sn in ALL_PAT_SHORT if gt_dict[sn] == 1 and sn in ref_images)

    try:
        emb, expl = extract_teacher_embedding(img_path, gt_dict)

        new_embs.append(emb)
        meta = {
            'image_path': img_path,
            'image_name': img_name,
            'patient_id': patient,
            'gt_labels': gt_str,
            'expl_len': len(expl),
            'n_ref_labels': n_refs,
        }
        for sn in ALL_PAT_SHORT:
            meta[f'gt_{sn}'] = gt_dict[sn]
        new_metas.append(meta)

        # Save explanation text
        expl_file = os.path.join(EXPL_DIR, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        with open(expl_file, 'w', encoding='utf-8') as f:
            f.write(expl)

        total_done = len(done_paths) + len(new_embs)
        elapsed = (time.time() - t0) / 60
        eta = elapsed / max(len(new_embs), 1) * (len(remaining) - len(new_embs))
        print(f"  [{total_done:4d}/{len(df_all)}] {img_name:30s} [{gt_str:15s}] "
              f"refs={n_refs} expl={len(expl):5d} ({elapsed:.1f}m, ETA {eta:.1f}m)")

    except Exception as e:
        errors += 1
        print(f"  [{len(done_paths)+len(new_embs):4d}/{len(df_all)}] "
              f"{img_name:30s} ⚠️ {str(e)[:80]}")

    # Incremental save every 20 images
    if len(new_embs) % 20 == 0 and len(new_embs) > 0:
        all_embs_save = done_embs + new_embs
        np.save(EMB_PATH, np.array(all_embs_save))

        new_meta_df = pd.DataFrame(new_metas)
        all_meta = pd.concat([done_df, new_meta_df], ignore_index=True)
        all_meta.to_csv(META_PATH, index=False, encoding='utf-8-sig')
        print(f"    💾 Saved checkpoint: {len(all_embs_save)} embeddings")

# ── Final save ──
if new_embs:
    all_embs_final = done_embs + new_embs
    np.save(EMB_PATH, np.array(all_embs_final))

    new_meta_df = pd.DataFrame(new_metas)
    all_meta = pd.concat([done_df, new_meta_df], ignore_index=True)
    all_meta.to_csv(META_PATH, index=False, encoding='utf-8-sig')

total = len(done_paths) + len(new_embs)
elapsed = (time.time() - t0) / 60

print(f"\n{'=' * 60}")
print(f"✅ EMBEDDING EXTRACTION COMPLETE (REFERANSLI)")
print(f"   Total: {total}/{len(df_all)} images")
print(f"   Errors: {errors}")
print(f"   Time: {elapsed:.1f} min")
print(f"   Embeddings: {EMB_PATH}")
print(f"   Metadata: {META_PATH}")
print(f"   Explanations: {EXPL_DIR}")

if os.path.exists(EMB_PATH) and os.path.exists(META_PATH):
    embs = np.load(EMB_PATH)
    meta = pd.read_csv(META_PATH)
    print(f"\n   Verify: {embs.shape[0]} embeddings × {embs.shape[1]} dim")
    print(f"   Label distribution:")
    for sn in ALL_PAT_SHORT:
        n = int(meta[f'gt_{sn}'].sum())
        if n > 0:
            print(f"     {sn:6s}: {n:4d}")
    # Ref usage stats
    if 'n_ref_labels' in meta.columns:
        print(f"\n   Ref usage:")
        print(f"     With refs:    {int((meta['n_ref_labels'] > 0).sum())}")
        print(f"     Without refs: {int((meta['n_ref_labels'] == 0).sum())}")
print(f"{'=' * 60}")

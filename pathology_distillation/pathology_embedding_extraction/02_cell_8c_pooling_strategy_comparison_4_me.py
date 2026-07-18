# ============================================================================
# CELL 8c: POOLING STRATEGY COMPARISON — 4 methods, same 10 images
# ============================================================================
# Reuses vl_model, vl_processor, df, df_pat from Cell 8b
# Tests: last_30, last_10, last_1, layer_minus_2
# ============================================================================

import torch
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

def extract_all_strategies(img_path, image_size=384, max_new_tokens=1000):
    """Generate once, extract 4 different embedding strategies (no GT guidance)"""
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

    with torch.no_grad():
        gen_output = vl_model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            output_hidden_states=True, return_dict_in_generate=True,
        )

    # Collect per-step embeddings from LAST layer and SECOND-TO-LAST layer
    last_layer_embs = []
    second_last_embs = []
    for step_hidden in gen_output.hidden_states:
        last_layer_embs.append(step_hidden[-1][0, -1, :].cpu().float())
        second_last_embs.append(step_hidden[-2][0, -1, :].cpu().float())

    last_layer_embs = torch.stack(last_layer_embs)    # (n_gen, hidden_dim)
    second_last_embs = torch.stack(second_last_embs)  # (n_gen, hidden_dim)

    n_gen = len(last_layer_embs)

    # Strategy 1: Last 30 tokens (current best)
    n30 = min(30, n_gen)
    emb_last30 = last_layer_embs[-n30:].mean(dim=0).numpy()

    # Strategy 2: Last 10 tokens (tighter reasoning window)
    n10 = min(10, n_gen)
    emb_last10 = last_layer_embs[-n10:].mean(dim=0).numpy()

    # Strategy 3: Last 1 token (final decision point)
    emb_last1 = last_layer_embs[-1].numpy()

    # Strategy 4: Second-to-last layer, last 30 tokens
    emb_layer2_30 = second_last_embs[-n30:].mean(dim=0).numpy()

    # Normalize all
    results = {}
    for name, emb in [('last_30', emb_last30), ('last_10', emb_last10),
                       ('last_1', emb_last1), ('layer-2_30', emb_layer2_30)]:
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        results[name] = emb

    # Explanation
    generated_ids = gen_output.sequences[0][input_len:]
    explanation = vl_processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return results, explanation, n_gen


# ── Same 10 test images ──
test_imgs = []
used = set()
for col, sn in zip(ALL_PAT_COLS[:5], ALL_PAT_SHORT[:5]):
    pos = df_pat[df_pat[col] == 1]
    for _, row in pos.iterrows():
        if row['image_path'] not in used:
            test_imgs.append(("PAT", row, sn))
            used.add(row['image_path'])
            break
for _, row in df[df["n_pathology"] == 0].sample(5, random_state=42).iterrows():
    test_imgs.append(("NOR", row, "Normal"))

print("=" * 80)
print("🔍 POOLING STRATEGY COMPARISON — 4 methods")
print(f"   {len(test_imgs)} images, max_tokens=1000")
print("=" * 80)

# Collect embeddings per strategy
all_embs = {s: [] for s in ['last_30', 'last_10', 'last_1', 'layer-2_30']}
all_labels = []
all_names = []

for idx, (group, row, primary) in enumerate(test_imgs):
    # gt_dict is kept ONLY for display / evaluation — it is NOT fed to the model
    gt_dict = {sn: int(row[col]) if col in row.index else 0
               for col, sn in zip(ALL_PAT_COLS, ALL_PAT_SHORT)}
    gt_str = "+".join([s for s in ALL_PAT_SHORT if gt_dict[s] == 1]) or "Normal"

    print(f"\n  [{idx+1:2d}] {os.path.basename(row['image_path']):30s} [{group}] GT=[{gt_str}]", end="")

    results, expl, n_gen = extract_all_strategies(row['image_path'])

    print(f"  gen={n_gen} tokens, expl={len(expl)} chars")

    for strategy, emb in results.items():
        all_embs[strategy].append(emb)

    all_labels.append(group)
    all_names.append(os.path.basename(row['image_path']))

# ── Compare strategies ──
print(f"\n{'=' * 80}")
print("📊 STRATEGY COMPARISON")
print(f"{'=' * 80}")
print(f"\n  {'Strategy':<15s} {'PP':>8s} {'NN':>8s} {'PN':>8s} {'Gap_PP':>8s} {'Gap_NN':>8s} {'Centroid':>10s} {'Overlap':>10s}")
print(f"  {'─' * 85}")

n_pat = sum(1 for l in all_labels if l == "PAT")

best_strategy = None
best_gap = -999

for strategy in ['last_30', 'last_10', 'last_1', 'layer-2_30']:
    embs = np.array(all_embs[strategy])
    sim = cosine_similarity(embs)

    # PAT vs NOR
    pp = sim[:n_pat, :n_pat].copy(); np.fill_diagonal(pp, np.nan)
    nn = sim[n_pat:, n_pat:].copy(); np.fill_diagonal(nn, np.nan)
    pn = sim[:n_pat, n_pat:]

    mean_pp = np.nanmean(pp)
    mean_nn = np.nanmean(nn)
    mean_pn = np.nanmean(pn)
    gap_pp = mean_pp - mean_pn
    gap_nn = mean_nn - mean_pn

    # Centroid accuracy
    pat_centroid = embs[:n_pat].mean(axis=0)
    pat_centroid = pat_centroid / (np.linalg.norm(pat_centroid) + 1e-8)
    nor_centroid = embs[n_pat:].mean(axis=0)
    nor_centroid = nor_centroid / (np.linalg.norm(nor_centroid) + 1e-8)

    n_correct = 0
    for i in range(len(embs)):
        sp = cosine_similarity(embs[i].reshape(1,-1), pat_centroid.reshape(1,-1))[0,0]
        sn = cosine_similarity(embs[i].reshape(1,-1), nor_centroid.reshape(1,-1))[0,0]
        if (all_labels[i] == "PAT" and sp > sn) or (all_labels[i] == "NOR" and sn > sp):
            n_correct += 1

    # Label overlap
    from itertools import combinations
    overlap_s = []
    no_overlap_s = []
    for i, j in combinations(range(len(embs)), 2):
        gt_i = {s: int(test_imgs[i][1][col]) if col in test_imgs[i][1].index else 0
                for col, s in zip(ALL_PAT_COLS, ALL_PAT_SHORT)}
        gt_j = {s: int(test_imgs[j][1][col]) if col in test_imgs[j][1].index else 0
                for col, s in zip(ALL_PAT_COLS, ALL_PAT_SHORT)}
        shared = sum(1 for s in ALL_PAT_SHORT if gt_i[s]==1 and gt_j[s]==1)
        s_val = cosine_similarity(embs[i].reshape(1,-1), embs[j].reshape(1,-1))[0,0]
        if shared > 0:
            overlap_s.append(s_val)
        else:
            no_overlap_s.append(s_val)

    overlap_gap = np.mean(overlap_s) - np.mean(no_overlap_s) if overlap_s and no_overlap_s else 0

    # Combined score for ranking
    combined = gap_pp + overlap_gap
    if combined > best_gap:
        best_gap = combined
        best_strategy = strategy

    print(f"  {strategy:<15s} {mean_pp:8.4f} {mean_nn:8.4f} {mean_pn:8.4f} "
          f"{gap_pp:8.4f} {gap_nn:8.4f} {n_correct:>5d}/10   {overlap_gap:10.4f}")

print(f"\n  🏆 Best strategy: {best_strategy} (combined gap = {best_gap:.4f})")
print(f"\n  Use this for full extraction.")

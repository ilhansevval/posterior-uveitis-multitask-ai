# ============================================================================
# PATOLOJI EMBEDDING QUALITY ANALYSIS — MedGemma-4B + Reference
# Same metrics as Qwen, paper-ready output
# ============================================================================

import os
import numpy as np
import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings('ignore')

# CONFIG — MEDGEMMA path
DATA_ROOT = r'C:\Users\gtu\Documents\cerrahpasa\files\files'
EMB_DIR = os.path.join(DATA_ROOT, 'teacher_embeddings_medgemma')   # ★ MedGemma
EMB_PATH = os.path.join(EMB_DIR, 'teacher_embeddings.npy')
META_PATH = os.path.join(EMB_DIR, 'teacher_metadata.csv')

ALL_PAT_SHORT = ['DKS', 'ODB', 'VI', 'MÖ', 'DDB', 'RI', 'HEM', 'PVK']
RARE = ['DDB', 'RI', 'HEM', 'PVK']

print(f"📂 Loading: {EMB_DIR}")
embs = np.load(EMB_PATH)
meta = pd.read_csv(META_PATH, encoding='utf-8-sig')
embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)

labels = meta[[f'gt_{sn}' for sn in ALL_PAT_SHORT]].values.astype(int)
patient_ids = meta['patient_id'].values
print(f"   Embeddings: {embs.shape}, Samples: {len(meta)}")

sim_matrix = cosine_similarity(embs_norm)

# ─────────────────────────────────────────────
# GLOBAL METRICS
# ─────────────────────────────────────────────
print(f"\n{'='*70}\n📊 GLOBAL METRICS — MedGemma-4B + Ref\n{'='*70}")

n = len(labels)
intra_sims, inter_sims, partial_sims = [], [], []
for i in range(n):
    for j in range(i+1, n):
        l_i, l_j = labels[i], labels[j]
        inter = np.logical_and(l_i, l_j).sum()
        uni = np.logical_or(l_i, l_j).sum()
        if uni == 0: continue
        jaccard = inter / uni
        s = sim_matrix[i, j]
        if jaccard == 1.0: intra_sims.append(s)
        elif jaccard == 0.0: inter_sims.append(s)
        else: partial_sims.append(s)

intra_sims = np.array(intra_sims); inter_sims = np.array(inter_sims)
multi_gap = intra_sims.mean() - inter_sims.mean()

sim_no_self = sim_matrix.copy()
np.fill_diagonal(sim_no_self, -1)

nn1_exact = 0
nn1_jaccard = 0
for i in range(n):
    j = sim_no_self[i].argmax()
    l_i, l_j = labels[i], labels[j]
    if np.array_equal(l_i, l_j):
        nn1_exact += 1
    inter = np.logical_and(l_i, l_j).sum()
    uni = np.logical_or(l_i, l_j).sum()
    nn1_jaccard += inter / max(uni, 1)

nn1_exact /= n
nn1_jaccard /= n

print(f"  Multi-label gap:        {multi_gap:+.4f}")
print(f"  NN-1 exact match:       {nn1_exact:.4f}")
print(f"  NN-1 mean Jaccard:      {nn1_jaccard:.4f}")

# ─────────────────────────────────────────────
# PER-CLASS METRICS
# ─────────────────────────────────────────────
print(f"\n{'='*70}\n📊 PER-CLASS METRICS\n{'='*70}")

results = {}
gkf = GroupKFold(n_splits=5)

for i_lbl, sn in enumerate(ALL_PAT_SHORT):
    y = labels[:, i_lbl]
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    if len(pos_idx) < 2:
        continue

    # Per-label intra/inter sim
    pp_sims = []
    for i in range(len(pos_idx)):
        for j in range(i+1, len(pos_idx)):
            pp_sims.append(sim_matrix[pos_idx[i], pos_idx[j]])
    pp_mean = np.mean(pp_sims) if pp_sims else 0.0
    pn_block = sim_matrix[np.ix_(pos_idx, neg_idx)]
    pn_mean = pn_block.mean()
    gap = pp_mean - pn_mean

    # Per-label NN-1
    nn1_correct = 0
    for i in pos_idx:
        j = sim_no_self[i].argmax()
        if labels[j, i_lbl] == 1:
            nn1_correct += 1
    nn1_acc = nn1_correct / len(pos_idx)

    # LP F1, AUC, AP
    f1s, aucs, aps = [], [], []
    for tr, va in gkf.split(embs_norm, y, groups=patient_ids):
        if y[tr].sum() == 0 or y[va].sum() == 0:
            continue
        clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
        clf.fit(embs_norm[tr], y[tr])
        y_pred = clf.predict(embs_norm[va])
        y_prob = clf.predict_proba(embs_norm[va])[:, 1]
        f1s.append(f1_score(y[va], y_pred, zero_division=0))
        try: aucs.append(roc_auc_score(y[va], y_prob))
        except: aucs.append(0.0)
        try: aps.append(average_precision_score(y[va], y_prob))
        except: aps.append(0.0)

    # Centroid
    pos_centroid = embs_norm[pos_idx].mean(axis=0)
    pos_centroid /= np.linalg.norm(pos_centroid) + 1e-8
    neg_centroid = embs_norm[neg_idx].mean(axis=0)
    neg_centroid /= np.linalg.norm(neg_centroid) + 1e-8
    pred = (embs_norm @ pos_centroid > embs_norm @ neg_centroid).astype(int)
    cent_acc = (pred == y).mean()

    results[sn] = {
        'N': int(y.sum()),
        'gap': float(gap),
        'nn1': float(nn1_acc),
        'lp_f1_mean': float(np.mean(f1s)),
        'lp_f1_std': float(np.std(f1s)),
        'lp_auc': float(np.mean(aucs)),
        'lp_ap': float(np.mean(aps)),
        'centroid_acc': float(cent_acc),
        'is_rare': sn in RARE,
    }

# Print per-class
print(f"\n  {'Class':>6s} {'N':>4s} {'Gap':>8s} {'NN-1':>7s} "
      f"{'LP F1 ± std':>14s} {'LP AUC':>8s} {'LP AP':>8s} {'Cent Acc':>9s}")
print(f"  {'─'*78}")
for sn in ALL_PAT_SHORT:
    if sn not in results: continue
    r = results[sn]
    rare_mark = " *" if r['is_rare'] else "  "
    print(f"  {sn:>6s}{rare_mark} {r['N']:>4d} {r['gap']:>+8.4f} {r['nn1']:>7.4f} "
          f"{r['lp_f1_mean']:.3f} ± {r['lp_f1_std']:.3f}   "
          f"{r['lp_auc']:>8.4f} {r['lp_ap']:>8.4f} {r['centroid_acc']:>9.4f}")

def macro(keys, key):
    vals = [results[k][key] for k in keys if k in results]
    return float(np.mean(vals)) if vals else 0.0

all_keys = list(results.keys())
common_keys = [k for k in all_keys if not results[k]['is_rare']]
rare_keys = [k for k in all_keys if results[k]['is_rare']]

print(f"  {'─'*78}")
for label, keys in [('Common (4)', common_keys), ('Rare (4) *', rare_keys), ('ALL (8)', all_keys)]:
    print(f"  {label:>9s}      {macro(keys,'gap'):>+8.4f} {macro(keys,'nn1'):>7.4f} "
          f"{macro(keys,'lp_f1_mean'):.3f}           "
          f"{macro(keys,'lp_auc'):>8.4f} {macro(keys,'lp_ap'):>8.4f} {macro(keys,'centroid_acc'):>9.4f}")

print(f"\n  Global multi-label gap:    {multi_gap:+.4f}")
print(f"  Global NN-1 exact match:   {nn1_exact:.4f}")

# ─────────────────────────────────────────────
# COMPARISON: MedGemma vs Qwen+Ref
# ─────────────────────────────────────────────
print(f"\n{'='*70}\n📊 MedGemma vs Qwen+Ref COMPARISON\n{'='*70}")

QWEN_REF = {
    'macro_all':    {'gap': 0.0045, 'nn1': 0.7188, 'lp_f1': 0.4496, 'lp_auc': 0.8814, 'lp_ap': 0.5704},
    'macro_rare':   {'gap':-0.0072, 'nn1': 0.6104, 'lp_f1': 0.2853, 'lp_auc': 0.8875, 'lp_ap': 0.4482},
    'macro_common': {'gap': 0.0161, 'nn1': 0.8272, 'lp_f1': 0.6142, 'lp_auc': 0.8754, 'lp_ap': 0.6926},
    'global_gap': 0.0437,
    'nn1_exact': 0.8146,
}

MEDGEMMA = {
    'macro_all':    {'gap': macro(all_keys,'gap'), 'nn1': macro(all_keys,'nn1'),
                     'lp_f1': macro(all_keys,'lp_f1_mean'), 'lp_auc': macro(all_keys,'lp_auc'),
                     'lp_ap': macro(all_keys,'lp_ap')},
    'macro_rare':   {'gap': macro(rare_keys,'gap'), 'nn1': macro(rare_keys,'nn1'),
                     'lp_f1': macro(rare_keys,'lp_f1_mean'), 'lp_auc': macro(rare_keys,'lp_auc'),
                     'lp_ap': macro(rare_keys,'lp_ap')},
    'macro_common': {'gap': macro(common_keys,'gap'), 'nn1': macro(common_keys,'nn1'),
                     'lp_f1': macro(common_keys,'lp_f1_mean'), 'lp_auc': macro(common_keys,'lp_auc'),
                     'lp_ap': macro(common_keys,'lp_ap')},
    'global_gap': multi_gap,
    'nn1_exact': nn1_exact,
}

print(f"\n  {'Metric':<22s} {'MedGemma':>10s} {'Qwen+Ref':>10s} {'Δ (MG-Qwen)':>15s}")
print(f"  {'─'*60}")
print(f"  {'Global multi-gap':<22s} {MEDGEMMA['global_gap']:>+10.4f} "
      f"{QWEN_REF['global_gap']:>+10.4f} {MEDGEMMA['global_gap']-QWEN_REF['global_gap']:>+15.4f}")
print(f"  {'NN-1 exact match':<22s} {MEDGEMMA['nn1_exact']:>10.4f} "
      f"{QWEN_REF['nn1_exact']:>10.4f} {MEDGEMMA['nn1_exact']-QWEN_REF['nn1_exact']:>+15.4f}")
print(f"  {'─'*60}")
for grp_name, grp_key in [('Common', 'macro_common'), ('Rare', 'macro_rare'), ('All-8', 'macro_all')]:
    print(f"  -- {grp_name} --")
    for m in ['gap', 'nn1', 'lp_f1', 'lp_auc', 'lp_ap']:
        mg = MEDGEMMA[grp_key][m]
        qr = QWEN_REF[grp_key][m]
        delta = mg - qr
        print(f"  {m:<22s} {mg:>10.4f} {qr:>10.4f} {delta:>+15.4f}")
    print()

# ─────────────────────────────────────────────
# UNIFIED PAPER TABLE (3 model row)
# ─────────────────────────────────────────────
print(f"\n{'='*70}\n📋 PAPER TABLE — All Models Combined\n{'='*70}\n")

QWEN_NOREF = {  # Önceki sonuç
    'macro_all_lpf1': 0.4438, 'macro_rare_lpf1': 0.2558,
    'macro_all_lpauc': 0.8711, 'nn1': 0.8021, 'gap': 0.0165
}

print(f"  {'Model':<22s} {'Type':<14s} {'Param':>6s} {'Med':>4s} {'Ret':>4s} "
      f"{'F1-all':>8s} {'F1-rare':>8s} {'AUC':>7s} {'NN-1':>7s} {'Gap':>9s}")
print(f"  {'─'*100}")

print(f"  {'Qwen2.5-VL':<22s} {'Gen.VLM':<14s} {'3B':>6s} {'✗':>4s} {'✗':>4s} "
      f"{QWEN_NOREF['macro_all_lpf1']:>8.3f} "
      f"{QWEN_NOREF['macro_rare_lpf1']:>8.3f} "
      f"{QWEN_NOREF['macro_all_lpauc']:>7.3f} "
      f"{QWEN_NOREF['nn1']:>7.3f} "
      f"{QWEN_NOREF['gap']:>+9.4f}")

print(f"  {'Qwen2.5-VL + Ref':<22s} {'Gen.VLM(DP)':<14s} {'3B':>6s} {'✗':>4s} {'✗':>4s} "
      f"{QWEN_REF['macro_all']['lp_f1']:>8.3f} "
      f"{QWEN_REF['macro_rare']['lp_f1']:>8.3f} "
      f"{QWEN_REF['macro_all']['lp_auc']:>7.3f} "
      f"{QWEN_REF['nn1_exact']:>7.3f} "
      f"{QWEN_REF['global_gap']:>+9.4f}")

print(f"  {'MedGemma-4B + Ref':<22s} {'Gen.VLM(DP)':<14s} {'4B':>6s} {'✓':>4s} {'✗':>4s} "
      f"{MEDGEMMA['macro_all']['lp_f1']:>8.3f} "
      f"{MEDGEMMA['macro_rare']['lp_f1']:>8.3f} "
      f"{MEDGEMMA['macro_all']['lp_auc']:>7.3f} "
      f"{MEDGEMMA['nn1_exact']:>7.3f} "
      f"{MEDGEMMA['global_gap']:>+9.4f}")

print(f"  {'─'*100}")
print(f"  {'InternVL3-2B':<22s} {'Gen.VLM(DP)':<14s} {'2B':>6s} {'✗':>4s} {'✗':>4s}     ...")
print(f"  {'RETFound':<22s} {'SSL ViT':<14s} {'300M':>6s} {'✓':>4s} {'✓':>4s}     ...")
print(f"  {'FLAIR':<22s} {'Contrastive':<14s} {'150M':>6s} {'✓':>4s} {'✓':>4s}     ...")

# ─────────────────────────────────────────────
# LATEX
# ─────────────────────────────────────────────
print(f"\n{'='*70}\n📋 LATEX — Per-class detail (MedGemma)\n{'='*70}\n")

print("\\begin{table}[h]")
print("\\centering")
print("\\caption{Per-class embedding quality for MedGemma-4B + Reference images.}")
print("\\label{tab:emb_quality_medgemma}")
print("\\begin{tabular}{lccccccc}")
print("\\toprule")
print("Class & N & Gap & NN-1 & LP F1 $\\pm$ std & LP AUC & LP AP & Cent. Acc \\\\")
print("\\midrule")
for sn in ALL_PAT_SHORT:
    if sn not in results: continue
    r = results[sn]
    rare = "$^{*}$" if r['is_rare'] else ""
    print(f"{sn}{rare} & {r['N']} & {r['gap']:+.4f} & {r['nn1']:.3f} & "
          f"{r['lp_f1_mean']:.3f} $\\pm$ {r['lp_f1_std']:.3f} & "
          f"{r['lp_auc']:.3f} & {r['lp_ap']:.3f} & {r['centroid_acc']:.3f} \\\\")
print("\\midrule")
for label, keys in [('Common', common_keys), ('Rare$^{*}$', rare_keys), ('Macro', all_keys)]:
    print(f"\\textbf{{{label}}} & -- & {macro(keys,'gap'):+.4f} & {macro(keys,'nn1'):.3f} & "
          f"{macro(keys,'lp_f1_mean'):.3f} & "
          f"{macro(keys,'lp_auc'):.3f} & {macro(keys,'lp_ap'):.3f} & {macro(keys,'centroid_acc'):.3f} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

# Save JSON
summary = {
    'model': 'MedGemma-4B-Ref',
    'n_samples': int(len(embs)),
    'emb_dim': int(embs.shape[1]),
    'global': {
        'multi_label_gap': float(multi_gap),
        'nn1_exact_match': float(nn1_exact),
        'nn1_jaccard': float(nn1_jaccard),
    },
    'macro_all': MEDGEMMA['macro_all'],
    'macro_rare': MEDGEMMA['macro_rare'],
    'macro_common': MEDGEMMA['macro_common'],
    'per_class': results,
}
out_path = os.path.join(EMB_DIR, 'embedding_quality_summary_v2.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\n💾 Saved: {out_path}")

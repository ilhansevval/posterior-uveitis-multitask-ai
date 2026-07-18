# ============================================================================
# UMAP EMBEDDING VISUALIZATION — 6 foundation models (paper figure)
# Run on Windows (same machine that holds the embeddings).
#
# Output: umap_embeddings.pdf + umap_embeddings.png  (in DATA_ROOT)
#
# Layout: 2 rows x 3 cols
#   Row 1 (encoder-only):  BiomedCLIP | RETFound-CFP | Qwen2.5-VL (no ref)
#   Row 2 (generative+DP):  Qwen+Ref  | MedGemma+Ref | Phi-3.5+Ref
#
# Coloring: each image is colored by its RAREST present pathology,
# so rare-class separation stands out. Images with no pathology -> "Normal".
#
# Install deps (once):
#   pip install umap-learn
# ============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

try:
    import umap
except ImportError:
    raise ImportError("Run:  pip install umap-learn")

# ----------------------------------------------------------------------
# CONFIG — paths exactly as in your analysis scripts
# ----------------------------------------------------------------------
DATA_ROOT = r'C:\Users\gtu\Documents\cerrahpasa\files\files'

# (title, folder)  -- order = panel order (row-major, 2x3)
MODELS = [
    ("BiomedCLIP",            'teacher_embeddings_biomedclip'),
    ("RETFound-CFP",          'teacher_embeddings_retfound_cfp'),
    ("Qwen2.5-VL",            'teacher_embeddings'),            # no-ref
    ("Qwen2.5-VL + Ref",      'teacher_embeddings_ref'),
    ("MedGemma-4B + Ref",     'teacher_embeddings_medgemma'),
    ("Phi-3.5-Vision + Ref",  'teacher_embeddings_phi'),
]

ALL_PAT_SHORT = ['DKS', 'ODB', 'VI', 'MÖ', 'DDB', 'RI', 'HEM', 'PVK']
# rarity order: rarest first
# counts: HEM 14, PVK 17, RI 26, DDB 56, MÖ 57, VI 84, ODB 151, DKS 296
RARITY_ORDER = ['HEM', 'PVK', 'RI', 'DDB', 'MÖ', 'VI', 'ODB', 'DKS']

DISPLAY = {
    'DKS':   'Diffuse Capillary Leakage',
    'ODB':   'Optic Disc Staining',
    'VI':    'Vitreous Inflammation',
    'MÖ':    'Macular Edema',
    'DDB':   'Vessel Wall Staining',
    'RI':    'Retinal Infiltrate',
    'HEM':   'Hemorrhage',
    'PVK':   'Perivascular Sheathing',
    'NORMAL': 'Normal / no finding',
}

COLORS = {
    'DKS':   '#1f77b4',
    'ODB':   '#ff7f0e',
    'VI':    '#2ca02c',
    'MÖ':    '#9467bd',
    'DDB':   '#8c564b',
    'RI':    '#e377c2',
    'HEM':   '#d62728',
    'PVK':   '#17becf',
    'NORMAL':'#cccccc',
}

# ----------------------------------------------------------------------
# UMAP hyperparameters
# n_neighbors : controls local vs global structure (15 = default)
# min_dist    : how tightly points pack together (0.1 = default)
# metric      : cosine matches your L2-normalised embeddings
# ----------------------------------------------------------------------
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST    = 0.1
UMAP_METRIC      = 'cosine'
UMAP_SEED        = 42

# ----------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def rarest_present_label(label_row):
    """Return the rarest pathology present in this multi-label row, else NORMAL."""
    present = {sn for sn, v in zip(ALL_PAT_SHORT, label_row) if v == 1}
    for sn in RARITY_ORDER:
        if sn in present:
            return sn
    return 'NORMAL'


def load_model(folder):
    emb_dir = os.path.join(DATA_ROOT, folder)
    embs = np.load(os.path.join(emb_dir, 'teacher_embeddings.npy'))
    meta = pd.read_csv(os.path.join(emb_dir, 'teacher_metadata.csv'),
                       encoding='utf-8-sig')
    labels = meta[[f'gt_{sn}' for sn in ALL_PAT_SHORT]].values.astype(int)
    point_label = np.array([rarest_present_label(r) for r in labels])
    # L2 normalize (cosine geometry, matches your analysis)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    return embs, point_label


fig, axes = plt.subplots(2, 3, figsize=(7.16, 5.0))
axes = axes.ravel()

LEGEND_ORDER = ['HEM', 'PVK', 'RI', 'DDB', 'MÖ', 'VI', 'ODB', 'DKS', 'NORMAL']

for ax, (title, folder) in zip(axes, MODELS):
    emb_dir = os.path.join(DATA_ROOT, folder)
    if not os.path.exists(os.path.join(emb_dir, 'teacher_embeddings.npy')):
        ax.text(0.5, 0.5, f"missing:\n{folder}", ha='center', va='center',
                fontsize=7, transform=ax.transAxes)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        continue

    print(f"[UMAP] {title} ...")
    embs, point_label = load_model(folder)

    # n_neighbors must be < n_samples
    n_neighbors = min(UMAP_N_NEIGHBORS, len(embs) - 1)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_SEED,
        low_memory=False,
    )
    xy = reducer.fit_transform(embs)

    # plot NORMAL first (background), then common, then rare on top
    draw_order = ['NORMAL', 'DKS', 'ODB', 'VI', 'MÖ', 'DDB', 'RI', 'PVK', 'HEM']
    for cls in draw_order:
        m = point_label == cls
        if not m.any():
            continue
        is_rare = cls in ('HEM', 'PVK', 'RI', 'DDB')
        ax.scatter(xy[m, 0], xy[m, 1],
                   s=(14 if is_rare else 6),
                   c=COLORS[cls],
                   alpha=(0.95 if is_rare else 0.45),
                   edgecolors=('black' if is_rare else 'none'),
                   linewidths=(0.3 if is_rare else 0),
                   zorder=(3 if is_rare else 1))

    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)

# shared legend at the bottom
handles = [Line2D([0], [0], marker='o', linestyle='',
                  markerfacecolor=COLORS[c], markeredgecolor='black',
                  markersize=6, label=DISPLAY[c])
           for c in LEGEND_ORDER]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=6.5,
           frameon=False, bbox_to_anchor=(0.5, -0.02),
           handletextpad=0.3, columnspacing=1.0)

plt.tight_layout(rect=[0, 0.07, 1, 1])

out_pdf = os.path.join(DATA_ROOT, 'umap_embeddings.pdf')
out_png = os.path.join(DATA_ROOT, 'umap_embeddings.png')
plt.savefig(out_pdf, bbox_inches='tight')
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"\nSaved:\n  {out_pdf}\n  {out_png}")

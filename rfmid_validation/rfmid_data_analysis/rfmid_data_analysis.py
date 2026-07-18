#!/usr/bin/env python3
"""
🔍 RFMiD 1.0 VERİ ANALİZİ
   External validation öncesi keşif:
   - 3 set (train/val/test) CSV'lerini oku
   - Tüm label listesini çıkar (45-46 hastalık)
   - Senin 8 label'ınla overlapping olanları bul
   - Her overlapping label için pozitif sayımı
   - Görüntü sayısı / format kontrolü
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
RFMID_ROOT = r'C:\Users\gtu\Downloads\A. RFMiD_All_Classes_Dataset'
IMG_DIR = os.path.join(RFMID_ROOT, '1. Original Images')
GT_DIR = os.path.join(RFMID_ROOT, '2. Groundtruths')

CSV_FILES = {
    'Training':   os.path.join(GT_DIR, 'a. RFMiD_Training_Labels.csv'),
    'Validation': os.path.join(GT_DIR, 'b. RFMiD_Validation_Labels.csv'),
    'Testing':    os.path.join(GT_DIR, 'c. RFMiD_Testing_Labels.csv'),
}
IMG_SUBDIRS = {
    'Training':   os.path.join(IMG_DIR, 'a. Training Set'),
    'Validation': os.path.join(IMG_DIR, 'b. Validation Set'),
    'Testing':    os.path.join(IMG_DIR, 'c. Testing Set'),
}

# Senin label'ların → RFMiD 1.0 olası karşılıkları (kesinleştireceğiz)
# RFMiD 1.0 label açıklamaları (literatürden):
RFMID_LABEL_DESC = {
    'DR': 'Diabetic Retinopathy', 'ARMD': 'Age-related Macular Degeneration',
    'MH': 'Media Haze', 'DN': 'Drusen', 'MYA': 'Myopia',
    'BRVO': 'Branch Retinal Vein Occlusion', 'TSLN': 'Tessellation',
    'ERM': 'Epiretinal Membrane', 'LS': 'Laser Scars', 'MS': 'Macular Scar',
    'CSR': 'Central Serous Retinopathy', 'ODC': 'Optic Disc Cupping',
    'CRVO': 'Central Retinal Vein Occlusion', 'TV': 'Tortuous Vessels',
    'AH': 'Asteroid Hyalosis', 'ODP': 'Optic Disc Pallor',
    'ODE': 'Optic Disc Edema', 'ST': 'Optociliary Shunt',
    'AION': 'Anterior Ischemic Optic Neuropathy', 'PT': 'Parafoveal Telangiectasia',
    'RT': 'Retinal Traction', 'RS': 'Retinitis', 'CRS': 'Chorioretinitis',
    'EDN': 'Exudation', 'RPEC': 'RPE Changes', 'MHL': 'Macular Hole',
    'RP': 'Retinitis Pigmentosa', 'CWS': 'Cotton Wool Spots', 'CB': 'Coloboma',
    'ODPM': 'Optic Disc Pit Maculopathy', 'PRH': 'Preretinal Hemorrhage',
    'MNF': 'Myelinated Nerve Fibers', 'HR': 'Hemorrhagic Retinopathy',
    'CRAO': 'Central Retinal Artery Occlusion', 'TD': 'Tilted Disc',
    'CME': 'Cystoid Macular Edema', 'PTCR': 'Post Traumatic Choroidal Rupture',
    'CF': 'Choroidal Folds', 'VH': 'Vitreous Hemorrhage', 'MCA': 'Macroaneurysm',
    'VS': 'Vasculitis', 'BRAO': 'Branch Retinal Artery Occlusion',
    'PLQ': 'Plaque', 'HPED': 'Hemorrhagic Pigment Epithelial Detachment',
    'CL': 'Collateral',
}

# Senin 8 label → RFMiD overlapping mapping (analiz sonrası kesinleşecek)
OVERLAP_MAPPING = {
    'MÖ (Makula Ödemi)':            ['CME'],          # Cystoid Macular Edema
    'ODB (Optik Disk Boyanması)':  ['ODE'],          # Optic Disc Edema
    'RI (Retinal İnfiltrat)':      ['RS', 'CRS'],    # Retinitis / Chorioretinitis
    'DDB (Damar Duvarı Boyanması)':['VS'],           # Vasculitis
    'HEM (Hemoraji)':              ['HR', 'VH', 'PRH'], # Hemorrhagic Retino / Vitreous / Preretinal
}

print("=" * 80)
print("🔍 RFMiD 1.0 VERİ ANALİZİ")
print("=" * 80)

# ─────────────────────────────────────────────
# 1. CSV'leri oku
# ─────────────────────────────────────────────
print(f"\n{'='*80}\n1. CSV DOSYALARI\n{'='*80}")
dfs = {}
for split, path in CSV_FILES.items():
    if not os.path.exists(path):
        print(f"  ❌ {split}: BULUNAMADI → {path}")
        continue
    df = pd.read_csv(path)
    dfs[split] = df
    print(f"  ✓ {split:12s}: {len(df):5d} satır, {len(df.columns)} kolon")

if not dfs:
    print("\n  ⚠️ Hiç CSV okunamadı! Yolları kontrol et.")
    raise SystemExit

# ─────────────────────────────────────────────
# 2. Kolon yapısı
# ─────────────────────────────────────────────
print(f"\n{'='*80}\n2. KOLON YAPISI (Training set)\n{'='*80}")
train_df = dfs.get('Training', list(dfs.values())[0])
print(f"\n  Kolonlar ({len(train_df.columns)}):")
print(f"  {list(train_df.columns)}")

# İlk birkaç satır
print(f"\n  İlk 3 satır:")
print(train_df.head(3).to_string())

# Label kolonları (ID ve Disease_Risk hariç)
meta_cols = [c for c in train_df.columns if c.upper() in ['ID', 'DISEASE_RISK']]
label_cols = [c for c in train_df.columns if c not in meta_cols]
print(f"\n  Meta kolonlar: {meta_cols}")
print(f"  Label kolonları ({len(label_cols)}): {label_cols}")

# ─────────────────────────────────────────────
# 3. Tüm label dağılımı (3 set birleşik)
# ─────────────────────────────────────────────
print(f"\n{'='*80}\n3. TÜM LABEL DAĞILIMI (3 set birleşik)\n{'='*80}")
all_df = pd.concat(dfs.values(), ignore_index=True)
print(f"  Toplam görüntü: {len(all_df)}")

print(f"\n  {'Label':6s} {'Açıklama':40s} {'N':>5s} {'%':>6s}")
print(f"  {'─'*62}")
label_counts = {}
for col in label_cols:
    n = int(all_df[col].sum())
    label_counts[col] = n
    desc = RFMID_LABEL_DESC.get(col.upper(), '?')
    print(f"  {col:6s} {desc:40s} {n:5d} {n/len(all_df)*100:5.1f}%")

# ─────────────────────────────────────────────
# 4. Senin label'larınla OVERLAP
# ─────────────────────────────────────────────
print(f"\n{'='*80}\n4. SENİN LABEL'LARINLA OVERLAP\n{'='*80}")
print(f"\n  {'Senin Label':32s} {'RFMiD':18s} {'N (3 set)':>10s} {'Var mı?'}")
print(f"  {'─'*75}")

available_cols = set(c.upper() for c in label_cols)
overlap_summary = {}
for my_label, rfmid_options in OVERLAP_MAPPING.items():
    found = []
    total_n = 0
    for opt in rfmid_options:
        # case-insensitive eşleştir
        matched = [c for c in label_cols if c.upper() == opt.upper()]
        if matched:
            real_col = matched[0]
            n = label_counts.get(real_col, 0)
            found.append(f"{real_col}({n})")
            total_n += n
    if found:
        status = "✅ VAR"
        overlap_summary[my_label] = total_n
    else:
        status = "❌ YOK"
    rfmid_str = ", ".join(found) if found else ", ".join(rfmid_options) + " (yok)"
    print(f"  {my_label:32s} {rfmid_str:18s} {total_n:>10d} {status}")

# ─────────────────────────────────────────────
# 5. Set bazında overlap dağılımı
# ─────────────────────────────────────────────
print(f"\n{'='*80}\n5. SET BAZINDA OVERLAP LABEL DAĞILIMI\n{'='*80}")
print(f"\n  {'RFMiD Label':8s} {'Train':>8s} {'Val':>8s} {'Test':>8s} {'TOPLAM':>8s}")
print(f"  {'─'*46}")

# tüm overlap RFMiD kolonlarını topla
all_overlap_cols = []
for opts in OVERLAP_MAPPING.values():
    all_overlap_cols.extend(opts)
all_overlap_cols = list(dict.fromkeys(all_overlap_cols))  # unique, order preserved

for opt in all_overlap_cols:
    matched = [c for c in label_cols if c.upper() == opt.upper()]
    if not matched:
        print(f"  {opt:8s} {'—':>8s} {'—':>8s} {'—':>8s} {'YOK':>8s}")
        continue
    real_col = matched[0]
    row = f"  {real_col:8s}"
    total = 0
    for split in ['Training', 'Validation', 'Testing']:
        if split in dfs and real_col in dfs[split].columns:
            n = int(dfs[split][real_col].sum())
            total += n
            row += f" {n:>8d}"
        else:
            row += f" {'—':>8s}"
    row += f" {total:>8d}"
    print(row)

# ─────────────────────────────────────────────
# 6. Görüntü kontrolü
# ─────────────────────────────────────────────
print(f"\n{'='*80}\n6. GÖRÜNTÜ KONTROLÜ\n{'='*80}")
for split, subdir in IMG_SUBDIRS.items():
    if not os.path.exists(subdir):
        print(f"  ❌ {split}: klasör yok → {subdir}")
        continue
    imgs = [f for f in os.listdir(subdir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    exts = set(os.path.splitext(f)[1].lower() for f in imgs)
    print(f"  ✓ {split:12s}: {len(imgs):5d} görüntü, formatlar: {exts}")
    if imgs:
        print(f"      Örnek: {imgs[:3]}")

# ─────────────────────────────────────────────
# 7. ÖZET
# ─────────────────────────────────────────────
print(f"\n{'='*80}\n📊 ÖZET\n{'='*80}")
print(f"\n  Toplam RFMiD görüntü: {len(all_df)}")
print(f"  Toplam label sayısı:  {len(label_cols)}")
print(f"\n  Overlapping label'lar (senin verinle eşleşen):")
for my_label, n in overlap_summary.items():
    print(f"    {my_label:32s}: {n} pozitif")

n_overlap = len(overlap_summary)
print(f"\n  → {n_overlap} label external validation için kullanılabilir")
print(f"\n{'='*80}")
print("✅ ANALİZ TAMAMLANDI")
print(f"{'='*80}")

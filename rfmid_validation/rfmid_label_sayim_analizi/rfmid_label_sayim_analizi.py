#!/usr/bin/env python3
"""
🔢 RFMiD LABEL SAYIM ANALİZİ
   Her label için train/test pozitif sayısı + eğitilebilirlik kararı
   Amaç: normal+KD karşılaştırması için hangi label'lar uygun, netleştir
"""

import os
import pandas as pd
import numpy as np

RFMID_ROOT = r'C:\Users\gtu\Downloads\A. RFMiD_All_Classes_Dataset'
GT_DIR = os.path.join(RFMID_ROOT, '2. Groundtruths')

CSV = {
    'train': os.path.join(GT_DIR, 'a. RFMiD_Training_Labels.csv'),
    'val':   os.path.join(GT_DIR, 'b. RFMiD_Validation_Labels.csv'),
    'test':  os.path.join(GT_DIR, 'c. RFMiD_Testing_Labels.csv'),
}

# Senin overlapping label'ların
OVERLAP_COLS = {'ODE', 'RS', 'CRS', 'CME', 'VS', 'HR', 'VH', 'PRH'}

dfs = {k: pd.read_csv(v) for k, v in CSV.items()}
label_cols = [c for c in dfs['train'].columns if c not in ['ID', 'Disease_Risk']]

print("=" * 78)
print("🔢 RFMiD LABEL SAYIM ANALİZİ — Eğitilebilirlik")
print("=" * 78)
print(f"\n  Set boyutları: train={len(dfs['train'])}, val={len(dfs['val'])}, test={len(dfs['test'])}")

# Eğitilebilirlik eşikleri
MIN_TRAIN = 30   # train'de en az bu kadar pozitif
MIN_TEST = 10    # test'te en az bu kadar pozitif

print(f"\n  Eğitilebilirlik kriteri: train≥{MIN_TRAIN} VE test≥{MIN_TEST}")
print(f"\n  {'Label':6s} {'Train':>6s} {'Val':>5s} {'Test':>5s} {'Toplam':>7s} {'Durum':>14s} {'Overlap'}")
print(f"  {'─'*62}")

rows = []
for col in label_cols:
    ntr = int(dfs['train'][col].sum())
    nva = int(dfs['val'][col].sum())
    nte = int(dfs['test'][col].sum())
    ntot = ntr + nva + nte
    trainable = (ntr >= MIN_TRAIN) and (nte >= MIN_TEST)
    is_overlap = col in OVERLAP_COLS
    rows.append({'label': col, 'train': ntr, 'val': nva, 'test': nte,
                 'total': ntot, 'trainable': trainable, 'overlap': is_overlap})

# Toplam pozitife göre sırala
rows.sort(key=lambda r: -r['total'])

for r in rows:
    durum = "✅ EĞİTİLEBİLİR" if r['trainable'] else "❌ yetersiz"
    ov = "🎯 OVERLAP" if r['overlap'] else ""
    print(f"  {r['label']:6s} {r['train']:>6d} {r['val']:>5d} {r['test']:>5d} {r['total']:>7d} "
          f"{durum:>14s} {ov}")

# ─────────────────────────────────────────────
# ÖZET
# ─────────────────────────────────────────────
trainable = [r for r in rows if r['trainable']]
overlap_trainable = [r for r in rows if r['trainable'] and r['overlap']]
overlap_all = [r for r in rows if r['overlap']]

print(f"\n{'='*78}\n📊 ÖZET\n{'='*78}")
print(f"\n  Toplam label: {len(label_cols)}")
print(f"  Eğitilebilir label (train≥{MIN_TRAIN}, test≥{MIN_TEST}): {len(trainable)}")
print(f"\n  Eğitilebilir label listesi:")
print(f"    {[r['label'] for r in trainable]}")

print(f"\n  Senin overlapping label'ların ({len(overlap_all)}):")
for r in overlap_all:
    durum = "✅ eğitilebilir" if r['trainable'] else "❌ yetersiz"
    print(f"    {r['label']:5s}: train={r['train']:3d} test={r['test']:3d} → {durum}")

print(f"\n  → Overlapping VE eğitilebilir: {[r['label'] for r in overlap_trainable]}")

# Farklı eşiklerde kaç label eğitilebilir
print(f"\n  Farklı eşiklerde eğitilebilir label sayısı:")
for mt, mte in [(20, 5), (30, 10), (50, 15), (100, 30)]:
    cnt = sum(1 for r in rows if r['train'] >= mt and r['test'] >= mte)
    cnt_ov = sum(1 for r in rows if r['overlap'] and r['train'] >= mt and r['test'] >= mte)
    print(f"    train≥{mt:3d}, test≥{mte:2d}: {cnt:2d} label (overlap: {cnt_ov})")

print(f"\n{'='*78}\n✅ ANALİZ TAMAMLANDI\n{'='*78}")

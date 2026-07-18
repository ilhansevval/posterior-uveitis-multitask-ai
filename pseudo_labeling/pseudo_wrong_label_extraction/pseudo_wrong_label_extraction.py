# ============================================================================
# NOISE CONTROL: İkinci en yüksek olasılıklı label ile pseudo-label
# Beklenti: Macro F1 düşmeli (v6'nın 0.811'inden aşağı)
# ============================================================================

import os
import numpy as np
import pandas as pd

DATA_ROOT = r'C:\Users\gtu\Documents\cerrahpasa\files\files'
AGREED_CSV = os.path.join(DATA_ROOT, 'pseudo_labels_agreed.csv')

ALL_SHORT = ['DKS', 'ODB', 'VI', 'MÖ', 'DDB', 'RI', 'HEM', 'PVK']

df = pd.read_csv(AGREED_CSV)
print(f"📂 Agreed pseudo-labels: {len(df)}")

# For each sample: swap to SECOND highest probability label
for idx, row in df.iterrows():
    probs = [(sn, row[f'prob_avg_{sn}']) for sn in ALL_SHORT]
    probs_sorted = sorted(probs, key=lambda x: -x[1])
    
    # First: highest prob label (correct pseudo-label)
    # Second: second highest prob label (WRONG label)
    best_label = probs_sorted[0][0]
    second_label = probs_sorted[1][0]
    second_prob = probs_sorted[1][1]
    
    # Set all gt_ to 0, then set second-best as the label
    for sn in ALL_SHORT:
        df.at[idx, f'gt_{sn}'] = 0.0
    
    # Use second-best label with its own probability as soft label
    df.at[idx, f'gt_{second_label}'] = second_prob
    df.at[idx, 'n_agreed'] = 1  # keep filter working
    
    # Track what we did
    df.at[idx, 'noise_from'] = best_label
    df.at[idx, 'noise_to'] = second_label
    df.at[idx, 'noise_prob'] = second_prob

# Stats
print(f"\n📊 NOISE LABEL DISTRIBUTION")
print(f"  Swap examples:")
for _, row in df.head(10).iterrows():
    print(f"    {row['noise_from']:5s} (correct) → {row['noise_to']:5s} (wrong, prob={row['noise_prob']:.3f})")

print(f"\n  Per-label noise counts (assigned as wrong label):")
for sn in ALL_SHORT:
    n = (df[f'gt_{sn}'] > 0).sum()
    avg = df.loc[df[f'gt_{sn}'] > 0, f'gt_{sn}'].mean() if n > 0 else 0
    print(f"    {sn:5s}: {n:5d} (avg prob={avg:.3f})")

# Save
noise_path = os.path.join(DATA_ROOT, 'pseudo_labels_noise.csv')
df.to_csv(noise_path, index=False)
print(f"\n💾 Noise pseudo-labels: {noise_path}")
print(f"\n   Next: Run v8 code with PSEUDO_PATH = 'pseudo_labels_noise.csv'")
print(f"   Save to: results_distillation_v8_noise")
print(f"   Expected: Macro F1 < 0.811 (should DECREASE)")

import os
import pandas as pd
import glob

DATA_ROOT = r'C:\Users\gtu\Documents\cerrahpasa\files\files'
CSV_PATH = os.path.join(DATA_ROOT, 'Resimler_birlesik_TF_plusYorum_clean_numeric.csv')

# Etiketli görüntüler
df = pd.read_csv(CSV_PATH, encoding='utf-8')
df.columns = [c.strip().replace('\xa0', '') for c in df.columns]
df['image_path'] = df.apply(
    lambda r: os.path.join(DATA_ROOT, str(r['Klasör']), str(r['Dosya ismi'])), axis=1)
labeled_paths = set(df['image_path'].values)
labeled_names = set(df['Dosya ismi'].values)

# Tüm görüntüler — numaralı klasörlerdeki jpg/png
all_images = []
for folder in os.listdir(DATA_ROOT):
    folder_path = os.path.join(DATA_ROOT, folder)
    if not os.path.isdir(folder_path): continue
    if not folder.isdigit(): continue  # sadece numaralı klasörler
    for img in glob.glob(os.path.join(folder_path, '*.jpg')) + glob.glob(os.path.join(folder_path, '*.png')):
        all_images.append({
            'folder': folder,
            'filename': os.path.basename(img),
            'image_path': img,
        })

all_df = pd.DataFrame(all_images)
all_paths = set(all_df['image_path'].values)

# Etiketlenmemiş = tüm - etiketli
unlabeled_paths = all_paths - labeled_paths

print(f"Toplam görüntü (numaralı klasörlerde): {len(all_images)}")
print(f"Etiketli: {len(labeled_paths)}")
print(f"Etiketlenmemiş: {len(unlabeled_paths)}")
print(f"Klasör sayısı: {all_df['folder'].nunique()}")

# Etiketlenmemiş verilerin klasör dağılımı
unlabeled_df = all_df[all_df['image_path'].isin(unlabeled_paths)]
print(f"\nEtiketlenmemiş klasör dağılımı (top 20):")
print(unlabeled_df['folder'].value_counts().head(20))

# Kaydet
unlabeled_df.to_csv(os.path.join(DATA_ROOT, 'unlabeled_images.csv'), index=False)
print(f"\nKaydedildi: unlabeled_images.csv ({len(unlabeled_df)} satır)")

# ============================================================================
# CELL 8b-PHI (SETUP) + CELL 9: BATCH EXTRACTION — Phi-3.5-Vision multi-image
# ============================================================================

import os, glob, time, gc
import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

import transformers
from transformers.cache_utils import DynamicCache

if not hasattr(DynamicCache, 'seen_tokens'):
    @property
    def seen_tokens(self):
        return self.get_seq_length()
    DynamicCache.seen_tokens = seen_tokens

if not hasattr(DynamicCache, 'get_max_length'):
    DynamicCache.get_max_length = lambda self: None

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_ROOT = r'C:\Users\gtu\Documents\cerrahpasa\files\files'
DATASET_CSV = os.path.join(DATA_ROOT, 'Resimler_birlesik_TF_plusYorum_clean_numeric.csv')
REF_ROOT = os.path.join(DATA_ROOT, 'fundus_choosen')
MODEL_PATH = "microsoft/Phi-3.5-vision-instruct"

ALL_PAT_SHORT = ['DKS', 'ODB', 'VI', 'MÖ', 'DDB', 'RI', 'HEM', 'PVK', 'RSLD', 'GV']
ALL_PAT_COLS = [
    'Diffüz kapiller sızıntı', 'Optik disk boyanması', 'Vitreus inflamasyonu',
    'Makula ödemi', 'Damar duvar boyanması', 'Retinal infiltrat',
    'Hemoraji', 'Perivasküler kılıflanma', 'Retina sinir lif defekti', 'Ghost vessel'
]

REF_FOLDER_MAP = {
    'ODB': 'Optik disk boyanması',
    'MÖ': 'makula ödemi',
    'DDB': 'damar duvar',
    'RI': 'Retinal infiltrat',
    'HEM': 'hemoraji',
    'PVK': 'Perivasküler kılıflanma',
}

LIGHT_CLEAN_EVERY = 3
HEAVY_CLEAN_EVERY = 50
MAX_SIZE = 360              # ★ 896 → 560 (OOM oranını çok azaltır)
NUM_CROPS = 2               # ★ 4 → 2 (her image için patch sayısı yarıya iner)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128"

vl_model = None
vl_processor = None


# ─────────────────────────────────────────────
# IMAGE RESIZE HELPER
# ─────────────────────────────────────────────
def resize_for_phi(img, max_size=MAX_SIZE):
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    if w >= h:
        new_w = max_size
        new_h = int(h * max_size / w)
    else:
        new_h = max_size
        new_w = int(w * max_size / h)
    return img.resize((new_w, new_h), Image.LANCZOS)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
print("📂 Loading dataset...")
df = pd.read_csv(DATASET_CSV, encoding='utf-8')
df.columns = [c.strip().replace('\xa0', '') for c in df.columns]
df['patient_id'] = df['Klasör'].astype(str)
df['image_name'] = df['Dosya ismi'].astype(str)
df['image_path'] = df.apply(
    lambda r: os.path.join(DATA_ROOT, r['patient_id'], r['image_name']), axis=1)
for c in ALL_PAT_COLS:
    df[c] = df[c].astype(int)
df['n_pathology'] = df[ALL_PAT_COLS].sum(axis=1)
df_pat = df[df['n_pathology'] > 0].reset_index(drop=True)
df_pat = df_pat[df_pat['image_path'].apply(os.path.exists)].reset_index(drop=True)
print(f"   Total: {len(df)} images, Pathological: {len(df_pat)}, Normal: {len(df)-len(df_pat)}")

# ─────────────────────────────────────────────
# LOAD REFERENCE IMAGES (resize ile, 2 ref korunuyor)
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# LOAD REFERENCE IMAGES (resize ile, 1 ref per label)
# ─────────────────────────────────────────────
print("\n📸 Loading reference images...")
ref_images = {}
for label_short, folder_name in REF_FOLDER_MAP.items():
    folder_path = os.path.join(REF_ROOT, folder_name)
    if not os.path.exists(folder_path):
        print(f"   ⚠️ {label_short}: folder not found: {folder_path}")
        continue
    imgs = sorted(glob.glob(os.path.join(folder_path, '*.jpg')) +
                  glob.glob(os.path.join(folder_path, '*.png')) +
                  glob.glob(os.path.join(folder_path, '*.jpeg')))
    ref_images[label_short] = [
        resize_for_phi(Image.open(p).convert("RGB"))
        for p in imgs[:1]                     # ★ 2 → 1
    ]
    sizes = [im.size for im in ref_images[label_short]]
    print(f"   {label_short}: {len(ref_images[label_short])} refs from {folder_name}/  sizes={sizes}")


# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────
def load_phi_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="cuda:0",
        trust_remote_code=True,
        _attn_implementation='eager',
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, trust_remote_code=True, num_crops=NUM_CROPS)   # ★ 2
    return model, processor


def deep_cleanup():
    for _ in range(5):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(0.3)
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def reload_model(max_retries=3):
    global vl_model, vl_processor
    
    print(f"      🔄 Heavy cleanup: reloading model...")
    t_reload = time.time()
    
    try:
        del vl_model
    except Exception:
        pass
    try:
        del vl_processor
    except Exception:
        pass
    vl_model = None
    vl_processor = None
    
    deep_cleanup()
    print(f"      VRAM after cleanup: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            vl_model, vl_processor = load_phi_model()
            print(f"      ✅ Reloaded in {time.time()-t_reload:.1f}s (attempt {attempt}), "
                  f"VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            return True
        except Exception as e:
            last_err = e
            print(f"      ⚠️ Reload attempt {attempt}/{max_retries} failed: {str(e)[:80]}")
            try:
                del vl_model
            except Exception:
                pass
            try:
                del vl_processor
            except Exception:
                pass
            vl_model = None
            vl_processor = None
            deep_cleanup()
            time.sleep(2)
    
    print(f"      ❌ All {max_retries} reload attempts failed")
    return False


def ensure_model_loaded():
    global vl_model, vl_processor
    if vl_model is None or vl_processor is None:
        print(f"      ⚙️ Model is not loaded, attempting to load...")
        return reload_model()
    return True


# ─────────────────────────────────────────────
# LOAD PHI-3.5-VISION (4-bit) — ilk yükleme
# ─────────────────────────────────────────────
print(f"\n🧠 Loading Phi-3.5-Vision-Instruct (4-bit, num_crops={NUM_CROPS})...")
vl_model, vl_processor = load_phi_model()
print(f"   ✅ Loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

HIDDEN_SIZE = vl_model.config.hidden_size if hasattr(vl_model.config, 'hidden_size') else 3072
print(f"   Hidden size: {HIDDEN_SIZE}")

# ─────────────────────────────────────────────
# LABEL DEFINITIONS
# ─────────────────────────────────────────────
LABEL_DEFS = {
    "DKS": (
        "Diffuse capillary leakage — widespread foggy brightness BETWEEN vessels across a large area. "
        "NORMAL ANATOMY NOTE: Some background brightness is normal. PATHOLOGICAL only if the brightness "
        "is widespread (covers >1/3 of image), has fuzzy borders, and reduces contrast significantly. "
        "Compare peripheral background to areas near disc — if both are hazy, DKS is likely."
    ),
    "ODB": (
        "Optic disc staining — the disc is ABNORMALLY bright with blurred/washed-out edges. "
        "NORMAL ANATOMY NOTE: The optic disc is ALWAYS somewhat bright in FA — this is NORMAL. "
        "A normal disc has clear sharp margins and moderate brightness. PATHOLOGICAL only if the disc "
        "is dramatically brighter than surrounding retina AND its edges are obscured by the brightness. "
        "Do NOT call normal disc brightness as ODB."
    ),
    "VI": (
        "Vitreous inflammation — the ENTIRE image is hazy/foggy with globally reduced clarity. "
        "NORMAL ANATOMY NOTE: Some FA images have slightly lower contrast due to technique. "
        "PATHOLOGICAL only if vessel edges and disc borders that should be sharp appear blurry, "
        "AND the haziness is uniform across the whole image, not just one region."
    ),
    "MÖ": (
        "Macular edema — flower-petal or star-shaped brightness at the image CENTER (macula). "
        "NORMAL ANATOMY NOTE: The macula center (fovea) is normally DARK due to avascular zone. "
        "PATHOLOGICAL only if you see abnormal brightness specifically at the macula — petal pattern, "
        "central glow, or bright cysts clustered at the center."
    ),
    "DDB": (
        "Vessel wall staining — bright glow along vessel WALLS (outside the lumen). "
        "NORMAL ANATOMY NOTE: Vessels are normally bright INSIDE because of blood flow. "
        "PATHOLOGICAL only if you see brightness OUTSIDE/ALONG the vessel walls, making vessels "
        "appear outlined, double-lined, or thicker than normal with a halo effect."
    ),
    "RI": (
        "Retinal infiltrate — a dark patch with irregular/fuzzy edges surrounded by a bright halo. "
        "NORMAL ANATOMY NOTE: The fovea is normally dark — do NOT confuse with infiltrate. "
        "PATHOLOGICAL only if a dark area has IRREGULAR borders AND a surrounding bright ring/halo. "
        "Usually found near vessels, not at the foveal center."
    ),
    "HEM": (
        "Hemorrhage — distinctly BLACK areas that block the underlying pattern completely. "
        "NORMAL ANATOMY NOTE: The fovea and some peripheral areas can be dark normally. "
        "PATHOLOGICAL only if dark areas have IRREGULAR borders, are flame-shaped or blot-shaped, "
        "and appear OUT OF PLACE — interrupting the normal vessel/background pattern."
    ),
    "PVK": (
        "Perivascular sheathing — white opaque coating/sleeve around vessel segments. "
        "NORMAL ANATOMY NOTE: Vessels have some natural wall visibility. "
        "PATHOLOGICAL only if you see a distinct white COATING wrapping around a vessel segment, "
        "creating a railroad-track appearance with opaque white on both sides."
    ),
    "RSLD": (
        "Nerve fiber layer defect — wedge/arc-shaped dark gap near the optic disc. "
        "NORMAL ANATOMY NOTE: Area around disc can have variable brightness. "
        "PATHOLOGICAL only if one sector around the disc is distinctly darker than adjacent sectors, "
        "forming a wedge or arc shape pointing away from the disc."
    ),
    "GV": (
        "Ghost vessel — vessel traces visible but EMPTY (no dye filling). "
        "NORMAL ANATOMY NOTE: All normal vessels should be bright (filled with dye). "
        "PATHOLOGICAL only if you see vessel outlines that are faint/transparent with no bright "
        "dye inside — like abandoned roads. Nearby vessels may be abnormally dilated."
    ),
}

COOCCURRENCE_INFO = """
CLINICAL CO-OCCURRENCE PATTERNS (use to validate your observations):
Group A (tend to appear together): DKS ↔ ODB ↔ MÖ ↔ DDB
  - When ODB present → MÖ also present 58% of the time
  - When ODB present → DDB also present 34% of the time
  - When DKS present → ODB also present 14% of the time
Group B (tend to appear together): VI ↔ RI ↔ HEM
  - When RI present → HEM also present 31% of the time (54x more likely)
  - When HEM present → RI also present 57% of the time (86x more likely)
  - When VI present → RI 4x more likely, HEM 6x more likely
Groups A and B RARELY overlap.
"""

GT_SYSTEM = (
    "You are a medical vision-language assistant specialized in retinal fluorescein "
    "angiography interpretation. The diagnoses are already confirmed by an expert. "
    "Your job is to explain the visual evidence that supports present findings and "
    "the missing visual evidence for absent findings. Be thorough and specific."
)

# ─────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────
def make_gt_prompt(gt_dict, max_neg=7):
    pos = [sn for sn, v in gt_dict.items() if v == 1]
    neg = [sn for sn, v in gt_dict.items() if v == 0][:max_neg]
    pos_text = "\n".join([f"  - {sn}: {LABEL_DEFS.get(sn, sn)}" for sn in pos]) if pos else "  - None (normal image)"
    neg_text = "\n".join([f"  - {sn}: {LABEL_DEFS.get(sn, sn)}" for sn in neg]) if neg else "  - None"
    return f"""You are analyzing a retinal fluorescein angiography (FA) image.
The diagnosis is ALREADY CONFIRMED by an expert ophthalmologist.
Your task is NOT to diagnose — it is to explain the visible evidence.

CRITICAL RULE — NORMAL vs PATHOLOGICAL:
Many structures in the eye are NORMALLY bright or dark in FA images:
- The OPTIC DISC is naturally bright — this alone is NOT pathological
- The FOVEA (macula center) is naturally dark — this alone is NOT hemorrhage
- Vessels are naturally bright inside — this alone is NOT vessel wall staining
- Some background brightness variation is normal — this alone is NOT leakage
ONLY mark something as pathological if it is ABNORMAL compared to what
a healthy eye would show. Each label definition includes what is normal
vs what is pathological — follow those guidelines strictly.

Image characteristics:
- Grayscale or green-tinted medical image
- Bright areas = dye leaking or accumulating
- Dark areas = something blocking the dye
- Blood vessels = bright branching lines
- Optic disc = round bright structure where vessels converge (NORMALLY bright)
- Macula = center area (NORMALLY darker)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIRMED PRESENT findings:
{pos_text}

CONFIRMED ABSENT findings:
{neg_text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{COOCCURRENCE_INFO}

INSTRUCTIONS:
1. For EACH confirmed PRESENT finding:
   - Describe the specific ABNORMAL visual evidence (not normal anatomy)
   - Mention WHERE in the image
   - Explain why this is PATHOLOGICAL and not normal anatomy

2. For EACH confirmed ABSENT finding:
   - State what ABNORMAL feature would be visible if it were present
   - Explain that you see only NORMAL anatomy in that region, not pathology

3. For each finding, explicitly state: "This is [NORMAL ANATOMY / PATHOLOGICAL] because..."

Be detailed and clinically focused. Use ONLY visible image evidence.
Distinguish normal anatomical appearances from pathological findings."""


# ─────────────────────────────────────────────
# CORE EMBEDDING FUNCTION (Phi-3.5-Vision)
# ─────────────────────────────────────────────
def extract_teacher_embedding(img_path, gt_dict, max_new_tokens=600, image_max_size=MAX_SIZE):
    pos_labels = [sn for sn, v in gt_dict.items() if v == 1]
    patient_pil = Image.open(img_path).convert("RGB")
    patient_pil = resize_for_phi(patient_pil, image_max_size)

    all_images = []
    image_descriptions = []
    img_counter = 1
    
    refs_added = set()
    for sn in pos_labels:
        if sn in ref_images and sn not in refs_added:
            for ref_img in ref_images[sn]:
                all_images.append(ref_img)
                image_descriptions.append(
                    f"Image {img_counter}: REFERENCE for {sn} ({LABEL_DEFS[sn]})"
                )
                img_counter += 1
            refs_added.add(sn)
    
    all_images.append(patient_pil)
    patient_idx = img_counter
    image_descriptions.append(
        f"Image {img_counter}: PATIENT IMAGE — this is the one to analyze"
    )
    
    placeholder = "".join(f"<|image_{i}|>\n" for i in range(1, len(all_images) + 1))
    image_legend = "\n".join(image_descriptions)
    gt_text = make_gt_prompt(gt_dict)
    
    user_content = (
        placeholder
        + "\n"
        + "Below is a legend explaining each image:\n"
        + image_legend
        + "\n\n"
        + GT_SYSTEM
        + "\n\n"
        + f"Now analyze IMAGE {patient_idx} (the patient image) using the references and the instructions below:\n\n"
        + gt_text
    )

    messages = [{"role": "user", "content": user_content}]
    prompt = vl_processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    inputs = vl_processor(prompt, all_images, return_tensors="pt").to(vl_model.device)
    input_len = inputs['input_ids'].shape[1]

    eos_ids = [vl_processor.tokenizer.eos_token_id]
    end_id = vl_processor.tokenizer.convert_tokens_to_ids("<|end|>")
    if end_id and end_id != vl_processor.tokenizer.unk_token_id:
        eos_ids.append(end_id)
    eos_ids = list(set(eos_ids))

    captured = []
    def _hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured.append(h[:, -1, :].detach().float().cpu())

    last_layer = vl_model.model.layers[-1]
    handle = last_layer.register_forward_hook(_hook)

    torch.cuda.empty_cache()
    gc.collect()

    explanation = ""
    try:
        with torch.inference_mode():
            gen_output = vl_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=vl_processor.tokenizer.pad_token_id or vl_processor.tokenizer.eos_token_id,
                eos_token_id=eos_ids,
                use_cache=True,
                return_dict_in_generate=True,
            )
        generated_ids = gen_output.sequences[:, input_len:].cpu()
        explanation = vl_processor.batch_decode(
            generated_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0].strip()
        del gen_output, generated_ids
    finally:
        handle.remove()

    del inputs, all_images, prompt
    torch.cuda.empty_cache()
    gc.collect()

    if len(captured) == 0:
        return np.zeros(HIDDEN_SIZE, dtype=np.float32), explanation

    gen_hiddens = captured[1:] if len(captured) > 1 else captured
    n_use = min(30, len(gen_hiddens))
    embs = torch.stack(gen_hiddens[-n_use:]).squeeze(1)
    emb = embs.mean(dim=0).numpy()
    emb = emb / (np.linalg.norm(emb) + 1e-8)

    captured.clear()
    del embs, gen_hiddens
    torch.cuda.empty_cache()

    return emb.astype(np.float32), explanation


# ═══════════════════════════════════════════════════════════════════
# CELL 9: BATCH EMBEDDING EXTRACTION — 561 PAT → .npy
# ═══════════════════════════════════════════════════════════════════
SAVE_DIR = os.path.join(DATA_ROOT, 'teacher_embeddings_phi')
os.makedirs(SAVE_DIR, exist_ok=True)

EMB_PATH  = os.path.join(SAVE_DIR, 'teacher_embeddings.npy')
META_PATH = os.path.join(SAVE_DIR, 'teacher_metadata.csv')
EXPL_DIR  = os.path.join(SAVE_DIR, 'explanations')
os.makedirs(EXPL_DIR, exist_ok=True)

df_all = df_pat.reset_index(drop=True)
print(f"📂 Total images to process: {len(df_all)} (pathological only)")
print(f"📂 Save dir: {SAVE_DIR}")
print(f"📸 Reference images loaded: {list(ref_images.keys())}")
print(f"🧹 Light cleanup every {LIGHT_CLEAN_EVERY} imgs, heavy reload every {HEAVY_CLEAN_EVERY} imgs")
print(f"🖼️ Image max size: {MAX_SIZE} px, num_crops: {NUM_CROPS}")

if os.path.exists(META_PATH):
    done_df    = pd.read_csv(META_PATH)
    done_paths = set(done_df['image_path'].values)
    done_embs  = list(np.load(EMB_PATH)) if os.path.exists(EMB_PATH) else []
    print(f"   Resuming: {len(done_paths)} already done")
else:
    done_df    = pd.DataFrame()
    done_paths = set()
    done_embs  = []
    print("   Starting fresh")

remaining = df_all[~df_all['image_path'].isin(done_paths)].reset_index(drop=True)
print(f"   Remaining: {len(remaining)} images")
print(f"   Estimated time: ~{len(remaining) * 12 / 60:.0f} min\n")

# Quick sanity test
if len(remaining) > 0:
    print("🧪 Testing extraction on first image...")
    test_row = remaining.iloc[0]
    test_gt = {sn: int(test_row[col]) if col in test_row.index else 0
               for col, sn in zip(ALL_PAT_COLS, ALL_PAT_SHORT)}
    try:
        test_emb, test_expl = extract_teacher_embedding(test_row['image_path'], test_gt)
        if np.linalg.norm(test_emb) > 1e-6 and len(test_expl) > 50:
            print(f"   ✅ Test passed! emb_norm={np.linalg.norm(test_emb):.4f}, "
                  f"expl_len={len(test_expl)}, dim={test_emb.shape[0]}")
            print(f"   Sample expl: {test_expl[:200]}\n")
        else:
            print(f"   ⚠️ Test weak: emb_norm={np.linalg.norm(test_emb):.4f}, expl_len={len(test_expl)}")
            print(f"   Continuing anyway...\n")
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        raise

new_metas, new_embs = [], []
errors = 0
oom_skipped = 0          # ★ OOM yüzünden atlanan sayısı
processed_count = 0
t0 = time.time()

for i, row in remaining.iterrows():
    img_path = row['image_path']
    img_name = os.path.basename(img_path)
    patient  = row['patient_id']

    gt_dict = {sn: int(row[col]) if col in row.index else 0
               for col, sn in zip(ALL_PAT_COLS, ALL_PAT_SHORT)}
    gt_str  = "+".join([s for s in ALL_PAT_SHORT if gt_dict[s] == 1])
    n_refs  = sum(1 for sn in ALL_PAT_SHORT if gt_dict[sn] == 1 and sn in ref_images)

    # Her iterasyon başında model var mı kontrol et
    if not ensure_model_loaded():
        errors += 1
        print(f"  [{len(done_paths)+len(new_embs):4d}/{len(df_all)}] "
              f"{img_name:30s} ⚠️ Model unavailable, skipping")
        continue

    success = False
    try:
        emb, expl = extract_teacher_embedding(img_path, gt_dict)

        new_embs.append(emb)
        meta = {
            'image_path': img_path, 'image_name': img_name,
            'patient_id': patient,  'gt_labels': gt_str,
            'expl_len': len(expl),  'n_ref_labels': n_refs,
        }
        for sn in ALL_PAT_SHORT:
            meta[f'gt_{sn}'] = gt_dict[sn]
        new_metas.append(meta)

        expl_file = os.path.join(EXPL_DIR, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        with open(expl_file, 'w', encoding='utf-8') as f:
            f.write(expl)

        total_done = len(done_paths) + len(new_embs)
        elapsed = (time.time() - t0) / 60
        eta = elapsed / max(len(new_embs), 1) * (len(remaining) - len(new_embs))
        print(f"  [{total_done:4d}/{len(df_all)}] {img_name:30s} [{gt_str:15s}] "
              f"refs={n_refs} expl={len(expl):5d} ({elapsed:.1f}m, ETA {eta:.1f}m)")
        success = True
        processed_count += 1

    except torch.cuda.OutOfMemoryError:
        # ★ OOM → karmaşık retry yok, sadece deep cleanup + skip
        errors += 1
        oom_skipped += 1
        print(f"  [{len(done_paths)+len(new_embs):4d}/{len(df_all)}] "
              f"{img_name:30s} ⚠️ OOM — skipping (total skipped: {oom_skipped})")
        deep_cleanup()
        # Eğer model bozulduysa yeniden yükle (sıradaki iter'in başlamasını sağla)
        if vl_model is None or vl_processor is None:
            ensure_model_loaded()

    except Exception as e:
        errors += 1
        print(f"  [{len(done_paths)+len(new_embs):4d}/{len(df_all)}] "
              f"{img_name:30s} ⚠️ {str(e)[:80]}")

    # ★ HAFİF TEMİZLİK
    if success and processed_count > 0 and processed_count % LIGHT_CLEAN_EVERY == 0:
        torch.cuda.empty_cache()
        gc.collect()

    # ★ AĞIR TEMİZLİK
    if success and processed_count > 0 and processed_count % HEAVY_CLEAN_EVERY == 0:
        reload_model()

    # Checkpoint save
    if len(new_embs) % 20 == 0 and len(new_embs) > 0:
        all_embs_save = done_embs + new_embs
        np.save(EMB_PATH, np.array(all_embs_save))
        new_meta_df = pd.DataFrame(new_metas)
        all_meta = pd.concat([done_df, new_meta_df], ignore_index=True)
        all_meta.to_csv(META_PATH, index=False, encoding='utf-8-sig')
        print(f"    💾 Saved checkpoint: {len(all_embs_save)} embeddings")

# Final save
if new_embs:
    all_embs_final = done_embs + new_embs
    np.save(EMB_PATH, np.array(all_embs_final))
    new_meta_df = pd.DataFrame(new_metas)
    all_meta = pd.concat([done_df, new_meta_df], ignore_index=True)
    all_meta.to_csv(META_PATH, index=False, encoding='utf-8-sig')

total   = len(done_paths) + len(new_embs)
elapsed = (time.time() - t0) / 60

print(f"\n{'='*60}")
print(f"✅ EMBEDDING EXTRACTION COMPLETE (PHI-3.5-VISION)")
print(f"   Total: {total}/{len(df_all)} images")
print(f"   Errors: {errors} (OOM skipped: {oom_skipped})")
print(f"   Time: {elapsed:.1f} min")
print(f"   Embeddings: {EMB_PATH}")
print(f"   Metadata:   {META_PATH}")

if os.path.exists(EMB_PATH) and os.path.exists(META_PATH):
    embs = np.load(EMB_PATH)
    meta = pd.read_csv(META_PATH)
    print(f"\n   Verify: {embs.shape[0]} embeddings × {embs.shape[1]} dim")
    print(f"   Label distribution:")
    for sn in ALL_PAT_SHORT:
        n = int(meta[f'gt_{sn}'].sum())
        if n > 0:
            print(f"     {sn:6s}: {n:4d}")
    if 'n_ref_labels' in meta.columns:
        print(f"\n   Ref usage:")
        print(f"     With refs:    {int((meta['n_ref_labels'] > 0).sum())}")
        print(f"     Without refs: {int((meta['n_ref_labels'] == 0).sum())}")
print(f"{'='*60}")

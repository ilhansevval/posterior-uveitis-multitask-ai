# Behçet — Vision-Language Knowledge Distillation on Fundus/Pathology Images

This repository contains the full code for a deep-learning pipeline targeting Behçet's
disease on fundus and pathology images. The core idea is to extract **embeddings** from
large vision-language and foundation teacher models (Qwen2.5-VL, MedGemma, Phi-3.5-Vision,
RETFound, BiomedCLIP) and transfer that knowledge to smaller student models via **knowledge
distillation**. The repo also covers pseudo-labeling, label-efficiency experiments,
robustness ablations, classical baselines, and external validation on the RFMiD dataset.

The code is organized into thematically grouped folders. Each folder holds a
standalone Python script (or a few sequential scripts).

---

## Setup and secrets

Some scripts download gated models from Hugging Face and need an access token for that.
**The token is never hardcoded.** Provide it as an environment variable before running:

```bash
# Linux / macOS
export HF_TOKEN=hf_YOUR_NEW_TOKEN

# Windows (PowerShell)
$env:HF_TOKEN = "hf_YOUR_NEW_TOKEN"

# Windows (persistent, open a new terminal afterward)
setx HF_TOKEN hf_YOUR_NEW_TOKEN
```

Scripts that require a token raise a clear error if the environment variable is not set.

> Note: Data paths in the scripts are absolute (`C:\Users\...`, `/Users/...`).
> You need to update them to match your own data folder.

---

## Folder structure

### `fundus_screening/`
- **fundus_binary** — Binary screening/classification model for fundus images.

### `pathology_distillation/`
Pipeline for extracting teacher embeddings from pathology images and distilling them into a student model.
- **pathology_training** — Base training of the pathology classification model.
- **pathology_embedding_extraction** — GT-guided teacher embedding extraction (including pooling-strategy comparison and batch extraction).
- **pathology_kd_training** — Student model distillation training using the teacher embeddings.
- **pathology_ref_training** — Reference-image training variant.
- **pathology_kd_2_reference_images_embedding_extraction_medgemma** — Embedding extraction with 2 reference images (MedGemma teacher).
- **pathology_kd_2_reference_images_embedding_extraction_qwen_25** — Same setup with the Qwen2.5-VL teacher.

### `teacher_models/`
Embedding extraction and analysis for the different teacher/foundation models.
- **qwen25_ref_2_embedding_analysis** — Qwen2.5-VL (2 references) embedding analysis.
- **qwen_no_ref_embedding_analysis** — Qwen (no reference) embedding analysis.
- **medgemma_2_ref_embedding_extraction / _analysis** — MedGemma 2-reference embedding extraction and analysis.
- **phi_35_vision_ref_embedding_extraction / _analysis** — Phi-3.5-Vision embedding extraction and analysis.
- **retfound_embedding_extraction / _analysis** — RETFound (fundus foundation model) embedding extraction and analysis.
- **biomedclip_embedding_extraction / _analysis** — BiomedCLIP embedding extraction and analysis.
- **umap_embedding_visualization** — 2D UMAP visualization of the extracted embeddings.

### `pseudo_labeling/`
Pseudo-label generation to leverage unlabeled data, plus training on those labels.
- **pseudo_label_extraction** — Pseudo-label generation for unlabeled images (agreement-filtered).
- **pseudo_label_training** — Student training with pseudo-labels.
- **pseudo_wrong_label_extraction** — Intentionally noisy/wrong pseudo-label generation (ablation).
- **pseudo_wrong_label_training** — Training with wrong pseudo-labels (ablation).

### `label_efficiency/`
Training with different fractions of ground-truth labels — label-efficiency experiments.
- **gt_10_percentage_training** — Training with 10% of the labels.
- **gt_25_percentage_training** — Training with 25%.
- **gt_50_percentage_training** — Training with 50%.
- **gt_75_percentage_training** — Training with 75%.

### `robustness_ablations/`
Robustness ablations against wrong/noisy signals.
- **wrong_prompt_extraction** — Embedding extraction with a wrong prompt.
- **wrong_anatomy_training** — Training with a wrong-anatomy reference.
- **irrelevant_training** — Training with irrelevant references/data.

### `baselines/`
Models trained from scratch for comparison, plus student-backbone ablations.
- **ml_baseline** — Classical machine-learning baseline.
- **resnet_50_baseline**, **densenet121_baseline**, **efficient_net_b4_vanilla_baseline**, **vit_baseline** — CNN and ViT baselines.
- **retfound_baseline**, **biomedclip_baseline** — Foundation-model-based baselines.
- **student_backbone_ablation_densenet_121 / _resnet50** — Ablations where the student backbone is swapped.

### `rfmid_validation/`
Cross-validation on the external RFMiD dataset.
- **rfmid_data_analysis** — Exploratory analysis of the RFMiD dataset.
- **rfmid_cross_modality_validation** — Cross-modality validation.
- **rfmid_label_sayim_analizi** — RFMiD label distribution/count analysis.
- **rfmid_normal_baseline** — Baseline training on RFMiD.
- **rfmid_teacher_embedding_extraction** — Teacher embedding extraction for RFMiD.
- **rfmid_kd_training** — Distillation training on RFMiD.

### `teacher_comparison/`
- **teacher_comparison_via_distillation** — Comparison of the different teacher models by distillation performance (two scripts: `_1`, `_2`).

---

## Overall flow (suggested reading order)

1. `fundus_screening` → binary screening.
2. `pathology_distillation` → extract teacher embeddings, distill to the student.
3. `teacher_models` → extract and analyze embeddings from different teachers.
4. `pseudo_labeling` → expand with unlabeled data via pseudo-labels.
5. `label_efficiency` + `robustness_ablations` → data/noise experiments.
6. `baselines` → comparison models.
7. `rfmid_validation` → external validation.
8. `teacher_comparison` → final comparison of teachers.

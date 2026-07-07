# BCDR_BioCLIP_Diagnosis_and_Recovery
Project diagnosing BioCLIP-family failure on insect camera trap imagery from alpine Switzerland. Parameter-efficient adaptation (soft-prompts + LoRA) recovers performance on the most-failing orders (Hemiptera, Coleoptera). Compared against DINOv2 and CNN baselines (ResNet-50, EfficientNet-B3).

# BioCLIP-DiopSIS: Diagnosing and Recovering BioCLIP Classification Performance on DiopSIS Camera Trap Data

 Project systematically diagnosing BioCLIP-family foundation model failure on real-world insect camera trap imagery and evaluating a structured recovery pathway through parameter-efficient adaptation.

## Overview

Vision-language foundation models such as BioCLIP promise automated taxonomic classification without task-specific training, but field deployment on camera trap imagery reveals substantial performance asymmetries across taxonomic groups. This project:

1. **Diagnoses** BioCLIP v1 and BioCLIP 2 failure mechanisms on DiopSIS insect camera trap data from an alpine Swiss deployment (39,788 re-annotated crops spanning 14 insect orders)
2. **Recovers** performance through parameter-efficient adaptation: soft-prompt tuning, LoRA fine-tuning of the visual encoder, and their combination
3. **Compares** the adapted BioCLIP 2 pipeline against DINOv2 (frozen and LoRA-adapted) and supervised CNN baselines (ResNet-50, EfficientNet-B3)

## Key findings

- **Failure is concentrated at the text-based classification interface**, not the visual encoder — evidenced by k-NN over frozen visual features substantially outperforming native zero-shot prediction
- **Coleoptera and Hemiptera fail** in both BioCLIP versions (F1 < 0.21), with high recall but catastrophic precision — models over-predict these orders for images of other classes
- **Soft-prompt tuning + LoRA recovers diagnostic classes**: Hemiptera F1 from 0.205 to 0.883, Coleoptera from 0.413 to 0.906
- **Rigidity-strength trade-off in DINOv2**: higher frozen-feature performance but resists LoRA adaptation
- **CNN baselines complement adapted BioCLIP 2** on micro-classes excluded from LoRA training

## Repository structure
```

├── notebooks/                                 # Jupyter notebooks for analysis
│   ├── sample_overview.ipynb                  # Detailed overview of BioCLIP V1 and V2 prediction without and  with pre-pocessing
│   ├── Visual_embeddings_(...).ipynb          # BioCLIP v1/v2 vis. embeddings k-NN
│   ├── 03_image_quality_regression/           # Logistic regression models
│   ├── 07_BioClip2_restricted.ipynb           # BioCLIP 2 restricted label space (14)
│   ├── 07_BioClip_restricted.ipynb            # BioCLIP restricted label space (14)
│   ├── 06_soft_prompt_tuning/                 # Soft-prompt training
│   ├── 01_sweep.ipynb                         # LoRA hp sweep on BioCLIP 2
│   ├── 02_cv_lora.ipynb                       # LoRA training on BioCLIP 2
│   ├── 003_cv_joint.ipynb                     # Combined LoRA + soft-prompt
│   ├── 04_dinov2_linear_probe.ipynb           # DINOv2 frozen
│   ├── 09_cv_dinov2_lora.ipynb                # DINOv2 frozen + LoRA 
│   ├── 09_cv_dinov2_lora-without_qkv.ipynb    # DINOv2 frozen + LoRA without qk
│   ├── resnet_50_diopsis.ipynb                # ResNet-50
│   └── efficiientnetb3.ipynb                  # EfficientNet-B3
│
├── results_csv/                          # Csv files containg experimental outputs
│   └── Predictions/                      # Zero-shot outputs
│   │    ├──bioclip_v1_predictions.csv    # BioCLIP V1
│   │    └── bioclip_v2_predictions.csv   # BioCLIP V2
│   └── detailed_sample_description/      # Count, metrices, recall for BioCLIP V1 and V2
│   │    ├── counts_b(...).csv            # Confusion matrix counts BioCLIP V1 and V2
│   │    ├── metrics_b(...).csv           # Classification performance results for BioCLIP V1 and V2
│   │    ├── recall_pct_b(...).csv        # Normalized recall for BioCLIP V1 and V2 
│   │    └── rescue_damage(...).csv       # Rescue/damage rate for pre-processed crops classification on BioCLIP V1
│   └── diagnostic_logistic_regression/   # Logistic regression - failure prediction of class with BioCLIP V1
│   │    ├── log1.csv                     # control
│   │    ├── log2_1.csv                   # crops and settings diagnostic
│   │    └── log3_1.csv                   # sensitivity analysis
│   └── output_knn/                       # BioCLIP V1 and V2 visual embeddings knn analysis results
│   ├── output_bioClip2_restricted.../    # BioCLIP V2 restricted label space (37) results
│   ├── cv_soft_prompt/                   # BioCLIP V2 soft-prompt CV results including restricted label space (14) baseline
│   ├── lora_hpsweep_output/              # BioCLIP V2 LoRA hp sweep output
│   ├── lora_cv_final/                    # BioCLIP V2 LoRA CV results
│   ├── output_cv_joint/                  # BioCLIP V2 LoRA CV & soft-prompt CV results
│   ├── output_dinov2/                    # DINOv2 baseline CV results
│   ├── output_dinov2_lora_without.../    # DINOv2 LoRA tuning without qkv
│   ├── output_dinov2_lora/               # DINOv2 LoRA tuning
│   ├── output_resnet50/                  # ResNet-50 results
│   └── output_efficientnetb3/            # EfficientNet-B3s results
│   
├── figures/                              # Generated figures for thesis
├── configs/                          # Hyperparameter configurations
└── docs/                             # Additional documentation
```
## Setup

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU for LoRA/soft-prompt training (evaluated on NVIDIA A100 80GB)
- Approximately 40 GB storage for embeddings and features

### Installation

```bash
# Clone repository
git clone https://github.com/[username]/bioclip-diopsis.git
cd bioclip-diopsis

# Create conda environment
conda create -n bioclip-diopsis python=3.10
conda activate bioclip-diopsis

# Install dependencies
pip install -r requirements.txt
```

### Key dependencies

- `open_clip_torch` — BioCLIP v1/v2 model access
- `peft` — LoRA implementation
- `torch`, `torchvision` — PyTorch training
- `scikit-learn` — logistic regression, k-NN, metrics
- `statsmodels` — inferential statistics for Section 4.3
- `opencv-python` — preprocessing techniques
- `pandas`, `numpy` — data handling

Full dependency list in `requirements.txt`.

## Data availability

The DiopSIS image data used in this project is subject to the Appolo project data sharing policy and is **not publicly available**. Access can be requested through the WSL.

For reproducibility, this repository includes:
- **Per-fold outputs**: classification reports and confusion matrices for all experiments

These artifacts enable full re-analysis without access to the raw image data.

## Reproducing results

### Diagnostic analysis (Section 4)

```bash
# BioCLIP v1/v2 zero-shot + k-NN diagnostic
jupyter notebook notebooks/02_bioclip_diagnosis/cv_bioclip_v1_v2.ipynb

# Image quality logistic regression
jupyter notebook notebooks/03_image_quality_regression/model_1_2_3.ipynb
```

### Interventions (Section 5)

```bash
# Soft-prompt tuning (BioCLIP 2, 4 tokens per class)
jupyter notebook notebooks/05_soft_prompt_tuning/cv_soft_prompt.ipynb

# LoRA fine-tuning (BioCLIP 2, rank 16, lr 5e-4)
jupyter notebook notebooks/06_lora_fine_tuning/cv_lora.ipynb

# Joint LoRA + soft-prompt
jupyter notebook notebooks/07_joint_lora_soft_prompt/cv_lora_soft.ipynb

# DINOv2 baselines
jupyter notebook notebooks/08_dinov2_baseline/cv_dinov2_lora.ipynb

# CNN baselines
jupyter notebook notebooks/09_cnn_baselines/cv_resnet50.ipynb
jupyter notebook notebooks/09_cnn_baselines/cv_efficientnet_b3.ipynb
```

### Cross-validation protocol

All CV experiments use 3-fold stratified splits with `random_state=0`. Classes with n<3 are retained in the training set across all folds and receive F1=0 in evaluation by construction. Classes with n<50 are excluded from LoRA and soft-prompt training but retained in validation for evaluation.

## Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{MarlenaDzikowska,
  title  = {Diagnosing and Recovering BioCLIP Classification Performance on DiopSIS Camera Trap Data},
  author = {Marlena Monika Dzikowska},
  year   = {2026},
  school = {University of Leeds},
  type   = {MSc Thesis}
}
```

## Related work

This project builds on and contributes to several lines of research:

- **BioCLIP** [1] introduced vision-language pretraining for biology; **BioCLIP 2** [2] expanded pretraining and architecture
- **Sajun et al. (2025)** [3] documented BioCLIP zero-shot failure on wildlife camera traps
- **Gardiner et al. (2025)** [4] compressed BioCLIP 2 for moth AMI monitoring via knowledge distillation
- **DINOv2** [5] provides general self-supervised visual features as an alternative baseline

Full reference list in the thesis document.

## Acknowledgements

This work was conducted as part of the project (Swiss Data Science Center, WSL) aimed at developing scalable pollinator monitoring pipelines. DiopSIS camera trap data was collected in the Davos region as part of the broader Apollo pipeline. The author thanks the entomologist consultant for re-annotation verification support.

## License

Code in this repository is released under the [MIT License](LICENSE). See `LICENSE` for details.

The DiopSIS image data and re-annotation labels are subject to separate data sharing agreements and are not covered by this license.

## Contact

For questions about the code or reproducing results:
- [Your Name]: [marlena.dzikowska(at)gmail.com]
- Issues: [https://github.com/marlena-dzikowska/BCDR_BioCLIP_Diagnosis_and_Recovery]

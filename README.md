# Antibody Design Pipeline: LoRA-Enhanced Generative AI

This repository implements an advanced, end-to-end deep learning pipeline for *de novo* therapeutic antibody design. Unlike standard wrapping scripts, our approach pioneers the use of **domain-specific Low-Rank Adaptation (LoRA)** on the RFdiffusion backbone, significantly enhancing its capability to model complex antibody-antigen interfaces using large-scale datasets (e.g., BigBind). By seamlessly coupling this structurally optimized backbone with **ProteinMPNN** for inverse folding, we provide a robust, high-fidelity framework for generating target-specific binders.

## ✨ Key Innovations

* **Domain-Specific LoRA Adaptation**: We directly integrate Low-Rank Adaptation (LoRA) into the RFdiffusion structural prior. By training specifically on antibody-antigen complexes, we overcome the generic limitations of foundational diffusion models.
* **Zero-Overhead Inference via Weight Merging**: Our custom `merge_lora.py` architecture permanently fuses the trained LoRA adapters into the base network. This achieves domain-expert generative performance with *zero* computational overhead or latency during the inference phase.
* **Robust End-to-End Orchestration**: Features a fully automated handoff from fixed-target backbone diffusion (`test.py`) to sequence optimization (`run_mpnn.py`), equipped with resilient PDB parsing and automatic CDR region detection.

## 📋 Pipeline Overview

The pipeline consists of three main phases:
1.  **Fine-tuning (LoRA):** Train the RFdiffusion model using LoRA (Low-Rank Adaptation) on the BigBind dataset to enhance antibody-antigen interface binding features.
2.  **Backbone Generation:** Use the merged RFdiffusion model to generate 3D candidate antibody backbones targeting specific epitopes.
3.  **Sequence Design (ProteinMPNN):** Run the ProteinMPNN model using the generated backbones for antibody chain sequence optimization while locking the target antigen.

## 📂 Project Structure

```text
.
├── training/                     # Phase 1, 2 & 3: Training and Inference Scripts
│   ├── train_lora.py             # LoRA fine-tuning script
│   ├── merge_lora.py             # Weight merging utility
│   └── dataset.py                # Antibody-complex dataset loader
├── evaluation/                   # Phase 1, 2 & 3: Training and Inference Scripts
│   ├── test.py                   # Backbone generation script (RFdiffusion)
│   └── run_mpnn.py               # Sequence design script (ProteinMPNN)
├── generation_results/           # Results generated with fine-tunned weight
├── mpnn_results/                 # Results generated with ProteinMPNN
├── data/                         # Data storage
│   ├── example/                  # Data used for test
│   └── raw/                      # Some data from processed BigBind dataset
├── models/                       # Model checkpoints
│   └── download_weights.sh       # Download the trained LoRA weights
├── RFdiffusion/                  # Phase 2: Structural Diffusion submodule
├── ProteinMPNN/                  # Phase 3: Sequence Design submodule
├── requirements.txt              # Python dependencies
└── README.md

```
## 🧬 Pipeline Architecture & Innovation Workflow

This diagram illustrates how domain-specific optimization (LoRA) is seamlessly integrated into the automated generative pipeline.

![Pipeline Architecture](./images/workflow.jpf)


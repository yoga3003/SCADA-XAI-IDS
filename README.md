# Leakage-Controlled and Robust Explainable Intrusion Detection for SCADA Networks

This repository contains the reproducible research code and experimental artifacts for the study:

**"Leakage-Controlled and Robust Explainable Intrusion Detection for SCADA Networks: An Integrated Evaluation of Generative Augmentation, Perturbation Resilience, and Explanation Stability"**

The work evaluates intrusion detection for SCADA/industrial control system network traffic with particular emphasis on data leakage control, detection performance, controlled perturbation robustness, generative augmentation, and explanation stability.

## Authors

- **YOGALAKSHMI S**
- **SURESH A**

School of Computer Science and Engineering  
Vellore Institute of Technology (VIT), Vellore, India

---

## Research Objective

The objective of this work is to provide an integrated and reproducible evaluation framework for SCADA intrusion detection that considers:

- Detection accuracy
- Data leakage caused by exact duplicate observations
- Classical machine-learning and deep-learning baselines
- Generative augmentation using GANs
- Robustness under controlled feature perturbations
- Perturbation-aware training
- Explainability using SHAP
- Stability of explanations under perturbation
- Multi-seed variability
- Ablation and experiment traceability

The contribution focuses on a **leakage-controlled evaluation protocol and integrated experimental analysis**, rather than proposing a new classifier architecture.

---

## Dataset

The experiments use the **WUSTL-IIOT-2018** dataset.

The primary experiments use the following six traffic features:

```text
Sport
TotPkts
TotBytes
SrcPkts
DstPkts
SrcBytes

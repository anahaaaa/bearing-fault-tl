# bearing-fault-tl

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)
![Status](https://img.shields.io/badge/Status-Research%20In%20Progress-orange)
![Task](https://img.shields.io/badge/Task-Bearing%20Fault%20Diagnosis-success)
![Transfer Learning](https://img.shields.io/badge/Transfer-Learning-purple)
![Domain Adaptation](https://img.shields.io/badge/Domain-Adaptation-yellow)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A comprehensive cross-dataset bearing fault diagnosis framework for studying the impact of vibration signal representations and domain adaptation techniques under varying operating conditions using deep transfer learning.

This project investigates how different signal preprocessing methods influence generalization across datasets and operating domains using a ResNet1D-based architecture and adversarial domain adaptation strategies.

---

## Overview

The repository focuses on cross-domain bearing fault diagnosis using:

- **CWRU (Case Western Reserve University) Bearing Dataset**
- **Paderborn University (PU) Bearing Dataset**

The framework evaluates how well models trained on one dataset generalize to another under significant domain shifts caused by:

- Different machines
- Sensor placements
- Operating conditions
- Load variations
- Signal characteristics

---

## Key Features

- Multi-signal vibration analysis framework
- Cross-dataset transfer learning
- ResNet1D-based fault diagnosis
- Domain-Adversarial Neural Network (DANN)
- Comparative evaluation across signal representations
- Modular preprocessing and training pipeline
- Transfer learning under varying operating conditions

---

## Signal Representations Studied

The project currently supports:

- Raw Signal
- TSA (Time Synchronous Averaging)
- Residual Signal
- Envelope Signal
- Differential Signal

---

## Current Research Direction

This work explores:

- Effectiveness of signal preprocessing for transfer learning
- Cross-dataset domain generalization
- Adversarial domain adaptation for fault diagnosis
- Relationship between signal representation and domain invariance

Initial experiments indicate that signal representation quality may significantly influence transfer performance, sometimes outperforming complex adversarial adaptation strategies.

---

## Implemented Models

### Baseline
- ResNet1D (Source-only Transfer Learning)

### Domain Adaptation
- DANN (Domain-Adversarial Neural Network)

### Planned
- Deep CORAL
- JMMD
- CDAN

---

## Experimental Setup

### Source Domain
- CWRU Dataset

### Target Domain
- PU Dataset

### Domain Shift Factors
- Different loads
- Different sensors
- Different acquisition systems
- Different operating environments

---

## Repository Structure

```bash
bearing-fault-tl/
│
├── data/
│   ├── cwru/
│   └── pu/
│
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   ├── adaptation/
│   ├── evaluation/
│   └── utils/
│
├── notebooks/
│
├── results/
│   ├── source_only/
│   ├── dann/
│   └── visualizations/
│
├── reports/
│
├── README.md
└── requirements.txt
```

---

## Current Findings

- Signal preprocessing has a major impact on transferability.
- Certain representations naturally reduce domain shift.
- DANN performance is highly sensitive to:
  - source-target sample balance
  - signal representation
  - adversarial strength
  - target diversity

The project is evolving toward a deeper study of:

> **Signal Representation vs Domain Adaptation for Cross-Dataset Bearing Fault Diagnosis**

---

## Tech Stack

- Python
- PyTorch
- NumPy
- SciPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## Future Work

- Deep CORAL integration
- Feature visualization using t-SNE
- Multi-branch signal fusion
- Self-supervised pretraining
- Semi-supervised adaptation
- Explainability analysis
- Lightweight deployment models

---

## Tags

`bearing-fault-diagnosis` `transfer-learning` `domain-adaptation` `dann` `deep-learning` `resnet1d` `vibration-analysis` `condition-monitoring` `predictive-maintenance` `fault-diagnosis` `signal-processing` `industrial-ai` `cross-domain-learning` `pytorch` `time-series` `rotating-machinery` `feature-engineering` `unsupervised-domain-adaptation`

---

## Citation

If you find this work useful in your research, please consider citing the repository.

---

## License

This project is licensed under the MIT License.

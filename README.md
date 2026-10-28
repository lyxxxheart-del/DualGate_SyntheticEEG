# When Physiological Plausibility Misleads: Covariance-Aligned Quality Gates for Synthetic Motor-Imagery EEG

> **Official Code Repository for the paper submitted to CSCWD 2027.**
>
> We reveal a critical representational mismatch between physiological ERD gates and downstream CSP decoders, and propose a Dual-Gate framework achieving 0% leakage on DDF-Net synthetic samples.

## 📖 Overview
Generative models are increasingly used to augment scarce EEG data. However, we show that an ERD gate measures temporal energy, whereas the downstream CSP classifier consumes spatial covariance. Our **Dual-Gate framework** (Covariance Plausibility Gate + Discriminative Projection Gate) restores baseline accuracy with no loss by filtering out harmful synthetic samples.

## 🛠️ Key Features
- **Gate 1 (Covariance Validity)**: Detects structural collapse in synthetic EEG (e.g., white noise, per-channel generation artifacts).
- **Gate 2 (Discriminative Projection)**: Detects class-ambiguous synthetic trials using Fisher LDA projection gap.
- **Fast & Lightweight**: Operates entirely on sensor-space covariance features; no expensive eLORETA reconstruction needed.

## 🚀 Quick Start

### 1. Environment Setup
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/lyxxxheart-del/DualGate_SyntheticEEG.git
cd DualGate_SyntheticEEG
pip install -r requirements.txt
```

### 2. Dataset Preparation
Download the BCI Competition IV-2a dataset from the official website and place the `.gdf` files in the `data/raw/` directory.

### 3. Reproduce Main Results
To run the 9-subject experiments and generate the Dual-Gate evaluation results:

```bash
python run_all_subjects.py
```

To generate the figures used in the paper:

```bash
python redraw_paper_figures.py
```

## 📝 Citation
If you find this code useful, please cite our paper:

```bibtex
@inproceedings{liu2026physiological,
  title={When Physiological Plausibility Misleads: Covariance-Aligned Quality Gates for Synthetic Motor-Imagery EEG},
  author={Liu, Yixin},
  booktitle={Proceedings of the 2027 IEEE International Conference on Computer Supported Cooperative Work in Design (CSCWD)},
  year={2027}
}
```

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

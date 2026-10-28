# Physiological Quality Gate for Synthetic Motor-Imagery EEG

A **sensor-space covariance quality gate** for filtering diffusion-model-generated
motor-imagery (MI) EEG, so that *filtered* synthetic data improves a downstream
CSP+LDA classifier in low-real-data regimes. Validated across all 9 subjects of
BCI Competition IV-2a.

## The core idea

Generative MI-EEG augmentation is usually filtered with a *physiological* prior,
most commonly **ERD** (event-related desynchronization: the temporal band-power
drop in the mu/beta rhythm). But the downstream classifier (CSP+LDA) never sees
ERD — CSP consumes the per-trial **spatial covariance** `X @ Xᵀ`. Gating on ERD
and classifying on covariance means the filter and the classifier live in two
different representational spaces: the ERD gate is not just weak, it can *hurt*
(significant accuracy loss on noise-corrupted data).

This project moves the gate **into the exact space the classifier consumes**:

- **Gate 1 — covariance plausibility.** The upper triangle of the (trace-normalized)
  per-trial `X @ Xᵀ` is compared by cosine similarity to the class template; a
  trial must fall inside the 5%/95% percentile window. This rejects samples whose
  spatial structure has collapsed (e.g. white-noise corruption) with ~100% recall.
- **Gate 2 — discriminative direction.** Fisher LDA on the same covariance features
  yields a direction `w`; a trial's projection `features · w` must fall inside its
  own class's score window. This rejects *class-ambiguous* blends whose projection
  lands in the gap between the two classes.

The **dual gate** (`Gate 1 AND Gate 2`) is the proposed filter.

## Results at a glance (9 subjects, 10 seeds, mean accuracy)

| Defect mode | Proportion | Unfiltered | Filtered-Dual | p (paired t) |
|---|---|---|---|---|
| 1. Class mixing | 25% | 0.644 | **0.689** | 0.016 |
| 1. Class mixing | 50% | 0.684 | **0.715** | 0.014 |
| 2. Covariance noise (0 dB) | 25% | 0.633 | **0.674** | 0.001 |
| 2. Covariance noise (0 dB) | 50% | 0.674 | **0.707** | 0.042 |
| 2. Covariance noise (0 dB) | 100% | 0.690 | **0.762** | 0.001 |
| 3. Temporal jitter (±200 ms) | all | ≈ baseline | ≈ baseline | n.s. |

The dual gate significantly recovers accuracy in the two modes it should (class
mixing, covariance noise) and does **not** harm the benign mode (temporal jitter,
which leaves covariance invariant). Traditional augmentation baselines (Gaussian
noise SNR=10 dB, sliding-window time shift) do **not** recover the loss — they are
also computed for comparison. Full tables are written to `reports/` on every run.

## Project structure

```
run_all_subjects.py            # main entry: full 9-subject experiment + stats + CSV
generate_figures.py            # accuracy / retention / projection-gap figures
generate_mechanism_figures.py  # the two paper mechanism figures
src/
  config.py                    # paths, channels, acquisition params, event mapping
  data_loader.py               # GDF -> band-pass-filtered mne.Epochs
  downstream.py                # CSP + LDA, plus the two augmentation baselines
  cov_filter.py                # the dual covariance gate
  experiment.py                # shared core (generators, split, run_subject)
adapters/
  load_ddf_net_synthetic.py    # load + screen DDF-Net synthetic samples
  README_ddf.md                # DDF-Net inference guide
  request_ddf_samples.md       # email template to request pre-generated samples
reports/                       # CSVs, PDF/PNG figures, methodology summary
```

`src/filter.py`, `src/pipeline.py`, `src/calibration.py`, and `test_step2..4.py`
are the earlier ERD/source-space (eLORETA) exploration that motivated this work;
they are kept for provenance but are not part of the final sensor-space pipeline.

## Installation

```bash
# Python 3.14 recommended (the project was developed against it).
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Unix: source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Download **BCI Competition IV dataset 2a** (motor imagery, 9 subjects A01–A09,
two sessions T/E, 22 EEG + 3 EOG channels, 250 Hz):

- Official page: <https://www.bbci.de/competition/iv/#datasets>
- Dataset description: <http://www.bbci.de/competition/iv/desc_2a.pdf>

Place the training GDF files `A01T.gdf` … `A09T.gdf` under `data/raw/`
(the path is configured in `src/config.py` as `RAW_DIR`). Only the `T` sessions
are used; left-hand (769) and right-hand (770) trials are the two classes.

## Reproduce the full experiment

```bash
# All 9 subjects, 10 seeds, the three defect modes, all conditions:
py run_all_subjects.py

# Useful options:
py run_all_subjects.py --subjects A01,A02 --seeds 5      # quick subset
py run_all_subjects.py --normalize trace                 # raw | trace | corr
py run_all_subjects.py --no-cache                        # reload from GDF, skip cache
```

Per-subject epoch arrays are cached under `cache/` (`.npy`) so re-runs skip the
GDF read + band-pass filtering. Outputs written to `reports/`:

- `per_subject_accuracy.csv` — subject-level mean accuracy (long format).
- `aggregate_accuracy.csv` — across-subject mean ± std per (mode, proportion, condition).
- `statistical_tests.csv` — F-Dual vs Unfiltered paired t-test and Wilcoxon.

A full 9-subject / 10-seed run takes **~2–3 minutes** (cold cache) on a desktop CPU.

## Generate the paper figures

```bash
py generate_figures.py            # Fig 1 accuracy, Fig 2 retention, Fig 3 projection gap
py generate_mechanism_figures.py  # mechanism figures: projection-gap KDE + covariance heatmap
```

Both write PDF (vector) and 300-dpi PNG to `reports/`.

## Using real diffusion-generated data (DDF-Net)

The gate is validated on hand-crafted defect modes; to also screen *real*
diffusion samples, the adapter loads DDF-Net output and routes it through the
dual gate. See [adapters/README_ddf.md](adapters/README_ddf.md) for inference
steps and usage:

```bash
py adapters/load_ddf_net_synthetic.py \
    --syn samples/ddf_bciiv2a.npy --syn-labels samples/ddf_bciiv2a_labels.npy \
    --subject A01
```

- DDF-Net code & weights: <https://github.com/HyperSystemAndImageProc/DDF-Net>
- DDF-Net paper: <https://www.sciencedirect.com/science/article/abs/pii/S0952197626002356>

## Methodological write-up

A 6-section method base-draft (Chinese) is in
[reports/methodology_summary.md](reports/methodology_summary.md).

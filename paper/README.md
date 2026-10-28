# IEEE Conference Paper — Covariance-Aligned Quality Gates for Synthetic MI-EEG

## Deliverable

- **Compiled PDF:** `main.pdf` — **5 pages** (within the ≤ 6-page IEEE limit).
- **Source:** `main.tex` (IEEEtran `conference` template, `booktabs` tables).
- **Bibliography:** `references.bib` — 20 real, Google-Scholar-retrievable entries.
- **Figures:** `figures/` (3 core PDF figures copied from `reports/`).

## Compile (MiKTeX)

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Narrative arc

phenomenon (ERD gate misalignment) → mechanism (temporal vs. spatial covariance)
→ solution (dual covariance-aligned gate) → real-model validation (DDF-Net).

## Headline result

DDF-Net (per-channel DDPM) collapses spatial covariance (synthetic cos-sim 0.368
vs. real 0.996). Unfiltered synthetic samples drop CSP accuracy 0.8333 → 0.6667;
the dual gate admits 0/144 samples (0% leakage) and losslessly restores 0.8333.

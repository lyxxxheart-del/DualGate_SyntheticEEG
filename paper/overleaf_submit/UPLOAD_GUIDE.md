# Overleaf Upload Guide — One-Click Submission Package

This directory is a self-contained Overleaf project. Upload these files to
Overleaf and compile; the paper builds to **5 pages** with **20 references**.

---

## 1. What is in this package

```
overleaf_submit/
├── main.tex                          # the full paper (IEEEtran conference)
├── references.bib                    # 20 real BibTeX entries
├── figures/
│   ├── fig_projection_gap.pdf        # Fig. 1 — Gate-2 LDA projection gap
│   ├── fig_covariance_collapse.pdf   # Fig. 2 — real vs. DDF-Net covariance
│   └── fig_cos_sim_rejection.pdf     # Fig. 3 — Gate-1 cos-sim rejection
└── UPLOAD_GUIDE.md                   # this file
```

`main.tex` references the figures with relative paths `figures/fig_*.pdf`, so
**you must keep the `figures/` folder and its three PDFs exactly as shown**,
alongside `main.tex`.

---

## 2. Creating the project (pick ONE)

### Option A — use the official IEEE conference template (recommended)

1. In Overleaf, click **New Project → Templates**.
2. Search for **"IEEE"** and select the **"Preparation of Papers for IEEE
   Sponsored Conferences & Symposia"** template (this is the IEEEtran
   `conference` class).
3. In the new project, delete the template's sample `.tex`/`.bib` files, then
   upload the files from this package (see §3).
4. Keep the template's `IEEEtran.cls` (it is already there). If the template
   ships a `IEEEtran.cls` in the project root, leave it — `main.tex` uses
   `\documentclass[conference]{IEEEtran}`.

### Option B — blank project (also works)

1. **New Project → Blank Project**, name it e.g. `synthetic-eeg-gate`.
2. Upload the files from this package (see §3).
3. `IEEEtran.cls` is pre-installed in Overleaf's TeX Live, so no class file
   needs uploading.

---

## 3. Files to upload (both options)

Drag the following into the project root (and create a `figures/` folder for the
PDFs):

| File / folder | Upload as |
|---|---|
| `main.tex` | project root |
| `references.bib` | project root |
| `figures/fig_projection_gap.pdf` | `figures/fig_projection_gap.pdf` |
| `figures/fig_covariance_collapse.pdf` | `figures/fig_covariance_collapse.pdf` |
| `figures/fig_cos_sim_rejection.pdf` | `figures/fig_cos_sim_rejection.pdf` |

> Tip: you can upload the whole `overleaf_submit/` folder at once (Overleaf's
> "Upload" accepts a folder), which preserves the `figures/` subfolder
> automatically.

---

## 4. Compiler settings

In the top-left **Menu → Settings** of the project:

- **Compiler:** `pdfLaTeX`
- **Main document:** `main.tex`

Overleaf runs `latexmk`, which automatically executes the
`pdflatex → bibtex → pdflatex → pdflatex` sequence needed to resolve the
bibliography. No manual steps are required.

---

## 5. If you get a "missing package" error

All required packages are **already declared** in the preamble of `main.tex`
(lines 5–9). Open `main.tex` and confirm these lines are present:

```latex
\usepackage{amsmath,amssymb,amsfonts}   % line 5  — math typesetting
\usepackage{graphicx}                   % line 6  — \includegraphics (figures)
\usepackage{booktabs}                   % line 7  — \toprule/\midrule/\bottomrule tables
\usepackage{cite}                       % line 8  — compressed [1]–[3] citations
\usepackage[colorlinks=false,urlcolor=black,linkcolor=black,citecolor=black]{hyperref} % line 9
```

**What each error means and the fix:**

| Error | Cause | Fix |
|---|---|---|
| `booktabs.sty not found` | `booktabs` not loaded | Add `\usepackage{booktabs}` after `\documentclass` (line 7 already has it — verify). |
| `graphicx.sty not found` | `graphicx` not loaded | Add `\usepackage{graphicx}` (line 6 already has it). |
| `File 'figures/fig_*.pdf' not found` | figures not uploaded, or wrong folder | Re-upload the three PDFs into a `figures/` subfolder, or edit the `\includegraphics{...}` paths to match your folder. |
| `! LaTeX Error: File 'IEEEtran.cls' not found` | class file missing (blank project only) | Use Option A, or upload a copy of `IEEEtran.cls` to the project root. |
| `Citation 'xyz' undefined` | stale `.aux`/`.bbl` | Click **Recompile from scratch** (Menu → "Recompile" dropdown → "Recompile from scratch"). |

All of these packages are standard TeX Live packages and are pre-installed on
Overleaf, so in practice only the **figure paths** and **IEEEtran.cls** (blank
project) can go wrong.

---

## 6. Expected result

After a successful compile you should see:

- **5 pages** total (within the IEEE 6-page limit for this venue),
- **20 references** in the bibliography, all cited in the text,
- **3 figures** (Fig. 1 projection gap, Fig. 2 covariance collapse, Fig. 3
  cos-sim rejection) and **2 tables** (controlled corruption + DDF-Net result),
- no `Overfull \hbox` warnings and no undefined citations.

---

## 7. Before submission

- Replace the placeholder author block in `main.tex` (lines 20–25:
  `Anonymous Authors` / `Anonymous Institution` / `author@example.com`) with the
  real author names, affiliations, and emails.
- Confirm the target venue's exact page limit (this build is 5 pages; it leaves
  one page of headroom for camera-ready tweaks).

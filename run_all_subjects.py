"""Full 9-subject experiment: covariance-gate across BCI IV-2a A01T..A09T.

Runs the complete pipeline per subject (calibrate gates -> generate the three
defect-mode synthetic sets -> evaluate the six conditions), aggregates
mean +/- std across subjects, and performs paired t-tests / Wilcoxon signed-rank
tests of F-Dual vs Unfiltered at the 9-subject level. Results are written to
``reports/`` as three CSVs:

  per_subject_accuracy.csv   subject-level mean accuracy (long format)
  aggregate_accuracy.csv     across-subject mean / std per (mode, prop, condition)
  statistical_tests.csv      F-Dual vs Unfiltered per (mode, prop)

The gate operates on sensor-space covariance (the exact quantity CSP consumes),
so **no eLORETA / source-feature cache is needed**. The only repeated cost is
loading and band-pass-filtering each subject's epochs; that is cached as raw
arrays under ``cache/`` so re-runs skip the GDF read + filtering.

Run:  py run_all_subjects.py [--subjects A01,A02,...] [--seeds 10]
                              [--modes 1,2,3] [--proportions 0.25,0.5,1.0]
                              [--normalize trace|raw|corr] [--no-cache]
"""

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

from src.config import EVENT_ID
from src.data_loader import load_epochs
from src.experiment import CONDITIONS, MODE_NAMES, run_subject

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
CACHE_DIR = ROOT / "cache"

ALL_SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]  # A01 .. A09

COND_LABELS = {
    "base": "Baseline",
    "unf": "Unfiltered",
    "fcov": "Filtered-Cov",
    "fdual": "Filtered-Dual",
    "noise": "Gaussian-Noise",
    "slide": "Sliding-Window",
}


# ---------------------------------------------------------------------------
# Subject-epoch cache (band-pass-filtered trial arrays + labels)
# ---------------------------------------------------------------------------

def load_subject_arrays(subject, run, cache_dir, use_cache=True):
    """Load a subject's (data, labels), caching the filtered arrays to disk."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path = cache_dir / f"{subject}_{run}_data.npy"
    lab_path = cache_dir / f"{subject}_{run}_labels.npy"

    if use_cache and data_path.exists() and lab_path.exists():
        return np.load(data_path), np.load(lab_path)

    epochs = load_epochs(subject, run)
    data = epochs.get_data()
    labels = epochs.events[:, 2]

    if use_cache:
        np.save(data_path, data)
        np.save(lab_path, labels)

    return data, labels


# ---------------------------------------------------------------------------
# Aggregation + statistics
# ---------------------------------------------------------------------------

def aggregate(subject_means, modes, proportions):
    """Return aggregate mean/std dicts and per-condition subject arrays."""
    agg = {m: {p: {} for p in proportions} for m in modes}
    arrs = {m: {p: {} for p in proportions} for m in modes}

    for m in modes:
        for p in proportions:
            for c in CONDITIONS:
                vals = np.array([subject_means[s][m][p][c] for s in subject_means],
                                dtype=float)
                arrs[m][p][c] = vals
                agg[m][p][c] = (float(np.nanmean(vals)),
                                float(np.nanstd(vals, ddof=1)))
    return agg, arrs


def _paired_tests(a, b):
    """Paired t-test and Wilcoxon signed-rank test; NaN if fewer than 2 pairs."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    t, pt = ttest_rel(a, b)
    if len(a) >= 2:
        w, pw = wilcoxon(a, b, zero_method="zsplit", method="asymptotic")
    else:
        w, pw = np.nan, np.nan
    return t, pt, w, pw


def write_csvs(subjects, subject_means, agg, arrs, modes, proportions):
    REPORTS.mkdir(parents=True, exist_ok=True)

    # 1) per-subject long table
    with open(REPORTS / "per_subject_accuracy.csv", "w", newline="") as f:
        f.write("mode,proportion,subject," + ",".join(CONDITIONS) + "\n")
        for m in modes:
            for p in proportions:
                for s in subjects:
                    row = [str(m), f"{p:.2f}", s]
                    row += [f"{subject_means[s][m][p][c]:.4f}" for c in CONDITIONS]
                    f.write(",".join(row) + "\n")

    # 2) aggregate mean/std
    with open(REPORTS / "aggregate_accuracy.csv", "w", newline="") as f:
        f.write("mode,proportion,condition,mean,std,n_subjects\n")
        for m in modes:
            for p in proportions:
                for c in CONDITIONS:
                    mu, sd = agg[m][p][c]
                    f.write(f"{m},{p:.2f},{c},{mu:.4f},{sd:.4f},{len(subjects)}\n")

    # 3) statistical tests (F-Dual vs Unfiltered)
    with open(REPORTS / "statistical_tests.csv", "w", newline="") as f:
        f.write("mode,proportion,t_stat,p_ttest,wilcoxon_stat,p_wilcoxon\n")
        for m in modes:
            for p in proportions:
                fdual = arrs[m][p]["fdual"]
                unf = arrs[m][p]["unf"]
                t, pt, w, pw = _paired_tests(fdual, unf)
                f.write(f"{m},{p:.2f},{t:+.4f},{pt:.4f},{w:+.4f},{pw:.4f}\n")


def print_summary(subjects, agg, arrs, modes, proportions):
    print("\n" + "=" * 78)
    print(f"Aggregate across {len(subjects)} subjects: "
          f"{', '.join(subjects)}")
    print("=" * 78)
    for m in modes:
        print(f"\n=== Mode {m}: {MODE_NAMES[m]} ===")
        hdr = (f"{'real%':>6} | " +
               " | ".join(f"{COND_LABELS[c]:>15}" for c in CONDITIONS))
        print(hdr)
        for p in proportions:
            cells = [f"{agg[m][p][c][0]:>6.3f}+-{agg[m][p][c][1]:.3f}"
                     for c in CONDITIONS]
            print(f"{int(p * 100):>5}% | " + " | ".join(cells))
            fdual = arrs[m][p]["fdual"]
            unf = arrs[m][p]["unf"]
            t, pt, _, pw = _paired_tests(fdual, unf)
            print(f"       F-Dual vs Unfiltered: t={t:+.3f} p={pt:.3f}  |  "
                  f"Wilcoxon p={pw:.3f}")


def main(args):
    subjects = args.subjects
    modes = args.modes
    proportions = args.proportions
    normalize = args.normalize

    subject_means = {}
    t0 = time.perf_counter()

    for s in subjects:
        t_subj = time.perf_counter()
        print(f"\n[{s}] loading (cache={'off' if args.no_cache else 'on'}) ...")
        data, labels = load_subject_arrays(s, "T", CACHE_DIR,
                                           use_cache=not args.no_cache)
        print(f"[{s}] {data.shape[0]} trials x {data.shape[1]} ch x "
              f"{data.shape[2]} samples")
        res = run_subject(data, labels, seeds=args.seeds, modes=modes,
                          proportions=proportions, normalize=normalize,
                          verbose=args.verbose)
        subject_means[s] = res["means"]
        print(f"[{s}] done in {time.perf_counter() - t_subj:.1f}s")

    agg, arrs = aggregate(subject_means, modes, proportions)
    write_csvs(subjects, subject_means, agg, arrs, modes, proportions)
    print_summary(subjects, agg, arrs, modes, proportions)

    print(f"\nWrote CSVs to {REPORTS}/")
    print(f"Total wall time: {time.perf_counter() - t0:.1f}s "
          f"for {len(subjects)} subject(s), {args.seeds} seeds")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", type=str, default=",".join(ALL_SUBJECTS),
                    help="comma-separated subject ids, e.g. A01,A02 (default: all 9)")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--modes", type=str, default="1,2,3")
    ap.add_argument("--proportions", type=str, default="0.25,0.5,1.0")
    ap.add_argument("--normalize", type=str, default="trace",
                    choices=["raw", "trace", "corr"])
    ap.add_argument("--no-cache", action="store_true",
                    help="skip the epoch-array cache and reload from GDF")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress per-gate calibration prints")
    args = ap.parse_args()

    args.subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    args.modes = [int(x) for x in args.modes.split(",")]
    args.proportions = [float(x) for x in args.proportions.split(",")]
    args.verbose = not args.quiet

    main(args)

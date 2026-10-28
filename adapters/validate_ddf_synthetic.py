"""Finale: dual-gate + downstream CSP validation of DDF-Net synthetic samples.

Produces the four-condition table the gate evaluation needs:

  Baseline       real training data only
  Unfiltered     real + ALL synthetic
  Filtered-Cov   real + synthetic passing Gate 1 (covariance plausibility)
  Filtered-Dual  real + synthetic passing Gate 1 AND Gate 2 (the proposed gate)

Saves the table + pass rates to ``reports/`` (CSV and plain text) and prints it.

Run:
  py adapters/validate_ddf_synthetic.py \
      --syn stage1_out/ddf_synthetic/data.npy \
      --syn-labels stage1_out/ddf_synthetic/labels.npy \
      --subject A01 --normalize trace
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]  # C:\BMI
_ADAPTERS = Path(__file__).resolve().parent        # C:\BMI\adapters
for p in (str(_ADAPTERS), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from load_ddf_net_synthetic import filter_with_dual_gate, load_ddf_net_synthetic  # noqa: E402
from src.data_loader import load_epochs                                              # noqa: E402
from src.downstream import accuracy, fit_csp_lda, predict_csp_lda                   # noqa: E402
from src.experiment import split_train_test                                          # noqa: E402


def main(args):
    syn_data, syn_labels = load_ddf_net_synthetic(args.syn, args.syn_labels)
    if syn_labels is None:
        raise SystemExit("ERROR: --syn-labels is required (the gate needs labels)")

    epochs = load_epochs(args.subject, "T")
    data = epochs.get_data()
    labels = epochs.events[:, 2]
    pool_idx, test_idx = split_train_test(labels)
    pool_data, pool_labels = data[pool_idx], labels[pool_idx]
    test_data, test_labels = data[test_idx], labels[test_idx]

    out = filter_with_dual_gate(syn_data, syn_labels, pool_data, pool_labels,
                                normalize=args.normalize)

    def acc(extra_data, extra_labels):
        tr = np.concatenate([pool_data, extra_data])
        tl = np.concatenate([pool_labels, extra_labels])
        m = fit_csp_lda(tr, tl)
        return accuracy(test_labels, predict_csp_lda(m, test_data))

    rows = {
        "Baseline": acc(pool_data[:0], pool_labels[:0]),
        "Unfiltered": acc(syn_data, syn_labels),
        "Filtered-Cov": acc(syn_data[out["mask_cov"]], syn_labels[out["mask_cov"]]),
        "Filtered-Dual": acc(syn_data[out["mask_dual"]], syn_labels[out["mask_dual"]]),
    }
    kept = {
        "Filtered-Cov": int(out["mask_cov"].sum()),
        "Filtered-Dual": int(out["mask_dual"].sum()),
    }

    print(f"[filter] pass rates -> covariance {out['pass_cov']:.3f} | "
          f"discriminative {out['pass_disc']:.3f} | dual {out['pass_dual']:.3f} "
          f"({int(out['mask_dual'].sum())}/{len(syn_data)} kept)")
    print("\nCondition        Accuracy")
    for k, v in rows.items():
        note = f"  (kept {kept[k]}/{len(syn_data)})" if k in kept else ""
        print(f"{k:<15}  {v:.4f}{note}")

    report_dir = _REPO_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.syn).stem
    csv_path = report_dir / f"ddf_gate_validation_{stem}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "accuracy", "n_synthetic_kept"])
        w.writerow(["Baseline", f"{rows['Baseline']:.6f}", "-"])
        w.writerow(["Unfiltered", f"{rows['Unfiltered']:.6f}", len(syn_data)])
        w.writerow(["Filtered-Cov", f"{rows['Filtered-Cov']:.6f}", kept["Filtered-Cov"]])
        w.writerow(["Filtered-Dual", f"{rows['Filtered-Dual']:.6f}", kept["Filtered-Dual"]])
    txt_path = report_dir / f"ddf_gate_validation_{stem}.txt"
    with open(txt_path, "w") as f:
        f.write(f"subject={args.subject} normalize={args.normalize} "
                f"n_syn={len(syn_data)}\n")
        f.write(f"pass rates: cov={out['pass_cov']:.4f} disc={out['pass_disc']:.4f} "
                f"dual={out['pass_dual']:.4f}\n\n")
        for k, v in rows.items():
            f.write(f"{k:<15} {v:.6f}\n")
    print(f"[save] {csv_path}")
    print(f"[save] {txt_path}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--syn", required=True)
    ap.add_argument("--syn-labels", required=True)
    ap.add_argument("--subject", default="A01")
    ap.add_argument("--normalize", default="trace")
    args = ap.parse_args()
    sys.exit(main(args))

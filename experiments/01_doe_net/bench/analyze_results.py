"""Aggregate the feature-selection benchmark (bench/PROTOCOL.md §5).

Reads every results*.jsonl in RUN_DIR (cells deduplicated by dataset/method/seed, error rows
reported separately) and writes RUN_DIR/analysis/{summary.json, tables.md, figures/*.png}:
  - downstream test accuracy with the top-k features: mean ± sample SD over seeds
  - methods' own test accuracy (all features)
  - paired t-test over seeds (doenet vs each method, per dataset)
  - average rank over datasets (selection methods only) and Wilcoxon signed-rank test over
    dataset means (doenet vs each method)
  - selection stability: mean pairwise Jaccard index of top-k sets across seeds
  - DOENet ANOVA variance shares; selected-pixel frequency maps for MNIST / Fashion-MNIST
Published numbers (Sequential Attention, ICLR'23, Table 2/3) are listed as external references.

Usage: python bench/analyze_results.py RUN_DIR [--ref doenet]
"""

import argparse
import glob
import itertools
import json
import os

import numpy as np
from scipy import stats

DATASETS = ["mice", "mnist", "fashion", "isolet", "coil20", "activity"]
ORDER = ["all_features", "random", "doenet", "doenet_noortho", "doenet_noorthoreg", "gated_mlp",
         "stg", "lassonet", "tabnet", "sa", "seql", "gl"]
REFERENCE_ONLY = {"all_features", "random"}
# Yasuda et al. (ICLR 2023) Table 2 (k = 50, mean of 5 runs) and Table 3 (all features)
PUBLISHED_SA = {
    "SA (paper)":   [0.993, 0.956, 0.854, 0.920, 0.997, 0.931],
    "SL (paper)":   [0.984, 0.959, 0.854, 0.920, 0.988, 0.931],
    "GL (paper)":   [0.985, 0.937, 0.834, 0.906, 0.997, 0.933],
    "LLY (paper)":  [0.981, 0.944, 0.843, 0.866, 0.994, 0.897],
    "OMP (paper)":  [0.994, 0.912, 0.829, 0.727, 0.967, 0.905],
    "CAE (paper)":  [0.956, 0.909, 0.839, 0.893, 0.972, 0.921],
    "All features (paper)": [0.963, 0.953, 0.869, 0.961, 0.986, 0.954],
}


def load(run_dir):
    cells, errors = {}, []
    for p in sorted(glob.glob(os.path.join(run_dir, "results*.jsonl"))):
        for line in open(p):
            r = json.loads(line)
            if r.get("val_only"):
                continue
            key = (r["dataset"], r["method"], int(r["seed"]))
            if "error" in r:
                errors.append({**{k: r[k] for k in ("dataset", "method", "seed")}, "error": r["error"][:200], "file": os.path.basename(p)})
                continue
            cells[key] = r
    return cells, errors


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ref", default="doenet")
    ap.add_argument("--quiet", action="store_true", help="do not print accuracy tables")
    args = ap.parse_args()
    out = os.path.join(args.run_dir, "analysis")
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    cells, errors = load(args.run_dir)
    methods = [m for m in ORDER if any(k[1] == m for k in cells)] + \
              sorted({k[1] for k in cells} - set(ORDER))

    acc = {m: {d: {} for d in DATASETS} for m in methods}
    own = {m: {d: {} for d in DATASETS} for m in methods}
    for (d, m, s), r in cells.items():
        acc[m][d][s] = r["downstream_test_acc"]
        if r.get("own_test_acc") is not None and np.isfinite(r["own_test_acc"]):
            own[m][d][s] = r["own_test_acc"]

    def msd(vals):
        v = np.array(list(vals), dtype=float)
        return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan"), len(v)) if len(v) else (np.nan, np.nan, 0)

    summary = {"n_cells": len(cells), "errors": errors, "accuracy": {}, "own_accuracy": {},
               "paired_t": {}, "avg_rank": {}, "wilcoxon": {}, "stability_jaccard": {}, "anova": {}}
    for m in methods:
        summary["accuracy"][m] = {d: dict(zip(("mean", "sd", "n"), msd(acc[m][d].values()))) for d in DATASETS}
        summary["own_accuracy"][m] = {d: dict(zip(("mean", "sd", "n"), msd(own[m][d].values()))) for d in DATASETS if own[m][d]}

    ref = args.ref
    for m in methods:
        if m == ref:
            continue
        summary["paired_t"][m] = {}
        for d in DATASETS:
            seeds = sorted(set(acc[ref][d]) & set(acc[m][d]))
            if len(seeds) >= 2:
                a = np.array([acc[ref][d][s] for s in seeds]); b = np.array([acc[m][d][s] for s in seeds])
                p = float(stats.ttest_rel(a, b).pvalue) if np.any(a != b) else 1.0
                summary["paired_t"][m][d] = {"diff_mean": float((a - b).mean()), "p": p, "n": len(seeds)}

    sel_methods = [m for m in methods if m not in REFERENCE_ONLY]
    full = [d for d in DATASETS if all(summary["accuracy"][m][d]["n"] > 0 for m in sel_methods)]
    if full:
        means = np.array([[summary["accuracy"][m][d]["mean"] for m in sel_methods] for d in full])
        ranks = np.array([stats.rankdata(-row) for row in means])
        summary["avg_rank"] = {"datasets": full, **{m: float(ranks[:, i].mean()) for i, m in enumerate(sel_methods)}}
        if ref in sel_methods:
            i_ref = sel_methods.index(ref)
            for i, m in enumerate(sel_methods):
                if m == ref:
                    continue
                diff = means[:, i_ref] - means[:, i]
                p = float(stats.wilcoxon(means[:, i_ref], means[:, i]).pvalue) if np.any(diff != 0) else 1.0
                summary["wilcoxon"][m] = {"wins": int((diff > 0).sum()), "ties": int((diff == 0).sum()),
                                          "losses": int((diff < 0).sum()), "p": p, "n_datasets": len(full)}

    for m in methods:
        if m == "all_features":
            continue
        summary["stability_jaccard"][m] = {}
        for d in DATASETS:
            sets = [cells[(d, m, s)]["selected"] for s in sorted(acc[m][d]) if cells[(d, m, s)].get("selected")]
            if len(sets) >= 2:
                summary["stability_jaccard"][m][d] = float(np.mean([jaccard(a, b) for a, b in itertools.combinations(sets, 2)]))

    for d in DATASETS:
        rows = [cells[(d, ref, s)]["anova"] for s in sorted(acc.get(ref, {}).get(d, {})) if "anova" in cells[(d, ref, s)]]
        if rows:
            summary["anova"][d] = {k: float(np.mean([r[k] for r in rows])) for k in ("main_effect_pct", "interaction_pct", "residual_pct")}

    # selected-pixel frequency maps (MNIST / Fashion)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for d in ("mnist", "fashion"):
            ms = [m for m in sel_methods if acc[m][d]]
            if not ms:
                continue
            fig, axes = plt.subplots(1, len(ms), figsize=(1.8 * len(ms), 2.2))
            for ax, m in zip(np.atleast_1d(axes), ms):
                freq = np.zeros(784)
                for s in acc[m][d]:
                    freq[cells[(d, m, s)]["selected"]] += 1
                ax.imshow((freq / max(1, len(acc[m][d]))).reshape(28, 28), cmap="gray_r", vmin=0, vmax=1)
                ax.set_title(m, fontsize=7)
                ax.axis("off")
            fig.suptitle(f"{d}: selection frequency of top-50 pixels over seeds", fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(out, "figures", f"{d}_selected_pixels.png"), dpi=200)
            plt.close(fig)
    except Exception as e:  # pragma: no cover
        summary["figure_error"] = repr(e)

    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)

    # markdown tables
    L = ["## Downstream test accuracy, top-50 features (mean ± SD over seeds)", "",
         "| method | " + " | ".join(DATASETS) + " |", "|---" * (len(DATASETS) + 1) + "|"]
    for m in methods:
        row = []
        for d in DATASETS:
            a = summary["accuracy"][m][d]
            row.append("–" if a["n"] == 0 else f"{a['mean']:.3f} ± {a['sd']:.3f} (n={a['n']})")
        L.append(f"| {m} | " + " | ".join(row) + " |")
    L += ["", "## Published reference (Sequential Attention, ICLR'23, Table 2/3; not part of the tests)", "",
          "| method | " + " | ".join(DATASETS) + " |", "|---" * (len(DATASETS) + 1) + "|"]
    for m, v in PUBLISHED_SA.items():
        L.append(f"| {m} | " + " | ".join(f"{x:.3f}" for x in v) + " |")
    if summary["avg_rank"]:
        L += ["", f"## Average rank over {len(summary['avg_rank']['datasets'])} datasets (selection methods)", ""]
        L += [f"- {m}: {r:.2f}" for m, r in sorted(((m, r) for m, r in summary["avg_rank"].items() if m != "datasets"), key=lambda x: x[1])]
    open(os.path.join(out, "tables.md"), "w").write("\n".join(L) + "\n")
    print(f"cells={len(cells)} errors={len(errors)} methods={len(methods)} -> {out}")
    if not args.quiet:
        print("\n".join(L))


if __name__ == "__main__":
    main()

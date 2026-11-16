"""Validation-only hyper-parameter selection (protocol v2, see bench/PROTOCOL.md).

For each method family and grid point, run_fs_benchmark.py is executed with --val_only on
seed 1 (the test split is never evaluated). Per dataset, the grid point with the highest
downstream validation accuracy of the top-k features is selected; ties go to the earlier
grid point (defaults are listed first).

Usage: python bench/tune_val.py --data_dir DATA --work_dir DIR [--families doenet,stg]
Writes DIR/<family>_<i>.jsonl, DIR/tuned_config.json, DIR/tuning_summary.json
"""

import argparse
import json
import os
import subprocess
import sys

GRID = {
    # defaults first (tie-break)
    "doenet": [{"lambda_sparse": ls, "lambda_ortho": lo}
               for ls in (2e-3, 1e-2, 5e-2) for lo in (1e-2, 1e-1)],
    "stg": [{"stg_lam": lam} for lam in (0.1, 0.01, 0.5)],
}
METHOD = {"doenet": "doenet", "stg": "stg"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--families", default="doenet,stg")
    ap.add_argument("--datasets", default="mice,coil20,isolet,activity,mnist,fashion")
    ap.add_argument("--seed", type=int, default=1)
    args, passthrough = ap.parse_known_args()
    os.makedirs(args.work_dir, exist_ok=True)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets = args.datasets.split(",")

    for fam in args.families.split(","):
        for i, cfg in enumerate(GRID[fam]):
            cfg_path = os.path.join(args.work_dir, f"{fam}_{i}.config.json")
            json.dump({fam: {ds: cfg for ds in datasets}}, open(cfg_path, "w"))
            cmd = [sys.executable, os.path.join(here, "run_fs_benchmark.py"),
                   "--data_dir", args.data_dir, "--out", os.path.join(args.work_dir, f"{fam}_{i}.jsonl"),
                   "--datasets", args.datasets, "--methods", METHOD[fam], "--seeds", str(args.seed),
                   "--config_json", cfg_path, "--val_only", *passthrough]
            print(f"[tune] {fam} grid {i}: {cfg}", flush=True)
            subprocess.run(cmd, cwd=here, check=False)

    tuned, summary = {}, {}
    for fam in args.families.split(","):
        tuned[fam], summary[fam] = {}, {}
        for ds in datasets:
            rows = []
            for i, cfg in enumerate(GRID[fam]):
                p = os.path.join(args.work_dir, f"{fam}_{i}.jsonl")
                recs = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
                r = next((r for r in recs if r["dataset"] == ds and "error" not in r), None)
                assert r is None or r["val_only"], "tuning rows must be val-only"
                rows.append((i, cfg, None if r is None else r["downstream_val_acc"]))
            ok = [x for x in rows if x[2] is not None]
            best = max(ok, key=lambda x: (x[2], -x[0])) if ok else (0, GRID[fam][0], None)
            tuned[fam][ds] = best[1]
            summary[fam][ds] = {"chosen_grid_index": best[0], "chosen": best[1],
                                "val_acc_by_grid": [{"i": i, **cfg, "val_acc": v} for i, cfg, v in rows]}
    json.dump(tuned, open(os.path.join(args.work_dir, "tuned_config.json"), "w"), indent=1)
    json.dump(summary, open(os.path.join(args.work_dir, "tuning_summary.json"), "w"), indent=1)
    print(json.dumps(tuned, indent=1))


if __name__ == "__main__":
    main()

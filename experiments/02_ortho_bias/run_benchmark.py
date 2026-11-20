"""OrthoBias benchmark suite.

Runs, in one command:

1. **Oracle recovery** on the five structured families: does the model recover the
   reference variance shares (and in particular the identifiable triple
   rho_1, rho_2, rho_3 + rho_4), and does it attribute high-order structure to the
   right block (axis-aligned H3 for the ``tree`` family, smooth H4 for ``deep``)?
2. **Orthogonality audit**: block-to-block correlations before orthogonalization,
   after the exact projection, and with the deployed (fixed-coefficient) operator.
3. **Ablations**: no-orthogonalization, no meta-profiler warm start, and a
   tau_interp sweep tracing the accuracy / interpretability frontier.
4. **Tables** in Markdown and LaTeX.

Usage
-----
    python3 run_benchmark.py --smoke
    python3 run_benchmark.py --seeds 0 1 2 3 4 --epochs 150 --ablations
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import gc

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.synthetic_oracle import FAMILIES, OracleTask, make_task
from models.baselines import library_baselines, neural_baselines
from models.orthobias import OrthoBias, OrthoBiasConfig, OrthoBiasTrainer, evaluate
from models.profiler import MetaProfiler, compute_signatures

EPS = 1e-8


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def build_model(task: OracleTask, args, profiler: Optional[MetaProfiler] = None,
                tau: Optional[float] = None, ortho: bool = True,
                meta: bool = True) -> OrthoBias:
    cfg = OrthoBiasConfig(
        n_features=task.d,
        pairs=task.pairs,
        n_bins=args.n_bins,
        n_trees=args.trees,
        tree_depth=args.tree_depth,
        hidden=args.hidden,
        res_blocks=args.res_blocks,
        tau_interp=tau,
        ortho_mode="running" if ortho else "none",
        lam_ortho=args.lam_ortho if ortho else 0.0,
        lam_l0=args.lam_l0,
        lam_mdl=args.lam_mdl,
    )
    model = OrthoBias(cfg).fit_preprocessing(task.X_train)
    if meta and profiler is not None:
        sig = compute_signatures(task.X_train, task.y_train, pairs=task.pairs,
                                 max_pairs=args.max_pairs, n_bins=args.n_bins, n_probe=768)
        model.apply_warm_start(profiler, sig)
    return model


def fit_model(model: OrthoBias, task: OracleTask, args, seed: int = 0) -> Dict[str, float]:
    trainer = OrthoBiasTrainer(
        model, lr=args.lr, lr_deep=args.lr_deep, tau_interp=model.cfg.tau_interp,
        lam_mdl=model.cfg.lam_mdl, lam_l0=model.cfg.lam_l0, lam_ortho=model.cfg.lam_ortho,
        seed=seed,
    )
    t0 = time.time()
    trainer.fit(task.X_train, task.y_train, epochs=args.epochs, batch_size=args.batch,
                staged=not args.no_staging, log_every=0, calibrate=args.calibrate)
    fit_time = time.time() - t0
    out = evaluate(model, task.X_test, task.y_test)
    out["fit_time"] = fit_time
    return out


def block_corr(f: torch.Tensor) -> torch.Tensor:
    c = f - f.mean(dim=0, keepdim=True)
    cov = c.T @ c / f.shape[0]
    d = torch.diagonal(cov).clamp_min(EPS)
    return cov / torch.sqrt(d[:, None] * d[None, :])


@torch.no_grad()
def exact_block_corr(model: OrthoBias, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(raw, exactly projected) block correlation matrices on ``X``.

    The exact projection uses the least-squares coefficients of the *given* sample,
    which is the population projection in the limit: it is the theoretical
    orthogonalization, not the deployed one.
    """
    raw = model.raw_blocks(X)
    p1 = model.phi1(X)
    p2 = torch.cat([p1, model.basis.pair_flat(X, model.h2.pairs)], dim=1)
    p3 = torch.cat([p2, raw["h3"].detach().unsqueeze(1)], dim=1)

    def proj(phi: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if phi.shape[1] == 0:
            return h
        scale = (phi * phi).sum(dim=0).clamp_min(1e-12).sqrt()
        xs = (phi.double() / scale)
        w = torch.linalg.lstsq(xs, h.detach().double().unsqueeze(-1)).solution.squeeze(-1)
        return h - (xs @ w).float()

    h1 = raw["h1"]
    h2 = proj(p1, raw["h2"])
    h3 = proj(p2, raw["h3"])
    h4 = proj(p3, raw["h4"])
    return block_corr(torch.stack([raw["h1"], raw["h2"], raw["h3"], raw["h4"]], 1)), \
        block_corr(torch.stack([h1, h2, h3, h4], 1))


def offdiag_stats(corr: torch.Tensor) -> Tuple[float, float]:
    c = corr.clone()
    c.fill_diagonal_(0.0)
    return float(c.abs().max()), float(c.abs().sum() / (c.shape[0] * (c.shape[0] - 1)))


def identified(rho: Sequence[float]) -> Tuple[float, float, float]:
    return float(rho[0]), float(rho[1]), float(rho[2] + rho[3])


def share_errors(rho: Sequence[float], rho_star: Sequence[float]) -> Dict[str, float]:
    r1, r2, rh = identified(rho)
    s1, s2, sh = identified(rho_star)
    return {
        "err_block_l1": float(sum(abs(a - b) for a, b in zip(rho, rho_star))),
        "err_ident_l1": abs(r1 - s1) + abs(r2 - s2) + abs(rh - sh),
        "err_main": abs(r1 - s1),
        "err_pair": abs(r2 - s2),
        "err_high": abs(rh - sh),
        "attrib_hi_to_h3": float(rho[2] / max(rho[2] + rho[3], EPS)),
    }


def mean_std(xs: Sequence[float]) -> str:
    if not xs:
        return "-"
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return f"{m:.3f}"
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return f"{m:.3f}±{sd:.3f}"


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------


def run_oracle(tasks: List[OracleTask], args, profiler: Optional[MetaProfiler]) -> List[Dict]:
    rows = []
    for task in tasks:
        t0 = time.time()
        model = build_model(task, args, profiler, tau=None)
        res = fit_model(model, task, args, seed=task.seed)
        corr_raw, corr_exact = exact_block_corr(model, task.X_test)
        diag = model.diagnostics(task.X_test)
        row = {
            "task": task.name, "family": task.family, "seed": task.seed,
            "r2": res["r2"], "rho": res["rho"], "ir": res["ir"], "gates": res["gates"],
            "pruned": res["pruned"], "fit_time": res["fit_time"],
            "rho_star": task.rho_star,
            "corr_raw_max": offdiag_stats(corr_raw)[0],
            "corr_exact_max": offdiag_stats(corr_exact)[0],
            "corr_deploy_max": offdiag_stats(diag["corr_res"])[0],
            "corr_raw_mean": offdiag_stats(corr_raw)[1],
            "corr_exact_mean": offdiag_stats(corr_exact)[1],
            "corr_deploy_mean": offdiag_stats(diag["corr_res"])[1],
            "n_params": res["n_params"],
        }
        row.update(share_errors(res["rho"], task.rho_star))
        # baselines
        if args.skip_baselines:
            rows.append(row)
            print(f"  [oracle] {task.name:24s} R2 {row['r2']:.3f}  rho {[round(v, 2) for v in row['rho']]}"
                  f"  err_id {row['err_ident_l1']:.3f}  corr {row['corr_raw_max']:.2f}/"
                  f"{row['corr_exact_max']:.1e}/{row['corr_deploy_max']:.2f}  ({time.time() - t0:.1f}s)")
            del model
            gc.collect()
            continue
        for base in neural_baselines(task.seed):
            base.fit(task.X_train, task.y_train, seed=task.seed)
            row[f"r2_{base.name}"] = r2_score(base.predict(task.X_test), task.y_test)
            del base
            gc.collect()
        for base in library_baselines(task.seed):
            try:
                base.fit(task.X_train, task.y_train, seed=task.seed)
                row[f"r2_{base.name}"] = r2_score(base.predict(task.X_test), task.y_test)
            except Exception as exc:  # pragma: no cover
                row[f"r2_{base.name}"] = float("nan")
            del base
            gc.collect()
        del model
        gc.collect()
        rows.append(row)
        print(f"  [oracle] {task.name:24s} R2 {row['r2']:.3f}  rho {[round(v, 2) for v in row['rho']]}"
              f"  err_id {row['err_ident_l1']:.3f}  corr(raw/exact/deploy) "
              f"{row['corr_raw_max']:.2f}/{row['corr_exact_max']:.1e}/{row['corr_deploy_max']:.2f}"
              f"  ({time.time() - t0:.1f}s)")
    return rows


def r2_score(pred: torch.Tensor, y: torch.Tensor) -> float:
    pred = pred.detach().float().reshape(-1)
    y = y.detach().float().reshape(-1)
    ss_res = ((y - pred) ** 2).mean()
    ss_tot = ((y - y.mean()) ** 2).mean().clamp_min(EPS)
    return float(1.0 - ss_res / ss_tot)


def run_ablations(tasks: List[OracleTask], args, profiler: Optional[MetaProfiler]) -> List[Dict]:
    variants = {
        "full": dict(ortho=True, meta=True),
        "no-ortho": dict(ortho=False, meta=True),
        "no-meta": dict(ortho=True, meta=False),
    }
    rows = []
    for task in tasks:
        for name, kw in variants.items():
            model = build_model(task, args, profiler, tau=None, **kw)
            res = fit_model(model, task, args, seed=task.seed)
            diag = model.diagnostics(task.X_test)
            row = {"task": task.name, "family": task.family, "seed": task.seed, "variant": name,
                   "r2": res["r2"], "ir": res["ir"], "rho": res["rho"],
                   "corr_deploy_max": offdiag_stats(diag["corr_res"])[0],
                   "fit_time": res["fit_time"]}
            row.update(share_errors(res["rho"], task.rho_star))
            del model
            gc.collect()
            rows.append(row)
            print(f"  [ablate] {task.name:20s} {name:10s} R2 {res['r2']:.3f} "
                  f"err_id {row['err_ident_l1']:.3f} corr {row['corr_deploy_max']:.2f}")
    return rows


def run_tau_sweep(tasks: List[OracleTask], args, profiler: Optional[MetaProfiler]) -> List[Dict]:
    taus = [t for t in args.taus]
    rows = []
    for task in tasks:
        for tau in taus:
            model = build_model(task, args, profiler, tau=tau)
            res = fit_model(model, task, args, seed=task.seed)
            row = {"task": task.name, "family": task.family, "seed": task.seed, "tau": tau,
                   "r2": res["r2"], "ir": res["ir"], "gates": res["gates"],
                   "pruned": res["pruned"], "rho": res["rho"]}
            del model
            gc.collect()
            rows.append(row)
            print(f"  [tau] {task.name:20s} tau {tau:.2f} R2 {res['r2']:.3f} IR {res['ir']:.3f} "
                  f"pruned {sum(res['pruned'])}/4")
    return rows


def run_prune_check(args) -> Dict[str, object]:
    """Exact-zero block pruning on a linear task (only H1 is needed)."""
    torch.manual_seed(0)
    n, d = 1500, 6
    w = torch.randn(d)
    X = torch.randn(n, d)
    y = X @ w + 0.05 * torch.randn(n)
    cfg = OrthoBiasConfig(
        n_features=d, pairs=[(0, 1), (2, 3), (4, 5)], n_bins=args.n_bins,
        n_trees=args.trees, tree_depth=args.tree_depth, hidden=args.hidden,
        res_blocks=args.res_blocks,
    )
    model = OrthoBias(cfg).fit_preprocessing(X)
    trainer = OrthoBiasTrainer(model, lr=args.lr, lr_deep=args.lr_deep)
    trainer.fit(X, y, epochs=args.epochs, batch_size=args.batch)
    before = evaluate(model, X, y)
    info = trainer.greedy_prune(X[:400], y[:400], tol=args.prune_tol)
    after = evaluate(model, X, y)
    return {
        "alpha": info["alpha"],
        "gates": info["gates"],
        "pruned": info["pruned_blocks"],
        "r2_before": before["r2"],
        "r2_after": after["r2"],
        "ir_before": before["ir"],
        "ir_after": after["ir"],
        "tol": args.prune_tol,
        "exact_zero": [abs(g) < 1e-12 for g in info["gates"]],
    }


def check_invariance(args) -> Dict[str, float]:
    """Permutation invariance of the profiler and of the fixed-basis model."""
    torch.manual_seed(0)
    task = make_task("hybrid", n_train=600, d=8, seed=0, max_pairs=4, n_bins=args.n_bins)
    prof = MetaProfiler()
    sig = compute_signatures(task.X_train, task.y_train, pairs=None, max_pairs=args.max_pairs,
                             n_bins=args.n_bins)
    perm = torch.randperm(task.d)
    sig2 = compute_signatures(task.X_train[:, perm], task.y_train, pairs=None,
                              max_pairs=args.max_pairs, n_bins=args.n_bins)
    with torch.no_grad():
        a, b = prof(sig), prof(sig2)
    return {
        "profiler_rho_maxdiff": float((a["rho"] - b["rho"]).abs().max()),
        "profiler_block_logit_maxdiff": float((a["block_logits"] - b["block_logits"]).abs().max()),
        "profiler_screening_setdiff": float(
            (a["feature_logits"].sort().values - b["feature_logits"].sort().values).abs().max()
        ),
    }


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def write_tables(rows: List[Dict], abl: List[Dict], tau_rows: List[Dict], args, extra: Dict,
                 tag: str = "all") -> Dict[str, str]:
    out_dir = os.path.join(args.out, "tables")
    os.makedirs(out_dir, exist_ok=True)
    md = []
    tex = []

    fams = [f for f in FAMILIES if any(r["family"] == f for r in rows)]
    base_names = [k[3:] for k in rows[0]
                  if k.startswith("r2_") and k[3:]] if rows else []

    md.append("## 1. Oracle recovery and predictive accuracy\n")
    base_hdr = (" | ".join(base_names) + " | ") if base_names else ""
    header = ("| family | R2 OrthoBias | " + base_hdr +
              "err(ident) | err(main) | err(pair) | err(high) | attribution H3 |")
    md.append(header)
    md.append("|" + "---|" * (len(base_names) + 7))
    for fam in fams:
        sel = [r for r in rows if r["family"] == fam]
        cells = [fam, mean_std([r["r2"] for r in sel])]
        for b in base_names:
            cells.append(mean_std([r[f"r2_{b}"] for r in sel if f"r2_{b}" in r and not math.isnan(r[f"r2_{b}"])]))
        cells += [
            mean_std([r["err_ident_l1"] for r in sel]),
            mean_std([r["err_main"] for r in sel]),
            mean_std([r["err_pair"] for r in sel]),
            mean_std([r["err_high"] for r in sel]),
            mean_std([r["attrib_hi_to_h3"] for r in sel]),
        ]
        md.append("| " + " | ".join(cells) + " |")

    md.append("\nGround-truth reference shares (first seed of each family):\n")
    md.append("| family | rho* (H1/H2/H3/H4) | rho_hat |")
    md.append("|---|---|---|")
    for fam in fams:
        r = [x for x in rows if x["family"] == fam][0]
        md.append(f"| {fam} | {[round(v, 3) for v in r['rho_star']]} | {[round(v, 3) for v in r['rho']]} |")

    md.append("\n## 2. Orthogonality audit (|corr| between block outputs)\n")
    md.append("| family | raw max | exact-projection max | deployed max | raw mean | exact mean | deployed mean |")
    md.append("|---|---|---|---|---|---|---|")
    for fam in fams:
        sel = [r for r in rows if r["family"] == fam]
        md.append("| " + " | ".join([
            fam,
            mean_std([r["corr_raw_max"] for r in sel]),
            f"{max(r['corr_exact_max'] for r in sel):.1e}",
            mean_std([r["corr_deploy_max"] for r in sel]),
            mean_std([r["corr_raw_mean"] for r in sel]),
            f"{max(r['corr_exact_mean'] for r in sel):.1e}",
            mean_std([r["corr_deploy_mean"] for r in sel]),
        ]) + " |")

    if abl:
        md.append("\n## 3. Ablations\n")
        md.append("| family | variant | R2 | err(ident) | IR | max block corr |")
        md.append("|---|---|---|---|---|---|")
        for fam in sorted(set(r["family"] for r in abl)):
            for var in ["full", "no-ortho", "no-meta"]:
                sel = [r for r in abl if r["family"] == fam and r["variant"] == var]
                if not sel:
                    continue
                md.append("| " + " | ".join([
                    fam, var,
                    mean_std([r["r2"] for r in sel]),
                    mean_std([r["err_ident_l1"] for r in sel]),
                    mean_std([r["ir"] for r in sel]),
                    mean_std([r["corr_deploy_max"] for r in sel]),
                ]) + " |")

    if tau_rows:
        md.append("\n## 4. Interpretability constraint sweep\n")
        md.append("| family | tau | R2 | IR | pruned blocks |")
        md.append("|---|---|---|---|---|")
        for fam in sorted(set(r["family"] for r in tau_rows)):
            for tau in sorted(set(r["tau"] for r in tau_rows)):
                sel = [r for r in tau_rows if r["family"] == fam and r["tau"] == tau]
                if not sel:
                    continue
                md.append("| " + " | ".join([
                    fam, f"{tau:.2f}",
                    mean_std([r["r2"] for r in sel]),
                    mean_std([r["ir"] for r in sel]),
                    mean_std([float(sum(r["pruned"])) for r in sel]),
                ]) + " |")

    if extra.get("prune"):
        p = extra["prune"]
        md.append("\n## 5. Exact-zero block pruning (linear target, only H1 needed)\n")
        md.append(f"- gate parameters alpha: {[round(a, 2) for a in p['alpha']]}")
        md.append(f"- evaluation gates: {[round(g, 3) for g in p['gates']]}")
        md.append(f"- exactly zero gates: {p['exact_zero']}")
        md.append(f"- pruned blocks (index): {p['pruned']}")
        md.append(f"- R2 {p['r2_before']:.4f} -> {p['r2_after']:.4f}, IR {p['ir_before']:.3f} -> {p['ir_after']:.3f}, tolerance {p['tol']}")

    if extra.get("invariance"):
        md.append("\n## 6. Invariance check\n")
        for k, v in extra["invariance"].items():
            md.append(f"- `{k}` = {v:.2e}")

    md_path = os.path.join(args.out, "benchmark.md" if tag == "all" else f"benchmark_{tag}.md")
    with open(md_path, "w") as f:
        f.write("# OrthoBias benchmark report\n\n")
        f.write(f"configuration: {json.dumps(extra.get('config', {}), indent=2)}\n\n")
        f.write("\n".join(md) + "\n")

    # ---- LaTeX -----------------------------------------------------------
    tex.append(r"\begin{table}[t]\centering\small")
    tex.append(r"\caption{Predictive accuracy and recovery of the reference variance shares.}")
    tex.append(r"\begin{tabular}{l" + "c" * (len(base_names) + 6) + "}")
    tex.append(r"\toprule")
    tex.append("Family & OrthoBias & " + " & ".join(base_names) + r" & $\varepsilon_{\mathrm{id}}$ & $\varepsilon_{1}$ & $\varepsilon_{2}$ & $\varepsilon_{hi}$ & $a_3$ \\")
    tex.append(r"\midrule")
    for fam in fams:
        sel = [r for r in rows if r["family"] == fam]
        cells = [fam.replace("_", r"\_"), mean_std([r["r2"] for r in sel]).replace("±", r"$\pm$")]
        for b in base_names:
            cells.append(mean_std([r[f"r2_{b}"] for r in sel if f"r2_{b}" in r and not math.isnan(r[f"r2_{b}"])]).replace("±", r"$\pm$"))
        cells += [mean_std([r["err_ident_l1"] for r in sel]).replace("±", r"$\pm$"),
                  mean_std([r["err_main"] for r in sel]).replace("±", r"$\pm$"),
                  mean_std([r["err_pair"] for r in sel]).replace("±", r"$\pm$"),
                  mean_std([r["err_high"] for r in sel]).replace("±", r"$\pm$"),
                  mean_std([r["attrib_hi_to_h3"] for r in sel]).replace("±", r"$\pm$")]
        tex.append(" & ".join(cells) + r" \\")
    tex.append(r"\bottomrule\end{tabular}\end{table}")
    tex_path = os.path.join(out_dir, "oracle_recovery.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(tex) + "\n")

    json_path = os.path.join(args.out, "metrics.json" if tag == "all" else f"metrics_{tag}.json")
    with open(json_path, "w") as f:
        json.dump({"oracle": rows, "ablations": abl, "tau": tau_rows, "extra": extra}, f, indent=2)
    return {"markdown": md_path, "latex": tex_path, "json": json_path}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def run_baselines_only(tasks: List[OracleTask], args) -> str:
    rows = []
    for task in tasks:
        row = {"task": task.name, "family": task.family, "seed": task.seed}
        for base in neural_baselines(task.seed) + library_baselines(task.seed):
            try:
                base.fit(task.X_train, task.y_train, seed=task.seed)
                row[f"r2_{base.name}"] = r2_score(base.predict(task.X_test), task.y_test)
            except Exception:
                row[f"r2_{base.name}"] = float("nan")
            del base
            gc.collect()
        rows.append(row)
        print(f"  [baseline] {task.name:24s} " + "  ".join(
            f"{k[3:]} {v:.3f}" for k, v in row.items() if k.startswith("r2_")))
    names = [k[3:] for k in rows[0] if k.startswith("r2_")] if rows else []
    md = ["# Baseline models (same optimiser/budget protocol)\n",
          "| family | " + " | ".join(names) + " |", "|" + "---|" * (len(names) + 1)]
    for fam in FAMILIES:
        sel = [r for r in rows if r["family"] == fam]
        if not sel:
            continue
        md.append("| " + fam + " | " + " | ".join(
            mean_std([r[f"r2_{n}"] for r in sel if f"r2_{n}" in r and not math.isnan(r[f"r2_{n}"])])
            for n in names) + " |")
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "baselines.md")
    with open(path, "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(args.out, "baselines.json"), "w") as f:
        json.dump(rows, f, indent=2)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="OrthoBias benchmark suite")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--families", nargs="*", default=list(FAMILIES))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--d", type=int, default=10)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr-deep", type=float, default=1e-3)
    ap.add_argument("--calibrate", action="store_true", help="re-solve the projection operator after training")
    ap.add_argument("--n-bins", type=int, default=8)
    ap.add_argument("--trees", type=int, default=24)
    ap.add_argument("--tree-depth", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--res-blocks", type=int, default=3)
    ap.add_argument("--max-pairs", type=int, default=6)
    ap.add_argument("--lam-l0", type=float, default=1e-3)
    ap.add_argument("--lam-mdl", type=float, default=1e-3)
    ap.add_argument("--lam-ortho", type=float, default=1e-2)
    ap.add_argument("--taus", nargs="*", type=float, default=[0.7, 0.9, 0.95])
    ap.add_argument("--no-staging", action="store_true")
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--tau-sweep", action="store_true")
    ap.add_argument("--check-invariance", action="store_true")
    ap.add_argument("--prune-check", action="store_true")
    ap.add_argument("--prune-tol", type=float, default=0.30)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--only", type=str, default="all",
                    help="comma separated subset of {oracle,ablations,tau,prune,invariance}")
    ap.add_argument("--baselines-only", action="store_true",
                    help="run only the baseline models (separate memory budget) and write baselines.md")
    ap.add_argument("--profiler", type=str, default="checkpoints/profiler.pt")
    ap.add_argument("--out", type=str, default="results")
    args = ap.parse_args(argv)

    if args.smoke:
        args.n_train, args.d, args.epochs = 1200, 8, 60
        args.trees, args.hidden, args.res_blocks = 16, 96, 2
        args.seeds = args.seeds[:1]
        args.max_pairs = 4
        args.taus = [0.7, 0.9]
        args.ablations = True
        args.tau_sweep = True
        args.check_invariance = True
        args.prune_check = True
        args.out = "results_smoke"
        print("[smoke] reduced configuration: 1 seed per family, 50 epochs")

    only = set(x.strip() for x in args.only.split(",") if x.strip())
    run_all = "all" in only

    torch.manual_seed(args.seed0 if hasattr(args, "seed0") else 0)
    tasks = []
    for fam in args.families:
        for s in args.seeds:
            tasks.append(make_task(fam, n_train=args.n_train, d=args.d, noise=args.noise,
                                   seed=s, max_pairs=args.max_pairs, n_bins=args.n_bins))
    print(f"{len(tasks)} tasks: {args.families} x seeds {args.seeds}")

    profiler = None
    if os.path.exists(args.profiler):
        try:
            ck = torch.load(args.profiler, map_location="cpu", weights_only=False)
            profiler = MetaProfiler(hidden=ck.get("config", {}).get("hidden", 96))
            profiler.load_state_dict(ck["state_dict"])
            profiler.eval()
            print(f"loaded meta-profiler from {args.profiler}")
        except Exception as exc:
            print(f"could not load profiler ({exc}); continuing without warm start")
    else:
        print(f"no profiler at {args.profiler}; runs without warm start "
              f"(train it with `python3 train_meta.py`)")

    extra: Dict[str, object] = {"config": vars(args).copy()}
    if args.check_invariance and (run_all or "invariance" in only):
        inv = check_invariance(args)
        extra["invariance"] = inv
        print("invariance:", {k: f"{v:.2e}" for k, v in inv.items()})

    if args.prune_check and (run_all or "prune" in only):
        print("\n== exact-zero pruning check ==")
        extra["prune"] = run_prune_check(args)
        p = extra["prune"]
        print(f"  alpha {[round(a, 2) for a in p['alpha']]}")
        print(f"  gates {[round(g, 3) for g in p['gates']]}  pruned blocks {p['pruned']}")
        print(f"  R2 {p['r2_before']:.4f} -> {p['r2_after']:.4f}   IR {p['ir_before']:.3f} -> {p['ir_after']:.3f}")

    if args.baselines_only:
        print("\n== baselines only ==")
        path = run_baselines_only(tasks, args)
        print("written:", path)
        return 0

    t0 = time.time()
    rows: List[Dict] = []
    if run_all or "oracle" in only:
        print("\n== oracle recovery ==")
        rows = run_oracle(tasks, args, profiler)
    abl: List[Dict] = []
    if args.ablations and (run_all or "ablations" in only):
        print("\n== ablations ==")
        abl = run_ablations(tasks, args, profiler)
    tau_rows: List[Dict] = []
    if args.tau_sweep and (run_all or "tau" in only):
        print("\n== tau_interp sweep ==")
        tau_rows = run_tau_sweep(tasks, args, profiler)
    if not rows:
        # keep the report well-formed when only a subset was run
        rows = [{"task": t.name, "family": t.family, "seed": t.seed, "r2": float("nan"),
                 "rho": [0, 0, 0, 0], "ir": 0.0, "gates": [0, 0, 0, 0], "pruned": [False] * 4,
                 "fit_time": 0.0, "rho_star": t.rho_star, "corr_raw_max": 0.0,
                 "corr_exact_max": 0.0, "corr_deploy_max": 0.0, "corr_raw_mean": 0.0,
                 "corr_exact_mean": 0.0, "corr_deploy_mean": 0.0, "n_params": 0,
                 "err_block_l1": 0.0, "err_ident_l1": 0.0, "err_main": 0.0, "err_pair": 0.0,
                 "err_high": 0.0, "attrib_hi_to_h3": 0.0} for t in tasks] if (abl or tau_rows) else []

    paths = write_tables(rows, abl, tau_rows, args, extra,
                         tag="all" if run_all else "_".join(sorted(only)))
    print(f"\ntotal {time.time() - t0:.1f}s")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

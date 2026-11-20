"""Pre-train the amortized meta-profiler.

Each meta-task is a synthetic structured regression problem drawn from the same
prior as the benchmark families but with random size, dimension and noise.  The
supervision comes from the task's *reference-chain* block shares (never from the
profiler's own output), so the profiler learns to predict, from cheap probe
statistics alone, which inductive bias the data rewards.

Loss = CE(rho) + BCE(feature support) + BCE(pair support).

Usage
-----
    python3 train_meta.py --smoke
    python3 train_meta.py --tasks 400 --epochs 30 --out checkpoints/profiler.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.synthetic_oracle import sample_meta_task
from models.profiler import MetaProfiler, compute_signatures


def build_dataset(n_tasks: int, seed0: int = 0, max_pairs: int = 6, n_bins: int = 8,
                  n_mc: int = 600, verbose: bool = True) -> List[Dict]:
    """Compute signatures and meta-targets for ``n_tasks`` synthetic tasks."""
    rows = []
    t0 = time.time()
    for i in range(n_tasks):
        task = sample_meta_task(seed0 + i, max_pairs=max_pairs, n_bins=n_bins, n_mc=n_mc)
        sig = compute_signatures(task.X_train, task.y_train, pairs=task.pairs, max_pairs=max_pairs,
                                 n_bins=n_bins, n_probe=768)
        rho = torch.tensor(task.rho_star, dtype=torch.float32)
        d = task.d
        feat_sup = torch.zeros(d)
        feat_sup[task.feature_support] = 1.0
        pair_sup = torch.zeros(len(task.pairs))
        gen = set(tuple(sorted(p)) for p in task.pair_support)
        for k, p in enumerate(task.pairs):
            if tuple(sorted(p)) in gen:
                pair_sup[k] = 1.0
        rows.append({
            "sig": sig,
            "rho": rho,
            "feat": feat_sup,
            "pair": pair_sup,
            "family": task.family,
            "n": task.X_train.shape[0],
        })
        if verbose and (i + 1) % 25 == 0:
            print(f"  signatures {i + 1}/{n_tasks}  ({time.time() - t0:.1f}s)")
    return rows


def collate(rows: List[Dict]):
    """Pad per-task tokens into a batch (feature/pair tokens vary in length)."""
    from models.profiler import PAIR_DIM, FEATURE_DIM

    max_d = max(r["sig"].feature_tokens.shape[0] for r in rows)
    max_p = max(max(r["sig"].pair_tokens.shape[0] for r in rows), 1)
    f_tok = torch.zeros(len(rows), max_d, FEATURE_DIM)
    p_tok = torch.zeros(len(rows), max_p, PAIR_DIM)
    f_mask = torch.zeros(len(rows), max_d)
    p_mask = torch.zeros(len(rows), max_p)
    glob = torch.stack([r["sig"].global_feats for r in rows])
    rho = torch.stack([r["rho"] for r in rows])
    feat = torch.zeros(len(rows), max_d)
    pair = torch.zeros(len(rows), max_p)
    for i, r in enumerate(rows):
        d = r["sig"].feature_tokens.shape[0]
        f_tok[i, :d] = r["sig"].feature_tokens
        f_mask[i, :d] = 1.0
        feat[i, :d] = r["feat"]
        p = r["sig"].pair_tokens.shape[0]
        if p:
            p_tok[i, :p] = r["sig"].pair_tokens
            p_mask[i, :p] = 1.0
            pair[i, :p] = r["pair"]
    return dict(glob=glob, f_tok=f_tok, p_tok=p_tok, f_mask=f_mask, p_mask=p_mask,
                rho=rho, feat=feat, pair=pair)


def masked_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (bce * mask).sum() / mask.sum().clamp_min(1.0)


def forward_batch(profiler: MetaProfiler, batch: Dict[str, torch.Tensor]):
    """Run the profiler on a padded batch through the same code path as inference."""
    from models.profiler import Signatures

    outs = []
    for i in range(batch["glob"].shape[0]):
        fd = int(batch["f_mask"][i].sum())
        pd_ = int(batch["p_mask"][i].sum())
        sig = Signatures(
            global_feats=batch["glob"][i],
            feature_tokens=batch["f_tok"][i, :fd],
            pair_tokens=batch["p_tok"][i, :pd_] if pd_ else torch.zeros(0, batch["p_tok"].shape[-1]),
            pairs=[],
            n=0,
            d=fd,
        )
        outs.append(profiler(sig))
    rho = torch.stack([o["rho"] for o in outs])
    feats = torch.zeros(batch["feat"].shape)
    pairs = torch.zeros(batch["pair"].shape)
    for i, o in enumerate(outs):
        fd = o["feature_logits"].numel()
        feats[i, :fd] = o["feature_logits"]
        pd_ = o["pair_logits"].numel()
        if pd_:
            pairs[i, :pd_] = o["pair_logits"]
    return rho, feats, pairs


def evaluate(profiler: MetaProfiler, rows: List[Dict]) -> Dict[str, float]:
    profiler.eval()
    batch = collate(rows)
    with torch.no_grad():
        rho, feats, pairs = forward_batch(profiler, batch)
    l1 = (rho - batch["rho"]).abs().mean().item()
    # per-block Pearson correlation between predicted and true shares
    corr = []
    for k in range(rho.shape[1]):
        a, b = rho[:, k], batch["rho"][:, k]
        if a.std() > 1e-6 and b.std() > 1e-6:
            corr.append(float(torch.corrcoef(torch.stack([a, b]))[0, 1]))
        else:
            corr.append(0.0)
    f_acc = (((feats > 0).float() == batch["feat"]) * batch["f_mask"]).sum() / batch["f_mask"].sum()
    p_acc = (((pairs > 0).float() == batch["pair"]) * batch["p_mask"]).sum() / batch["p_mask"].sum().clamp_min(1)
    profiler.train()
    return {
        "rho_l1": l1,
        "rho_corr_block1": corr[0],
        "rho_corr_block2": corr[1],
        "rho_corr_high": corr[2] if len(corr) > 2 else 0.0,
        "feature_screen_acc": float(f_acc),
        "pair_screen_acc": float(p_acc),
    }


def train(rows: List[Dict], epochs: int = 25, lr: float = 2e-3, batch: int = 8,
          hidden: int = 96, seed: int = 0, val_rows: List[Dict] = None,
          verbose: bool = True) -> Tuple[MetaProfiler, Dict[str, float]]:
    torch.manual_seed(seed)
    profiler = MetaProfiler(hidden=hidden)
    opt = torch.optim.AdamW(profiler.parameters(), lr=lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    n = len(rows)
    steps_per_epoch = max(math.ceil(n / batch), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch)
    for ep in range(epochs):
        profiler.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            b = collate([rows[i] for i in idx])
            rho_hat, feats, pairs = forward_batch(profiler, b)
            loss = F.kl_div((rho_hat + 1e-8).log(), b["rho"], reduction="batchmean")
            loss = loss + 0.3 * masked_bce(feats, b["feat"], b["f_mask"])
            loss = loss + 0.3 * masked_bce(pairs, b["pair"], b["p_mask"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(profiler.parameters(), 1.0)
            opt.step()
            try:
                sched.step()
            except Exception:
                pass
            tot += float(loss) * len(idx)
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            msg = f"  epoch {ep + 1:3d} loss {tot / n:.4f}"
            if val_rows:
                m = evaluate(profiler, val_rows)
                msg += f"  val rho-L1 {m['rho_l1']:.4f}  corr(hi) {m['rho_corr_high']:.3f}"
            print(msg)
    metrics = evaluate(profiler, val_rows) if val_rows else {}
    return profiler, metrics


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pre-train the OrthoBias meta-profiler")
    ap.add_argument("--smoke", action="store_true", help="tiny configuration for a correctness check")
    ap.add_argument("--tasks", type=int, default=200)
    ap.add_argument("--val-tasks", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--max-pairs", type=int, default=6)
    ap.add_argument("--n-bins", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="checkpoints/profiler.pt")
    ap.add_argument("--report", type=str, default="results/meta_pretrain.md")
    args = ap.parse_args(argv)

    if args.smoke:
        args.tasks, args.val_tasks, args.epochs = 10, 4, 6
        args.out = args.out if args.out != "checkpoints/profiler.pt" else "checkpoints/profiler_smoke.pt"
        print("[smoke] tiny meta-training run")

    print(f"building {args.tasks} meta-tasks (+{args.val_tasks} validation)")
    rows = build_dataset(args.tasks, seed0=args.seed, max_pairs=args.max_pairs, n_bins=args.n_bins)
    val = build_dataset(args.val_tasks, seed0=10_000 + args.seed, max_pairs=args.max_pairs, n_bins=args.n_bins)

    base = evaluate(MetaProfiler(hidden=args.hidden), val)
    print(f"untrained profiler: rho-L1 {base['rho_l1']:.4f}  feat-acc {base['feature_screen_acc']:.3f}")

    profiler, metrics = train(rows, epochs=args.epochs, lr=args.lr, batch=args.batch,
                              hidden=args.hidden, seed=args.seed, val_rows=val)
    print("final:", json.dumps({k: round(v, 4) for k, v in metrics.items()}))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"state_dict": profiler.state_dict(), "config": {"hidden": args.hidden,
                                                                "max_pairs": args.max_pairs,
                                                                "n_bins": args.n_bins},
                "metrics": metrics, "baseline": base}, args.out)
    print("saved ->", args.out)

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w") as f:
            f.write("# Meta-profiler pre-training\n\n")
            f.write(f"- tasks: {args.tasks} train / {args.val_tasks} val, epochs {args.epochs}\n")
            f.write(f"- checkpoint: `{args.out}`\n\n")
            f.write("| metric | untrained | trained |\n|---|---|---|\n")
            for k in ("rho_l1", "rho_corr_block1", "rho_corr_block2", "rho_corr_high",
                      "feature_screen_acc", "pair_screen_acc"):
                f.write(f"| {k} | {base.get(k, 0):.4f} | {metrics.get(k, 0):.4f} |\n")
        print("report ->", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

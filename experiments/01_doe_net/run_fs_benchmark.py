"""Supervised feature-selection benchmark (CAE / LassoNet / Sequential Attention protocol).

For each dataset x method x seed:
  1. The method ranks all P features using the training split (validation split for
     early stopping / checkpoint selection only).
  2. The top-k features are fed to the same downstream classifier, trained from scratch;
     we report its test accuracy (the protocol used by the feature-selection literature).
  3. For trainable predictors we also report the method's own test accuracy on all features.

Results are appended one JSON line per (dataset, method, seed); finished cells are skipped,
so the script can be resumed.

Usage:
  python run_fs_benchmark.py --data_dir DATA --out results.jsonl \
      --datasets mice,mnist,fashion,isolet,coil20,activity \
      --methods all_features,random,doenet,doenet_noortho,gated_mlp,stg,lassonet,tabnet \
      --k 50 --seeds 0 1 2 3 4
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.doe_net import DOENet
from models.baselines import GatedMLP_Ablation


# ----------------------------------------------------------------------------- utils
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(data_dir, name, seed, val_frac):
    per_seed = os.path.join(data_dir, f"{name}_seed{seed}.npz")  # bench/prepare_sa_data.py
    d = np.load(per_seed if os.path.exists(per_seed) else os.path.join(data_dir, f"{name}.npz"))
    Xtr, ytr, Xte, yte = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
    if "X_val" in d:
        Xva, yva = d["X_val"], d["y_val"]
    else:  # stratified hold-out from the training split, fixed by seed
        rng = np.random.RandomState(seed)
        va_idx = []
        for c in np.unique(ytr):
            idx = np.where(ytr == c)[0]
            rng.shuffle(idx)
            va_idx.append(idx[:max(1, int(round(val_frac * len(idx))))])
        va_idx = np.concatenate(va_idx)
        mask = np.ones(len(ytr), bool)
        mask[va_idx] = False
        Xva, yva = Xtr[~mask], ytr[~mask]
        Xtr, ytr = Xtr[mask], ytr[mask]
    f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)
    i64 = lambda a: np.ascontiguousarray(a, dtype=np.int64)
    return f32(Xtr), i64(ytr), f32(Xva), i64(yva), f32(Xte), i64(yte)


def batches(n, bs, shuffle, rng=None):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


@torch.no_grad()
def predict_logits(model, X, device, bs=1024, gate_model=False):
    model.eval()
    out = []
    for b in batches(len(X), bs, False):
        xb = torch.from_numpy(X[b]).to(device)
        o = model(xb)
        out.append((o[0] if isinstance(o, tuple) else o).float().cpu())
    return torch.cat(out)


def accuracy(logits, y):
    return float((logits.argmax(1).numpy() == y).mean())


# ----------------------------------------------------------------------------- downstream
class OneHiddenMLP(nn.Module):
    """Downstream model of Sequential Attention (SparseModel): input BatchNorm ->
    Dense(67, ReLU) -> Dense(C, softmax). Applied to the k selected columns only
    (equivalent to the reference's BN-then-mask, since BN is per feature)."""
    def __init__(self, d_in, d_hidden, n_cls, input_bn=True):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_in) if input_bn else nn.Identity()
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, n_cls))

    def forward(self, x):
        return self.net(self.bn(x))


# Sequential Attention experiments.py / hyperparams_sa.py (batch size, re-fit epochs)
SA_BATCH = {"mice": 16, "mnist": 349, "fashion": 391, "isolet": 23, "coil20": 451, "activity": 183}
SA_EPOCHS_FIT = {"mice": 250, "mnist": 50, "fashion": 54, "isolet": 80, "coil20": 166, "activity": 134}


def _fit_mlp(Xt, yt, d_in, n_cls, lr, epochs, bs, args, device, seed):
    set_seed(seed)
    m = OneHiddenMLP(d_in, args.ds_hidden, n_cls).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    rng = np.random.RandomState(seed)
    n = len(yt)
    for ep in range(epochs):
        m.train()
        perm = rng.permutation(n)
        for i in range(0, n - bs + 1, bs):  # drop_remainder=True as in the reference
            b = perm[i:i + bs]
            loss = F.cross_entropy(m(Xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m


def downstream_accuracy(Xtr, ytr, Xva, yva, Xte, yte, n_cls, args, device, seed, ds_name=None):
    """Identical downstream protocol for every method. Learning rate picked from
    args.ds_lrs by validation accuracy of the final-epoch model; test accuracy of that
    final-epoch model is reported (no test-time checkpoint selection)."""
    if args.downstream == "mlp":
        bs = SA_BATCH.get(ds_name, args.ds_batch)
        epochs = SA_EPOCHS_FIT.get(ds_name, args.ds_epochs)
        bs = min(bs, len(ytr))
        Xt, yt = torch.from_numpy(Xtr).to(device), torch.from_numpy(ytr).to(device)
        best = (-1.0, None, None)
        for lr in args.ds_lrs:
            m = _fit_mlp(Xt, yt, Xtr.shape[1], n_cls, lr, epochs, bs, args, device, seed)
            va = accuracy(predict_logits(m, Xva, device), yva)
            if va > best[0]:
                te = float("nan") if args.val_only else accuracy(predict_logits(m, Xte, device), yte)
                best = (va, lr, te)
        return best[2], best[0], best[1]
    if args.downstream == "extratrees":
        from sklearn.ensemble import ExtraTreesClassifier
        clf = ExtraTreesClassifier(n_estimators=args.et_trees, random_state=seed, n_jobs=8)
        clf.fit(np.concatenate([Xtr, Xva]), np.concatenate([ytr, yva]))
        return float(clf.score(Xte, yte)), float("nan"), None
    raise ValueError(args.downstream)


# ----------------------------------------------------------------------------- selectors
def temperature(epoch, max_epochs, t_start=0.66, t_min=0.1):
    decay = math.log(t_min / t_start) / max_epochs
    return max(t_min, t_start * math.exp(decay * epoch))


CONFIG = {}       # per-dataset hyper-parameter overrides (--config_json)
PRECOMP = {}      # precomputed selections (--precomputed), key (method, dataset, seed)


def resolve_hp(args, family, ds):
    """Global args + per-dataset overrides for a method family ('doenet' | 'stg')."""
    hp = dict(vars(args))
    hp.update(CONFIG.get(family, {}).get(ds, {}))
    return argparse.Namespace(**hp)


def epochs_for(n_train, batch, base_epochs, min_steps):
    """Pre-specified rule: at least `min_steps` optimizer steps (small datasets)."""
    steps = max(1, n_train // batch)
    return max(base_epochs, math.ceil(min_steps / steps))


def _finite_list(a, nd=5):
    return [round(float(v), nd) if np.isfinite(v) else None for v in np.asarray(a, dtype=float)]


def train_gate_model(model, kind, Xtr, ytr, Xva, yva, hp, epochs, lambda_ortho, device, seed,
                     budget_k=None, trace=None):
    """Train DOENet / GatedMLP with the repo's objective; keep best-val checkpoint.

    budget_k: if given, lambda_sparse is not fixed but a Lagrange multiplier for the
    constraint E[||z||_0] <= budget_k, updated by projected dual ascent after every step:
        lambda <- max(0, lambda + dual_lr * (E[||z||_0] - budget_k) / P).
    Checkpoint = best validation accuracy among epochs whose expected L0 is within
    budget_slack * budget_k (fallback: last epoch)."""
    set_seed(seed)
    lam = 0.0 if budget_k else hp.lambda_sparse
    P = Xtr.shape[1]
    opt = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    rng = np.random.RandomState(seed)
    Xt, yt = torch.from_numpy(Xtr).to(device), torch.from_numpy(ytr).to(device)
    best, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        temp = temperature(ep, epochs)
        for b in batches(len(Xtr), hp.batch_size, True, rng):
            if len(b) < 2:
                continue
            y_pred, _ = model(Xt[b], current_temperature=temp)
            if kind == "doenet":
                ld = model.compute_loss(y_pred, yt[b], lambda_sparse=lam,
                                        lambda_ortho=lambda_ortho)
            else:
                ld = model.compute_loss(y_pred, yt[b], lambda_sparse=lam)
            opt.zero_grad()
            ld["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            if budget_k:
                lam = max(0.0, lam + hp.dual_lr * (float(ld["sparse_loss"]) - budget_k) / P)
        sched.step()
        va = accuracy(predict_logits(model, Xva, device), yva)
        l0 = float(model.screening_gate.get_l0_loss()) if kind == "doenet" else float(model.gate.get_l0_loss())
        if trace is not None:
            trace.append([ep, round(lam, 6), round(l0, 2), round(va, 4)])
        feasible = (not budget_k) or l0 <= hp.budget_slack * budget_k
        if feasible and va > best:
            best, best_state = va, {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is None:  # constraint never met: use the last epoch
        best = va
    else:
        model.load_state_dict(best_state)
    return best


@torch.no_grad()
def doenet_interpretation(m, ranking, device, k=50, n_feat=10, n_grid=21, n_pairs=10):
    """Main-effect shape functions of the top features and the strongest pairwise
    interactions among the top-k features (inputs are min-max scaled to [0, 1])."""
    m.eval()
    blk = m.anova_block
    g_ = m.screening_gate
    s = torch.sigmoid(g_.log_alpha)
    g = torch.clamp(s * (g_.zeta - g_.gamma) + g_.gamma, 0.0, 1.0)
    grid = torch.linspace(0, 1, n_grid, device=device)
    shapes = []
    for i in [int(v) for v in ranking[:n_feat]]:
        h = torch.relu((grid * g[i]).unsqueeze(1) * blk.main_weight[i].unsqueeze(0) + blk.main_bias[i].unsqueeze(0))
        f = (blk.main_out(h) - blk.main_running_mean[i].unsqueeze(0)) * g[i]       # (G, C)
        shapes.append({"feature": i, "gate": round(float(g[i]), 4),
                       "values": np.round(f.cpu().numpy(), 5).tolist()})
    sel = torch.as_tensor([int(v) for v in ranking[:k]], device=device)
    Us = blk.inter_u.weight[:, sel] * g[sel]                                            # (R, k)
    Vs = blk.inter_v.weight[:, sel] * g[sel]
    M = torch.einsum("cr,ri,rj->cij", blk.inter_head.weight, Us, Vs)                   # (C, k, k)
    S = (0.5 * (M + M.transpose(1, 2))).pow(2).sum(0).sqrt()
    iu = torch.triu_indices(len(sel), len(sel), 1, device=device)
    vals = S[iu[0], iu[1]]
    top = torch.topk(vals, min(n_pairs, vals.numel()))
    pairs = [[int(sel[iu[0][t]]), int(sel[iu[1][t]]), round(float(v), 6)]
             for v, t in zip(top.values.tolist(), top.indices.tolist())]
    return {"main_effect_shapes": shapes, "top_interactions": pairs}


def select_doenet(Xtr, ytr, Xva, yva, Xte, yte, n_cls, args, device, seed, ds, variant="full"):
    """variant: 'full' | 'noortho' (W_proj not used, original architecture) |
    'noorthoreg' (W_proj used, lambda_ortho = 0)."""
    hp = resolve_hp(args, "doenet", ds)
    epochs = epochs_for(len(ytr), hp.batch_size, hp.epochs, hp.min_steps)
    lam_o = 0.0 if variant == "noorthoreg" else hp.lambda_ortho
    budget = None
    if variant == "budget":  # tuning-free: target k, repo default lambda_ortho
        budget, lam_o = args.k, args.lambda_ortho
    set_seed(seed)
    m = DOENet(Xtr.shape[1], out_features=n_cls, num_factors=hp.num_factors,
               hidden_dim=hp.hidden_dim, interaction_rank=hp.interaction_rank,
               task="classification", use_ortho_proj=(variant != "noortho")).to(device)
    trace = []
    val = train_gate_model(m, "doenet", Xtr, ytr, Xva, yva, hp, epochs, lam_o, device, seed,
                           budget_k=budget, trace=trace)
    info = m.get_feature_importance(threshold=0.1)
    ranking = np.asarray(info["sorted_indices"])
    own = float("nan") if args.val_only else accuracy(predict_logits(m, Xte, device), yte)
    extra = {"num_active_gates": int(info["num_active"]), "epochs": epochs,
             "hp": {"lambda_sparse": "dual" if budget else hp.lambda_sparse, "lambda_ortho": lam_o,
                    "lr": hp.lr, "batch_size": hp.batch_size,
                    **({"budget_k": budget, "dual_lr": hp.dual_lr, "budget_slack": hp.budget_slack} if budget else {})},
             "trace": trace,
             "scores": _finite_list(info["log_alpha"])}
    X_anova = Xva if args.val_only else Xte
    extra["anova"] = m.get_anova_variance_decomposition(torch.from_numpy(X_anova[:2048]).to(device))
    if args.interpret and variant in ("full", "budget"):
        extra["interpretation"] = doenet_interpretation(m, ranking, device, k=args.k)
    return ranking, own, val, extra


def select_gated_mlp(Xtr, ytr, Xva, yva, Xte, yte, n_cls, args, device, seed, ds):
    hp = resolve_hp(args, "doenet", ds)   # ablation: same sparsity / budget as DOENet
    epochs = epochs_for(len(ytr), hp.batch_size, hp.epochs, hp.min_steps)
    set_seed(seed)
    m = GatedMLP_Ablation(Xtr.shape[1], hidden_dim=hp.hidden_dim, out_features=n_cls,
                          task="classification").to(device)
    val = train_gate_model(m, "gated_mlp", Xtr, ytr, Xva, yva, hp, epochs, 0.0, device, seed)
    la = m.gate.log_alpha.detach().cpu().numpy()
    own = float("nan") if args.val_only else accuracy(predict_logits(m, Xte, device), yte)
    return np.argsort(-la, kind="stable"), own, val, {
        "epochs": epochs, "hp": {"lambda_sparse": hp.lambda_sparse}, "scores": _finite_list(la)}


def select_stg(Xtr, ytr, Xva, yva, Xte, yte, n_cls, args, device, seed, ds):
    import collections
    import collections.abc
    for _n in ["Sequence", "Mapping", "Iterable", "Callable", "MutableMapping", "Set", "Hashable"]:
        setattr(collections, _n, getattr(collections.abc, _n))  # stg 0.1.2 on Python >= 3.10
    from stg import STG
    hp = resolve_hp(args, "stg", ds)
    epochs = epochs_for(len(ytr), 100, hp.stg_epochs, hp.min_steps)
    set_seed(seed)
    # stg 0.1.2 defaults (hidden [400,200], relu, sigma 0.5, batch 100, wd 1e-3) except
    # learning_rate (library default 1e-5) and per-epoch shuffling (default off).
    m = STG(task_type="classification", input_dim=Xtr.shape[1], output_dim=n_cls,
            hidden_dims=list(hp.stg_hidden), activation="relu", optimizer="Adam",
            learning_rate=hp.stg_lr, batch_size=100, feature_selection=True,
            sigma=0.5, lam=hp.stg_lam, random_state=seed, device=device)
    m.fit(Xtr, ytr, nr_epochs=epochs, valid_X=Xva, valid_y=yva, verbose=False,
          print_interval=10 ** 9, shuffle=True)
    mu = np.asarray(m.get_gates(mode="raw")).ravel()
    m._model.to("cpu")  # stg 0.1.2 predict() feeds CPU tensors regardless of device
    m.device = "cpu"
    own = float("nan")
    if not args.val_only:
        own = float((np.asarray(m.predict(Xte, verbose=False)).ravel() == yte).mean())
    return np.argsort(-mu, kind="stable"), own, float("nan"), {
        "epochs": epochs, "hp": {"lam": hp.stg_lam}, "scores": _finite_list(mu)}


def select_lassonet(Xtr, ytr, Xva, yva, Xte, yte, n_cls, args, device, seed, ds):
    from lassonet import LassoNetClassifier
    set_seed(seed)
    m = LassoNetClassifier(hidden_dims=tuple(args.lassonet_hidden), M=args.lassonet_M,
                           random_state=seed, torch_seed=seed, device=device, verbose=0)
    path = m.path(Xtr, ytr, X_val=Xva, y_val=yva, return_state_dicts=False)
    imp = np.asarray(m.feature_importances_.cpu() if torch.is_tensor(m.feature_importances_)
                     else m.feature_importances_, dtype=float)
    return np.argsort(-imp, kind="stable"), float("nan"), float("nan"), {
        "path_len": len(path), "scores": _finite_list(imp)}


def select_tabnet(Xtr, ytr, Xva, yva, Xte, yte, n_cls, args, device, seed, ds):
    from pytorch_tabnet.tab_model import TabNetClassifier
    set_seed(seed)
    m = TabNetClassifier(seed=seed, device_name=device.type, verbose=0)  # library defaults
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric=["accuracy"],
          max_epochs=args.tabnet_epochs, patience=args.tabnet_patience,
          batch_size=args.tabnet_batch, virtual_batch_size=min(128, args.tabnet_batch))
    imp = np.asarray(m.feature_importances_)
    own = float("nan") if args.val_only else float((m.predict(Xte) == yte).mean())
    return np.argsort(-imp, kind="stable"), own, float("nan"), {"scores": _finite_list(imp)}


def select_random(Xtr, *a, seed=0, **k):
    return np.random.RandomState(1000 + seed).permutation(Xtr.shape[1]), float("nan"), float("nan"), {}


def make_precomputed(method):
    """Selections produced outside this script (e.g. official Sequential Attention code)."""
    def _sel(Xtr, *a, seed=0, ds=None, **k):
        rec = PRECOMP[(method, ds, seed)]
        return np.asarray(rec["selected"]), float("nan"), float("nan"), {
            k2: v for k2, v in rec.items() if k2 not in ("method", "dataset", "seed", "selected")}
    return _sel


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", default="mice,mnist,fashion,isolet,coil20,activity")
    ap.add_argument("--methods", default="all_features,random,doenet,doenet_noortho,doenet_noorthoreg,gated_mlp,stg,lassonet,tabnet")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])  # SA experiments.py
    ap.add_argument("--val_frac", type=float, default=0.1)
    # DOENet / GatedMLP (repo defaults, train_benchmark.py)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lambda_sparse", type=float, default=2e-3)
    ap.add_argument("--lambda_ortho", type=float, default=1e-2)
    ap.add_argument("--num_factors", type=int, default=24)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--interaction_rank", type=int, default=16)
    ap.add_argument("--dual_lr", type=float, default=1e-2)      # doenet_budget: dual ascent step on lambda
    ap.add_argument("--budget_slack", type=float, default=1.2)  # doenet_budget: feasible if E[L0] <= slack * k
    ap.add_argument("--min_steps", type=int, default=10000)  # epochs = max(epochs, ceil(min_steps/steps_per_epoch)); DOENet family + STG
    # protocol v2
    ap.add_argument("--config_json", default=None, help='per-dataset overrides {"doenet": {ds: {...}}, "stg": {ds: {...}}}')
    ap.add_argument("--precomputed", default=None, help="JSONL with method, dataset, seed, selected (e.g. official SA code)")
    ap.add_argument("--val_only", action="store_true", help="tuning mode: never evaluate on the test split")
    ap.add_argument("--interpret", action="store_true", help="store DOENet main-effect shapes and top interactions")
    # downstream
    ap.add_argument("--downstream", default="mlp", choices=["mlp", "extratrees"])
    ap.add_argument("--ds_hidden", type=int, default=67)          # SA deep_layers=67
    ap.add_argument("--ds_epochs", type=int, default=100)         # fallback if dataset not in SA_EPOCHS_FIT
    ap.add_argument("--ds_lrs", type=float, nargs="+", default=[1e-3, 3e-3, 1e-2])
    ap.add_argument("--ds_batch", type=int, default=256)          # fallback if dataset not in SA_BATCH
    ap.add_argument("--et_trees", type=int, default=50)
    ap.add_argument("--ds_device", default="cpu")  # tiny MLP: CPU is ~10x faster than the shared GPU
    ap.add_argument("--ds_threads", type=int, default=1)
    # baselines (library / paper defaults)
    ap.add_argument("--stg_hidden", type=int, nargs="+", default=[400, 200])
    ap.add_argument("--stg_lr", type=float, default=1e-3)
    ap.add_argument("--stg_lam", type=float, default=0.1)
    ap.add_argument("--stg_epochs", type=int, default=100)
    ap.add_argument("--lassonet_hidden", type=int, nargs="+", default=[100])
    ap.add_argument("--lassonet_M", type=float, default=10)
    ap.add_argument("--tabnet_epochs", type=int, default=100)
    ap.add_argument("--tabnet_patience", type=int, default=10)
    ap.add_argument("--tabnet_batch", type=int, default=1024)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_device = torch.device(args.ds_device)
    torch.set_num_threads(args.ds_threads)
    if args.config_json:
        CONFIG.update(json.load(open(args.config_json)))
    if args.precomputed:
        for line in open(args.precomputed):
            r = json.loads(line)
            PRECOMP[(r["method"], r["dataset"], int(r["seed"]))] = r
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            done.add((r["dataset"], r["method"], r["seed"]))
    print(f"device={device} downstream_device={ds_device} done_cells={len(done)}", flush=True)
    with open(args.out + ".config.json", "w") as f:
        json.dump({**vars(args), "config": CONFIG}, f, indent=1)

    selectors = {
        "doenet": lambda *a, **k: select_doenet(*a, **k, variant="full"),
        "doenet_noortho": lambda *a, **k: select_doenet(*a, **k, variant="noortho"),
        "doenet_noorthoreg": lambda *a, **k: select_doenet(*a, **k, variant="noorthoreg"),
        "doenet_budget": lambda *a, **k: select_doenet(*a, **k, variant="budget"),
        "gated_mlp": select_gated_mlp,
        "stg": select_stg,
        "lassonet": select_lassonet,
        "tabnet": select_tabnet,
        "random": select_random,
    }
    for (m_, _, _) in PRECOMP:
        selectors.setdefault(m_, make_precomputed(m_))

    for ds in args.datasets.split(","):
        for seed in args.seeds:
            Xtr, ytr, Xva, yva, Xte, yte = load_dataset(args.data_dir, ds, seed, args.val_frac)
            n_cls = int(max(ytr.max(), yva.max(), yte.max()) + 1)
            for method in args.methods.split(","):
                if (ds, method, seed) in done:
                    continue
                t0 = time.time()
                rec = {"dataset": ds, "method": method, "seed": seed, "k": args.k,
                       "n_train": len(ytr), "n_val": len(yva), "n_test": len(yte),
                       "n_features": int(Xtr.shape[1]), "n_classes": n_cls,
                       "downstream": args.downstream, "val_only": bool(args.val_only)}
                try:
                    if method == "all_features":
                        sel, own, val, extra = np.arange(Xtr.shape[1]), float("nan"), float("nan"), {}
                        cols = sel
                    else:
                        sel, own, val, extra = selectors[method](
                            Xtr, ytr, Xva, yva, Xte, yte, n_cls, args, device, seed=seed, ds=ds)
                        cols = np.asarray(sel)[:args.k]
                    t_sel = time.time() - t0
                    acc, ds_val, ds_lr = downstream_accuracy(Xtr[:, cols], ytr, Xva[:, cols], yva,
                                                             Xte[:, cols], yte, n_cls, args, ds_device, seed,
                                                             ds_name=ds)
                    rec.update({"selected": [int(c) for c in cols][:args.k] if method != "all_features" else None,
                                "downstream_test_acc": acc, "downstream_val_acc": ds_val, "downstream_lr": ds_lr,
                                "own_test_acc": own, "own_val_acc": val,
                                "time_select_s": t_sel, "time_total_s": time.time() - t0, **extra})
                except Exception as e:
                    rec.update({"error": repr(e)[:500], "time_total_s": time.time() - t0})
                with open(args.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                msg = f"{ds:9s} {method:15s} seed={seed} " + (
                    f"val@k={rec['downstream_val_acc']:.4f} test@k={rec['downstream_test_acc']:.4f}"
                    if "error" not in rec else f"ERROR {rec['error'][:120]}")
                print(msg + f" ({rec['time_total_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

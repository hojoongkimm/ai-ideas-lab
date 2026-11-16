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
                best = (va, lr, accuracy(predict_logits(m, Xte, device), yte))
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


def train_gate_model(model, kind, Xtr, ytr, Xva, yva, args, device, seed):
    """Train DOENet / GatedMLP with the repo's objective; keep best-val checkpoint."""
    set_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    rng = np.random.RandomState(seed)
    Xt, yt = torch.from_numpy(Xtr).to(device), torch.from_numpy(ytr).to(device)
    best, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        model.train()
        temp = temperature(ep, args.epochs)
        for b in batches(len(Xtr), args.batch_size, True, rng):
            if len(b) < 2:
                continue
            y_pred, _ = model(Xt[b], current_temperature=temp)
            if kind == "doenet":
                ld = model.compute_loss(y_pred, yt[b], lambda_sparse=args.lambda_sparse,
                                        lambda_ortho=args.lambda_ortho)
            else:
                ld = model.compute_loss(y_pred, yt[b], lambda_sparse=args.lambda_sparse)
            opt.zero_grad()
            ld["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        sched.step()
        va = accuracy(predict_logits(model, Xva, device), yva)
        if va > best:
            best, best_state = va, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best


def select_doenet(Xtr, ytr, Xva, yva, Xte, yte, n_cls, args, device, seed, use_ortho):
    set_seed(seed)
    m = DOENet(Xtr.shape[1], out_features=n_cls, num_factors=args.num_factors,
               hidden_dim=args.hidden_dim, interaction_rank=args.interaction_rank,
               task="classification", use_ortho_proj=use_ortho).to(device)
    val = train_gate_model(m, "doenet", Xtr, ytr, Xva, yva, args, device, seed)
    info = m.get_feature_importance(threshold=0.1)
    own = accuracy(predict_logits(m, Xte, device), yte)
    extra = {"num_active_gates": int(info["num_active"])}
    try:
        extra["anova"] = m.get_anova_variance_decomposition(torch.from_numpy(Xte[:2048]).to(device))
    except Exception as e:  # pragma: no cover
        extra["anova_error"] = repr(e)
    return np.asarray(info["sorted_indices"]), own, val, extra


def select_gated_mlp(Xtr, ytr, Xva, yva, Xte, yte, n_cls, args, device, seed):
    set_seed(seed)
    m = GatedMLP_Ablation(Xtr.shape[1], hidden_dim=args.hidden_dim, out_features=n_cls,
                          task="classification").to(device)
    val = train_gate_model(m, "gated_mlp", Xtr, ytr, Xva, yva, args, device, seed)
    la = m.gate.log_alpha.detach().cpu().numpy()
    own = accuracy(predict_logits(m, Xte, device), yte)
    return np.argsort(-la, kind="stable"), own, val, {}


def select_stg(Xtr, ytr, Xva, yva, Xte, yte, n_cls, args, device, seed):
    import collections
    import collections.abc
    for _n in ["Sequence", "Mapping", "Iterable", "Callable", "MutableMapping", "Set", "Hashable"]:
        setattr(collections, _n, getattr(collections.abc, _n))  # stg 0.1.2 on Python >= 3.10
    from stg import STG
    set_seed(seed)
    # stg 0.1.2 defaults (hidden [400,200], relu, sigma 0.5, lam 0.1, batch 100, wd 1e-3)
    # except learning_rate (library default 1e-5) and per-epoch shuffling (default off).
    m = STG(task_type="classification", input_dim=Xtr.shape[1], output_dim=n_cls,
            hidden_dims=list(args.stg_hidden), activation="relu", optimizer="Adam",
            learning_rate=args.stg_lr, batch_size=100, feature_selection=True,
            sigma=0.5, lam=args.stg_lam, random_state=seed, device=device)
    m.fit(Xtr, ytr, nr_epochs=args.stg_epochs, valid_X=Xva, valid_y=yva, verbose=False,
          print_interval=10 ** 9, shuffle=True)
    mu = np.asarray(m.get_gates(mode="raw")).ravel()
    m._model.to("cpu")  # stg 0.1.2 predict() feeds CPU tensors regardless of device
    m.device = "cpu"
    pred = np.asarray(m.predict(Xte, verbose=False)).ravel()
    own = float((pred == yte).mean())
    return np.argsort(-mu, kind="stable"), own, float("nan"), {}


def select_lassonet(Xtr, ytr, Xva, yva, Xte, yte, n_cls, args, device, seed):
    from lassonet import LassoNetClassifier
    set_seed(seed)
    m = LassoNetClassifier(hidden_dims=tuple(args.lassonet_hidden), M=args.lassonet_M,
                           random_state=seed, torch_seed=seed, device=device, verbose=0)
    path = m.path(Xtr, ytr, X_val=Xva, y_val=yva, return_state_dicts=False)
    imp = np.asarray(m.feature_importances_.cpu() if torch.is_tensor(m.feature_importances_)
                     else m.feature_importances_)
    # own accuracy: dense model at the start of the path is not stored; report best-val point
    own = float("nan")
    return np.argsort(-imp, kind="stable"), own, float("nan"), {"path_len": len(path)}


def select_tabnet(Xtr, ytr, Xva, yva, Xte, yte, n_cls, args, device, seed):
    from pytorch_tabnet.tab_model import TabNetClassifier
    set_seed(seed)
    m = TabNetClassifier(seed=seed, device_name=device.type, verbose=0)  # library defaults
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric=["accuracy"],
          max_epochs=args.tabnet_epochs, patience=args.tabnet_patience,
          batch_size=args.tabnet_batch, virtual_batch_size=min(128, args.tabnet_batch))
    imp = np.asarray(m.feature_importances_)
    own = float((m.predict(Xte) == yte).mean())
    return np.argsort(-imp, kind="stable"), own, float("nan"), {}


def select_random(Xtr, *a, seed=0, **k):
    return np.random.RandomState(1000 + seed).permutation(Xtr.shape[1]), float("nan"), float("nan"), {}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", default="mice,mnist,fashion,isolet,coil20,activity")
    ap.add_argument("--methods", default="all_features,random,doenet,doenet_noortho,gated_mlp,stg,lassonet,tabnet")
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
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            done.add((r["dataset"], r["method"], r["seed"]))
    print(f"device={device} downstream_device={ds_device} done_cells={len(done)}", flush=True)
    with open(args.out + ".config.json", "w") as f:
        json.dump(vars(args), f, indent=1)

    selectors = {
        "doenet": lambda *a, **k: select_doenet(*a, **k, use_ortho=True),
        "doenet_noortho": lambda *a, **k: select_doenet(*a, **k, use_ortho=False),
        "gated_mlp": select_gated_mlp,
        "stg": select_stg,
        "lassonet": select_lassonet,
        "tabnet": select_tabnet,
        "random": select_random,
    }

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
                       "downstream": args.downstream}
                try:
                    if method == "all_features":
                        sel, own, val, extra = np.arange(Xtr.shape[1]), float("nan"), float("nan"), {}
                        cols = sel
                    else:
                        sel, own, val, extra = selectors[method](
                            Xtr, ytr, Xva, yva, Xte, yte, n_cls, args, device, seed=seed)
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
                    f"acc@k={rec['downstream_test_acc']:.4f} own={rec['own_test_acc']:.4f}"
                    if "error" not in rec else f"ERROR {rec['error'][:120]}")
                print(msg + f" ({rec['time_total_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

"""Dataset meta-profiler: orthogonal probe signatures + permutation-invariant encoder.

The profiler is the *amortized compiler* of OrthoBias.  For a new dataset it maps
a vector of cheap, model-free statistics ("orthogonal probe signatures") to

* ``rho``            - a predicted block variance share (the warm start of the
                       block gates),
* ``feature_logits`` - a variable-screening prior,
* ``pair_logits``    - a pair-screening prior,
* ``block_logits``   - free-form logits for diagnostics.

Design notes
------------
*Invariance.*  The encoder is a DeepSets-style pooler over per-feature tokens
together with a permutation-equivariant per-token head, so the block-level
prediction is invariant to the ordering of the columns and the screening output
is equivariant.  This is checked in ``run_benchmark.py --check-invariance``.

*Sufficiency is not claimed.*  The signatures are not a sufficient statistic of
the data-generating process; the profiler is an amortized estimator trained by
imitation against the analytic block shares of synthetic tasks (``train_meta.py``)
and is used as a prior, never as a hard constraint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8

# ---------------------------------------------------------------------------
# low level statistics
# ---------------------------------------------------------------------------


def _as_float(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float()


def _safe(v: float, lo: float = -1e3, hi: float = 1e3) -> float:
    if not math.isfinite(v):
        return 0.0
    return float(min(max(v, lo), hi))


def _entropy_bits(idx: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Empirical entropy (nats) of an integer tensor, normalised by log(n_bins)."""
    cnt = torch.bincount(idx.reshape(-1).long(), minlength=n_bins).float()
    p = cnt / cnt.sum().clamp_min(1.0)
    h = -(p * (p + EPS).log()).sum()
    return h / math.log(n_bins)


def _quantile_bins(x: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Quantile bin indices of a 1-D tensor."""
    qs = torch.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    if x.numel() < n_bins * 2:
        return torch.zeros_like(x, dtype=torch.long)
    edges = torch.quantile(x, qs)
    return torch.bucketize(x.contiguous(), edges.contiguous()).clamp(0, n_bins - 1)


@torch.no_grad()
def _ridge_r2(
    A_tr: torch.Tensor,
    y_tr: torch.Tensor,
    A_te: torch.Tensor,
    y_te: torch.Tensor,
    lam: float = 1e-3,
) -> float:
    """Held-out R^2 of a ridge regression (used for the probe signatures)."""
    m = A_tr.shape[1]
    if m == 0:
        return 0.0
    scale = (A_tr * A_tr).mean().clamp_min(EPS)
    G = A_tr.T @ A_tr / A_tr.shape[0] + lam * scale * torch.eye(m)
    b = A_tr.T @ y_tr / A_tr.shape[0]
    try:
        w = torch.linalg.solve(G, b)
    except Exception:  # pragma: no cover
        w = torch.linalg.lstsq(G, b.unsqueeze(-1)).solution.squeeze(-1)
    pred = A_te @ w
    ss_res = ((y_te - pred) ** 2).mean()
    ss_tot = ((y_te - y_te.mean()) ** 2).mean().clamp_min(EPS)
    return _safe(float(1.0 - ss_res / ss_tot), -1.0, 1.0)


@torch.no_grad()
def _knn_r2(X_tr: torch.Tensor, y_tr: torch.Tensor, X_te: torch.Tensor, y_te: torch.Tensor, k: int = 10) -> float:
    """Held-out R^2 of a k-nearest-neighbour smoother (smoothness probe)."""
    n_tr = X_tr.shape[0]
    if n_tr <= k:
        return 0.0
    d2 = torch.cdist(X_te, X_tr) ** 2
    idx = d2.topk(k, dim=1, largest=False).indices
    pred = y_tr[idx].mean(dim=1)
    ss_res = ((y_te - pred) ** 2).mean()
    ss_tot = ((y_te - y_te.mean()) ** 2).mean().clamp_min(EPS)
    return _safe(float(1.0 - ss_res / ss_tot), -1.0, 1.0)


@torch.no_grad()
def _stump_r2(x: torch.Tensor, y: torch.Tensor, n_bins: int = 8) -> float:
    """Best single-threshold (stump) R^2 of one feature: an axis-aligned probe."""
    b = _quantile_bins(x, n_bins)
    best = -1.0
    mu = y.mean()
    for t in range(1, n_bins):
        left = b < t
        n_l = left.sum()
        if int(n_l) < 2 or int((~left).sum()) < 2:
            continue
        ml = y[left].mean()
        mr = y[~left].mean()
        pred = torch.where(left, ml, mr)
        ss_res = ((y - pred) ** 2).mean()
        ss_tot = ((y - mu) ** 2).mean().clamp_min(EPS)
        best = max(best, float(1.0 - ss_res / ss_tot))
    return _safe(best, -1.0, 1.0)


@torch.no_grad()
def _mi_bins(a: torch.Tensor, b: torch.Tensor, n_a: int, n_b: int) -> float:
    """Mutual information (nats) of two integer tensors, normalised by min entropy."""
    idx = a.reshape(-1).long() * n_b + b.reshape(-1).long()
    cnt = torch.bincount(idx, minlength=n_a * n_b).float().reshape(n_a, n_b)
    n = cnt.sum().clamp_min(1.0)
    pxy = cnt / n
    px = pxy.sum(dim=1, keepdim=True)
    py = pxy.sum(dim=0, keepdim=True)
    nz = pxy > 0
    mi = (pxy[nz] * (pxy[nz] / (px @ py)[nz]).log()).sum()
    hx = -(px[px > 0] * px[px > 0].log()).sum()
    hy = -(py[py > 0] * py[py > 0].log()).sum()
    denom = torch.minimum(hx, hy).clamp_min(EPS)
    return _safe(float(mi / denom), 0.0, 1.0)


# ---------------------------------------------------------------------------
# pair selection by conditional mutual-information gain
# ---------------------------------------------------------------------------


@torch.no_grad()
def _additive_residual(X: torch.Tensor, y: torch.Tensor, n_bins: int = 4, lam: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor]:
    """Residual of the best binned additive fit (a model-free additive probe)."""
    d = X.shape[1]
    B = torch.stack([_quantile_bins(X[:, j], n_bins) for j in range(d)], dim=1)
    A = F.one_hot(B.clamp(0, n_bins - 1), n_bins).float().reshape(X.shape[0], -1)
    A = torch.cat([A, torch.ones(X.shape[0], 1)], dim=1)
    scale = (A * A).mean().clamp_min(EPS)
    G = A.T @ A / A.shape[0] + lam * scale * torch.eye(A.shape[1])
    beta = torch.linalg.solve(G, A.T @ y / A.shape[0])
    return y - A @ beta, B


@torch.no_grad()
def _pair_interaction_energy(
    r: torch.Tensor, B: torch.Tensor, j: int, k: int, n_bins: int = 4
) -> float:
    """Variance of the double-centred cell means of the residual.

    ``r`` are the residuals of an additive binned fit and ``B`` the bin codes, so
    this quantity is a *pure* second-order interaction energy: it is zero when the
    residual is additive and grows with the strength of the ``(j, k)``
    interaction.  It is the basis of both the pair selection rule and the
    ``interaction_energy`` signature.
    """
    n_bins = int(B.max().item()) + 1 if B.numel() else n_bins
    idx = B[:, j].clamp(0, n_bins - 1) * n_bins + B[:, k].clamp(0, n_bins - 1)
    s = torch.zeros(n_bins * n_bins, device=r.device, dtype=r.dtype)
    c = torch.zeros(n_bins * n_bins, device=r.device, dtype=r.dtype)
    s.scatter_add_(0, idx, r)
    c.scatter_add_(0, idx, torch.ones_like(r))
    T = (s / c.clamp_min(1.0)).reshape(n_bins, n_bins)
    T = T - T.mean(dim=1, keepdim=True) - T.mean(dim=0, keepdim=True) + T.mean()
    return _safe(float((T ** 2).mean()), 0.0, 1e6)


@torch.no_grad()
def select_pairs(
    X: torch.Tensor,
    y: torch.Tensor,
    max_pairs: int = 8,
    n_bins: int = 4,
    max_rows: int = 1024,
    seed: int = 0,
    include_all: bool = False,
) -> List[Tuple[int, int]]:
    """Pick candidate interaction pairs by residual interaction energy.

    An additive binned model is fitted once, and every pair is scored by the
    variance of the double-centred cell means of its residuals.  The rule is
    deterministic (seeded subsampling), model-free and shared by all model
    variants, so it cannot advantage one block family over another.
    """
    X = _as_float(X)
    y = _as_float(y).reshape(-1)
    g = torch.Generator().manual_seed(seed)
    if X.shape[0] > max_rows:
        sel = torch.randperm(X.shape[0], generator=g)[:max_rows]
        X, y = X[sel], y[sel]
    d = X.shape[1]
    all_pairs = [(j, k) for j in range(d) for k in range(j + 1, d)]
    if include_all or len(all_pairs) <= max_pairs:
        return all_pairs
    r, B = _additive_residual(X, (y - y.mean()) / y.std().clamp_min(1e-6), n_bins=n_bins)
    scores = [(_pair_interaction_energy(r, B, j, k, n_bins), (j, k)) for (j, k) in all_pairs]
    scores.sort(key=lambda t: -t[0])
    return [p for _, p in scores[:max_pairs]]


# ---------------------------------------------------------------------------
# signatures
# ---------------------------------------------------------------------------


@dataclass
class Signatures:
    """Container of the probe signatures of one dataset."""

    global_feats: torch.Tensor                     # (G,)
    feature_tokens: torch.Tensor                   # (d, F)
    pair_tokens: torch.Tensor                      # (K, P)
    pairs: List[Tuple[int, int]]
    n: int
    d: int
    names: Dict[str, int] = field(default_factory=dict)

    def to(self, device: torch.device) -> "Signatures":
        return Signatures(
            self.global_feats.to(device),
            self.feature_tokens.to(device),
            self.pair_tokens.to(device),
            self.pairs,
            self.n,
            self.d,
            self.names,
        )


GLOBAL_DIM = 17
FEATURE_DIM = 15
PAIR_DIM = 6


@torch.no_grad()
def compute_signatures(
    X: torch.Tensor,
    y: torch.Tensor,
    pairs: Optional[Sequence[Tuple[int, int]]] = None,
    n_bins: int = 8,
    max_pairs: int = 8,
    n_probe: int = 1024,
    knn: int = 10,
    seed: int = 0,
    y_bins: int = 8,
) -> Signatures:
    """Compute the orthogonal probe signatures of a tabular regression task.

    The probes are deliberately *orthogonal to the model family*: a linear probe,
    a stump probe (axis-aligned, depth 1), a binned additive probe, a pairwise
    tensor probe, and a kNN probe.  Their gaps identify which inductive bias the
    data actually rewards, without training any of the blocks.
    """
    X = _as_float(X)
    y = _as_float(y).reshape(-1)
    n, d = X.shape
    g = torch.Generator().manual_seed(seed)

    # deterministic subsample for the probes
    if n > n_probe:
        sel = torch.randperm(n, generator=g)[:n_probe]
        Xp, yp = X[sel], y[sel]
    else:
        Xp, yp = X, y
    np_ = Xp.shape[0]
    perm = torch.randperm(np_, generator=g)
    n_tr = max(int(0.7 * np_), 8)
    tr, te = perm[:n_tr], perm[n_tr:]
    if len(te) < 4:
        te = tr
    Xtr, ytr, Xte, yte = Xp[tr], yp[tr], Xp[te], yp[te]

    # ---- standardisation of the target for stable probes --------------------
    y_mu, y_sd = yp.mean(), yp.std().clamp_min(1e-6)
    yz = (yp - y_mu) / y_sd
    yz_tr, yz_te = yz[tr], yz[te]

    # ---- binned designs -----------------------------------------------------
    def bins_of(x: torch.Tensor) -> torch.Tensor:
        return torch.stack([_quantile_bins(x[:, j], n_bins) for j in range(d)], dim=1)

    B_all = bins_of(Xp)
    B_tr, B_te = B_all[tr], B_all[te]
    onehot = F.one_hot(B_all.clamp(0, n_bins - 1), n_bins).float()     # (n, d, M)
    A_add_all = onehot.reshape(np_, -1)
    A_add_tr, A_add_te = A_add_all[tr], A_add_all[te]

    r2_add = _ridge_r2(A_add_tr, yz_tr, A_add_te, yz_te)

    # ---- pair designs and interaction energies ------------------------------
    if pairs is None:
        pairs = select_pairs(Xp, yp, max_pairs=max_pairs, n_bins=4, seed=seed)
    pairs = list(pairs)
    # The pair probe uses the three strongest candidate pairs only: with ``d*M``
    # additive columns plus ``K*M*M`` tensor columns the design would otherwise be
    # wider than the probe sample and the held-out R^2 would measure variance, not
    # structure.
    probe_pairs = pairs[: min(3, len(pairs))]
    pair_cols_all = []
    for (j, k) in probe_pairs:
        pair_cols_all.append(torch.einsum("na,nb->nab", onehot[:, j], onehot[:, k]).reshape(np_, -1))
    A_pair = torch.cat([A_add_all] + pair_cols_all, dim=1) if pair_cols_all else A_add_all
    r2_pair = _ridge_r2(A_pair[tr], yz_tr, A_pair[te], yz_te)

    # Residual of an additive binned fit; the double-centred interaction energy of
    # these residuals is the cleanest model-free signal that a pair carries
    # structure that additivity cannot express.
    p_bins = 4
    n_add = min(np_, 2048)
    r_add, B4 = _additive_residual(Xp[:n_add], yz[:n_add], n_bins=p_bins)
    var_r = float(r_add.var().clamp_min(EPS))
    energies = [_pair_interaction_energy(r_add, B4, j, k, p_bins) for (j, k) in pairs]
    energies_norm = [e / var_r for e in energies]

    # ---- smooth (kNN) and stub probes --------------------------------------
    n_knn = min(512, len(tr))
    r2_knn = _knn_r2(Xtr[:n_knn], yz_tr[:n_knn], Xte[:256], yz_te[:256], k=knn)
    stump = torch.tensor([_stump_r2(Xp[:, j], yz, n_bins) for j in range(d)])
    lin_r2 = torch.tensor(
        [_safe(float(torch.corrcoef(torch.stack([Xp[:, j], yp]))[0, 1] ** 2), 0.0, 1.0) for j in range(d)]
    )

    # ---- per-feature statistics --------------------------------------------
    mu = Xp.mean(dim=0)
    sd = Xp.std(dim=0).clamp_min(1e-6)
    z = (Xp - mu) / sd
    skew = (z ** 3).mean(dim=0)
    kurt = (z ** 4).mean(dim=0) - 3.0
    uniq = torch.tensor([float(Xp[:, j].unique().numel()) / np_ for j in range(d)])
    ent = torch.stack([_entropy_bits(B_all[:, j], n_bins) for j in range(d)])
    corr = torch.corrcoef(Xp.T) if d > 1 else torch.ones(1, 1)
    corr = torch.nan_to_num(corr, nan=0.0)
    abs_corr_off = corr.abs() - torch.eye(d)
    mean_abs_corr = abs_corr_off.sum(dim=1) / max(d - 1, 1)
    max_abs_corr = abs_corr_off.max(dim=1).values
    # univariate binned R^2 per feature
    uni_r2 = torch.tensor(
        [
            _ridge_r2(
                F.one_hot(B_tr[:, j], n_bins).float(),
                yz_tr,
                F.one_hot(B_te[:, j], n_bins).float(),
                yz_te,
            )
            for j in range(d)
        ]
    )
    mi_y = torch.tensor([_mi_bins(B_all[:, j], _quantile_bins(yz, y_bins), n_bins, y_bins) for j in range(d)])

    # curvature of the conditional-mean profile (how far the shape is from linear)
    curv = []
    for j in range(d):
        prof = torch.zeros(n_bins)
        cnt = torch.zeros(n_bins)
        prof.scatter_add_(0, B_all[:, j].clamp(0, n_bins - 1), yz)
        cnt.scatter_add_(0, B_all[:, j].clamp(0, n_bins - 1), torch.ones_like(yz))
        prof = prof / cnt.clamp_min(1.0)
        if n_bins >= 3:
            curv.append(float(((prof[1:] - prof[:-1]) ** 2).mean()))
        else:
            curv.append(0.0)
    curv = torch.tensor(curv)

    missing = torch.tensor([float(torch.isnan(Xp[:, j]).float().mean()) for j in range(d)])

    # pair salience per feature (aggregated over the candidate pairs)
    salience = torch.zeros(d)
    for idx_p, (j, k) in enumerate(pairs):
        salience[j] += energies_norm[idx_p]
        salience[k] += energies_norm[idx_p]
    if pairs:
        salience = salience / len(pairs)

    feature_tokens = torch.stack(
        [
            torch.tanh(mu),
            torch.log1p(sd),
            skew.clamp(-5, 5),
            kurt.clamp(-5, 20),
            uniq,
            ent,
            lin_r2,
            stump,
            uni_r2.clamp(-1, 1),
            mi_y,
            mean_abs_corr,
            max_abs_corr,
            salience,
            torch.log1p(curv),
            missing,
        ],
        dim=1,
    )                                                       # (d, 15)
    assert feature_tokens.shape[1] == FEATURE_DIM, feature_tokens.shape

    # ---- pair tokens --------------------------------------------------------
    ptok = []
    bz = _quantile_bins(yz, y_bins)
    for idx_p, (j, k) in enumerate(pairs):
        joint = B_all[:, j] * n_bins + B_all[:, k]
        mi_jk = _mi_bins(joint, bz, n_bins * n_bins, y_bins)
        onehot_j_tr = F.one_hot(B_tr[:, j], n_bins).float()
        onehot_k_tr = F.one_hot(B_tr[:, k], n_bins).float()
        onehot_j_te = F.one_hot(B_te[:, j], n_bins).float()
        onehot_k_te = F.one_hot(B_te[:, k], n_bins).float()
        dims = (onehot_j_tr.shape[1], onehot_k_tr.shape[1])
        tensor_tr = torch.einsum("na,nb->nab", onehot_j_tr, onehot_k_tr).reshape(len(tr), -1)
        tensor_te = torch.einsum("na,nb->nab", onehot_j_te, onehot_k_te).reshape(len(te), -1)
        p_add = _ridge_r2(
            torch.cat([onehot_j_tr, onehot_k_tr], 1), yz_tr,
            torch.cat([onehot_j_te, onehot_k_te], 1), yz_te,
        )
        p_pair = _ridge_r2(
            torch.cat([onehot_j_tr, onehot_k_tr, tensor_tr], 1), yz_tr,
            torch.cat([onehot_j_te, onehot_k_te, tensor_te], 1), yz_te,
        )
        ptok.append(
            [
                mi_jk,                                        # dependence with y
                energies_norm[idx_p],                         # residual interaction energy
                math.log1p(energies[idx_p]),                  # log energy
                max(p_pair - p_add, 0.0),                     # held-out R^2 gain of the pair
                max(float(uni_r2[j]) + float(uni_r2[k]), 0.0),  # marginal strength
                abs(float(corr[j, k])),                       # collinearity of the pair
            ]
        )
    pair_tokens = torch.tensor(ptok, dtype=torch.float32).reshape(len(pairs), PAIR_DIM) if ptok else torch.zeros(0, PAIR_DIM)

    # ---- global statistics --------------------------------------------------
    y_hist_bins = 32
    y_ent = float(_entropy_bits(_quantile_bins(yz, y_hist_bins), y_hist_bins))
    y_skew = _safe(float(((yz - yz.mean()) / yz.std().clamp_min(1e-6)).pow(3).mean()), -10, 10)
    y_kurt = _safe(float((((yz - yz.mean()) / yz.std().clamp_min(1e-6)).pow(4)).mean() - 3.0), -5, 50)
    evals = torch.linalg.eigvalsh(corr.double()).float().clamp_min(0.0)
    eff_rank = float((evals.sum() ** 2) / (evals.pow(2).sum().clamp_min(EPS)) / d)
    glob = torch.tensor(
        [
            math.log10(max(n, 2)),
            math.log10(max(d, 2)),
            _safe(math.log10(max(d, 1) / max(n, 1)), -3, 1),
            float(mean_abs_corr.mean()),
            float(max_abs_corr.max()) if d > 1 else 0.0,
            eff_rank,
            y_skew,
            y_kurt,
            y_ent,
            _safe(math.log1p(abs(float(y_mu)) / float(y_sd)), 0, 10),
            float(missing.mean()) if missing.numel() else 0.0,
            r2_add,
            r2_pair,
            r2_knn,
            max(r2_knn - r2_add, 0.0),
            max(r2_pair - r2_add, 0.0),
            float(lin_r2.max()) if d > 0 else 0.0,
        ],
        dtype=torch.float32,
    )
    assert glob.numel() == GLOBAL_DIM, glob.numel()

    names = {
        "log10_n": 0,
        "log10_d": 1,
        "log10_d_over_n": 2,
        "mean_abs_corr": 3,
        "max_abs_corr": 4,
        "eff_rank_ratio": 5,
        "y_skew": 6,
        "y_kurt": 7,
        "y_entropy": 8,
        "y_scale_ratio": 9,
        "missing_rate": 10,
        "probe_additive_r2": 11,
        "probe_pair_r2": 12,
        "probe_knn_r2": 13,
        "probe_smooth_gap": 14,
        "probe_pair_gap": 15,
        "probe_linear_max": 16,
    }
    return Signatures(glob, feature_tokens, pair_tokens, pairs, n, d, names)


# ---------------------------------------------------------------------------
# meta-profiler network
# ---------------------------------------------------------------------------


class MetaProfiler(nn.Module):
    """Permutation-invariant amortized compiler.

    ``feature_logits`` is equivariant (one logit per feature token), ``rho`` and
    ``block_logits`` are invariant (pooled), and ``pair_logits`` is equivariant
    over the candidate pairs.
    """

    def __init__(
        self,
        global_dim: int = GLOBAL_DIM,
        feature_dim: int = FEATURE_DIM,
        pair_dim: int = PAIR_DIM,
        hidden: int = 96,
        n_blocks: int = 4,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.feat_enc = nn.Sequential(nn.Linear(feature_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.pair_enc = nn.Sequential(nn.Linear(pair_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.glob_enc = nn.Sequential(nn.Linear(global_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.trunk = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.block_head = nn.Linear(hidden, n_blocks)
        self.rho_head = nn.Linear(hidden, n_blocks)
        self.feat_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.pair_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, sig: Signatures) -> Dict[str, torch.Tensor]:
        tf = self.feat_enc(sig.feature_tokens)                       # (d, H)
        pooled_f = torch.cat([tf.mean(dim=0), tf.max(dim=0).values])
        if sig.pair_tokens.numel() > 0:
            tp = self.pair_enc(sig.pair_tokens)
            pooled_p = torch.cat([tp.mean(dim=0), tp.max(dim=0).values])
        else:
            tp = torch.zeros(0, tf.shape[1])
            pooled_p = torch.zeros_like(pooled_f)
        g = self.glob_enc(sig.global_feats)
        h = self.trunk(torch.cat([pooled_f + pooled_p, g, tf.mean(dim=0) * g], dim=-1))
        ctx = h.detach()
        feat_logits = self.feat_head(torch.cat([tf, ctx.expand(tf.shape[0], -1)], dim=1)).squeeze(-1)
        if tp.shape[0] > 0:
            pair_logits = self.pair_head(torch.cat([tp, ctx.expand(tp.shape[0], -1)], dim=1)).squeeze(-1)
        else:
            pair_logits = torch.zeros(0)
        return {
            "block_logits": self.block_head(h),
            "rho_logits": self.rho_head(h),
            "rho": torch.softmax(self.rho_head(h), dim=-1),
            "feature_logits": feat_logits,
            "pair_logits": pair_logits,
            "embedding": h,
        }

    # -- warm start -----------------------------------------------------------
    @torch.no_grad()
    def warm_start(
        self,
        sig: Signatures,
        model,
        alpha_scale: float = 2.0,
        alpha_clip: Tuple[float, float] = (-1.0, 3.0),
        alpha_offset: float = 1.5,
        screen_strength: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Initialise the gates and screening priors of an ``OrthoBias`` model.

        The predicted share is mapped to the gate parameter through
        ``alpha_k = logit(clip(alpha_scale * rho_k)) + alpha_offset``, clipped to
        ``alpha_clip``.  With a uniform prediction (``rho_k = 1/4``) this returns
        ``alpha = alpha_offset``, i.e. exactly the default open gate, so an
        uninformed profiler is a no-op rather than a silent prior that closes
        every block.  The warm start is a *prior*, never a hard mask: a
        misprediction can still be corrected by the task loss.
        """
        out = self.forward(sig)
        rho = out["rho"]
        p = (alpha_scale * rho).clamp(1e-3, 0.95)
        alpha = torch.log(p / (1 - p)) + alpha_offset
        alpha = alpha.clamp(*alpha_clip)
        model.gates.alpha.data.copy_(alpha)
        if sig.feature_tokens.shape[0] == model.n_features:
            prob = torch.sigmoid(out["feature_logits"])
            # a screening prior only acts when the profiler is *confident* that a
            # feature is useless (prob < 0.5); at prob = 0.5 it is a no-op
            screen = (1.0 - screen_strength * (1.0 - 2.0 * prob).clamp_min(0.0)).clamp(0.0, 1.0)
            model.screening = screen.detach()
        if out["pair_logits"].numel() == len(model.h2.pairs):
            prob = torch.sigmoid(out["pair_logits"])
            model.pair_screening = (1.0 - screen_strength * (1.0 - 2.0 * prob).clamp_min(0.0)).clamp(0.0, 1.0).detach()
        return out

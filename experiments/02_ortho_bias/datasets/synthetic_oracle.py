"""Synthetic structured tasks with *reference-chain* ground-truth variance shares.

Why a reference chain
---------------------
The Hoeffding-Sobol decomposition of a function with independent inputs is unique,
so the shares of the order-1 and order-2 components are properties of the data
generating process itself.  The split of the *remaining* variance between an
axis-aligned piecewise-constant class (H3) and a smooth class (H4) is **not** a
property of the data: it depends on which function classes are compared.  We
therefore define the ground truth relative to an explicit nested chain

    W1 < W2 < W3

(W1 = binned additive, W2 = + pair tensors for the selected pairs, W3 = + the
tree-class fit of the remainder) and report

    rho_k* = || P_{W_k} f - P_{W_{k-1}} f ||^2 / ||f||^2 ,   k = 1, 2, 3
    rho_4* = remainder,

together with the Sobol shares (``sobol_main``, ``sobol_pair``).  The nested
increments are orthogonal by construction, so the four numbers sum to one exactly.
This makes the recovery target well defined and keeps the claim honest: rho_1 and
rho_2 identify Sobol indices; rho_3 + rho_4 identify the total higher-order share;
the internal split is identified relative to the chain only.

Families
--------
``main``      additive non-linear main effects
``pairwise``  quadratic/interaction structure plus weak main effects
``tree``      conjunctions of 2-3 thresholded features (axis-aligned, discontinuous)
``deep``      smooth dense ridge functions (no axis alignment)
``hybrid``    a mixture of all four
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from models.profiler import select_pairs

EPS = 1e-8
FAMILIES = ("main", "pairwise", "tree", "deep", "hybrid")


# ---------------------------------------------------------------------------
# component shapes
# ---------------------------------------------------------------------------


def _shape(rng: torch.Generator, kind: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """A scalar non-linear shape function of one feature."""
    if kind == 0:
        w = 1.0 + 2.0 * torch.rand((), generator=rng).item()
        return lambda x: torch.sin(w * x)
    if kind == 1:
        return lambda x: torch.tanh(2.5 * x)
    if kind == 2:
        return lambda x: x.pow(2) - 1.0
    if kind == 3:
        return lambda x: x.abs() - 0.8
    if kind == 4:
        t = 0.5 * torch.randn((), generator=rng).item()
        return lambda x: torch.sigmoid(3.0 * (x - t)) - 0.5
    return lambda x: torch.clamp(x, -1.5, 1.5)


def _pair_shape(rng: torch.Generator, kind: int) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if kind == 0:
        return lambda a, b: a * b
    if kind == 1:
        return lambda a, b: torch.tanh(2.0 * a) * torch.tanh(2.0 * b)
    if kind == 2:
        return lambda a, b: torch.relu(a) * torch.relu(b)
    if kind == 3:
        return lambda a, b: torch.sin(a + b)
    return lambda a, b: (a > 0).float() * (b > 0).float() - 0.25


def _design_matrix(
    rng: torch.Generator, d: int, kind: str, n_active: int
) -> Tuple[List[Callable], List[int], List[Tuple[int, int]]]:
    """Return (component callables, active feature list, active pair list)."""
    feats = torch.randperm(d, generator=rng)[:n_active].tolist()
    comps: List[Callable] = []
    pairs: List[Tuple[int, int]] = []
    if kind == "main":
        for j in feats:
            c = _shape(rng, int(torch.randint(0, 6, (1,), generator=rng)))
            comps.append(lambda x, j=j, c=c: c(x[:, j]))
    elif kind == "pairwise":
        for j in feats[: max(2, n_active // 3)]:
            c = _shape(rng, int(torch.randint(0, 6, (1,), generator=rng)))
            comps.append(lambda x, j=j, c=c: 0.35 * c(x[:, j]))
        for k in range(0, len(feats) - 1, 2):
            j1, j2 = feats[k], feats[k + 1]
            g = _pair_shape(rng, int(torch.randint(0, 5, (1,), generator=rng)))
            comps.append(lambda x, j1=j1, j2=j2, g=g: g(x[:, j1], x[:, j2]))
            pairs.append((j1, j2))
    elif kind == "tree":
        # Parity of multi-step (axis-aligned, discontinuous) univariate functions.
        # A parity structure has (almost) no main and no pairwise Sobol share, so
        # the whole signal sits in the higher-order components while remaining
        # expressible by axis-aligned partitions: the controlled contrast to the
        # smooth high-order ``deep`` family below.
        if n_active >= 3:
            js = feats[:3]
            a = 0.9 + 0.5 * float(torch.rand((), generator=rng))
            comps.append(
                lambda x, js=js, a=a: torch.sign(torch.sin(a * x[:, js[0]]))
                * torch.sign(torch.sin(a * x[:, js[1]]))
                * torch.sign(torch.sin(a * x[:, js[2]]))
            )
        for j in feats[3:4]:
            c = _shape(rng, int(torch.randint(0, 6, (1,), generator=rng)))
            comps.append(lambda x, j=j, c=c: 0.15 * c(x[:, j]))
    elif kind == "deep":
        # Products of dense (non-axis-aligned) ridge functions: smooth, genuinely
        # high-order, and with an almost vanishing additive projection.
        W1 = torch.randn(d, 1, generator=rng) / math.sqrt(d)
        W2 = torch.randn(d, 1, generator=rng) / math.sqrt(d)
        comps.append(lambda x, W1=W1, W2=W2: torch.sin(1.1 * (x @ W1)).squeeze(-1)
                     * torch.sin(1.1 * (x @ W2)).squeeze(-1))
    else:  # hybrid
        for j in feats[: max(2, n_active // 2)]:
            c = _shape(rng, int(torch.randint(0, 6, (1,), generator=rng)))
            comps.append(lambda x, j=j, c=c: 0.7 * c(x[:, j]))
        if len(feats) >= 4:
            j1, j2 = feats[0], feats[1]
            g = _pair_shape(rng, int(torch.randint(0, 5, (1,), generator=rng)))
            comps.append(lambda x, j1=j1, j2=j2, g=g: 0.8 * g(x[:, j1], x[:, j2]))
            pairs.append((j1, j2))
        js = feats[2:5] if len(feats) >= 5 else feats[:3]
        ts = [0.5 * torch.randn((), generator=rng).item() for _ in js]
        comps.append(
            lambda x, js=js, ts=ts: torch.stack([(x[:, j] > t).float() for j, t in zip(js, ts)], 1).prod(1)
        )
        W = torch.randn(d, 4, generator=rng)
        comps.append(lambda x, W=W: 0.5 * torch.tanh(x @ W) @ torch.ones(W.shape[1]))
    return comps, feats, pairs


# ---------------------------------------------------------------------------
# reference tree candidates (used to define W3)
# ---------------------------------------------------------------------------


def _cart_leaf_indicators(
    B: torch.Tensor, r: torch.Tensor, depth: int = 3, min_leaf: int = 12, max_leaf_keep: int = 64
) -> torch.Tensor:
    """Leaf indicators of a least-squares CART fit of ``r`` on the binned design.

    A greedy depth-``depth`` tree on the bin grid is the natural representative of
    the axis-aligned piecewise-constant class; using it (instead of random trees)
    to define the reference chain keeps the rho_3 target attainable.
    """
    n, d = B.shape
    n_bins = int(B.max().item()) + 1
    rows = [torch.arange(n)]

    def split(idx: torch.Tensor, level: int):
        if level >= depth or len(idx) < 2 * min_leaf or d == 0:
            return ("leaf", idx)
        base = ((r[idx] - r[idx].mean()) ** 2).sum()
        best = (base, None)
        for j in range(d):
            col = B[idx, j]
            for t in range(n_bins - 1):
                m = col > t
                nl = int(m.sum())
                if nl < min_leaf or len(idx) - nl < min_leaf:
                    continue
                sl = ((r[idx][m] - r[idx][m].mean()) ** 2).sum()
                sr = ((r[idx][~m] - r[idx][~m].mean()) ** 2).sum()
                if sl + sr < best[0] - 1e-12:
                    best = (sl + sr, (j, t))
        if best[1] is None:
            return ("leaf", idx)
        j, t = best[1]
        m = B[idx, j] > t
        return ("node", j, t, split(idx[m], level + 1), split(idx[~m], level + 1))

    tree = split(rows[0], 0)
    leaves = []

    def collect(node):
        if node[0] == "leaf":
            idx = node[1]
            if len(idx) > 0:
                col = torch.zeros(n, dtype=torch.float32)
                col[idx] = 1.0
                leaves.append(col)
        else:
            collect(node[3])
            collect(node[4])

    collect(tree)
    if not leaves:
        return torch.zeros(n, 0)
    return torch.stack(leaves[:max_leaf_keep], dim=1)


def _project(residual: torch.Tensor, phi: torch.Tensor, ridge: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Least-squares projection of ``residual`` on ``span(phi)``; returns (fit, residual)."""
    if phi.shape[1] == 0:
        return torch.zeros_like(residual), residual
    x = phi.double()
    y = residual.double()
    d = (x * x).sum(dim=0).clamp_min(1e-12).sqrt()
    xs = x / d
    g = xs.T @ xs / x.shape[0]
    r = xs.T @ y / x.shape[0]
    a = g + ridge * torch.eye(g.shape[0], dtype=g.dtype)
    try:
        w = torch.linalg.solve(a, r)
    except Exception:  # pragma: no cover
        w = torch.linalg.lstsq(a, r.unsqueeze(-1)).solution.squeeze(-1)
    fit = (xs @ w).float()
    return fit, residual - fit


# ---------------------------------------------------------------------------
# task container
# ---------------------------------------------------------------------------


@dataclass
class OracleTask:
    name: str
    family: str
    seed: int
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    f_test: torch.Tensor
    rho_star: List[float]
    sobol_main: float
    sobol_pair: float
    sigma2: float
    pairs: List[Tuple[int, int]]
    feature_support: List[int]
    pair_support: List[Tuple[int, int]]
    meta: Dict[str, float] = field(default_factory=dict)

    @property
    def d(self) -> int:
        return self.X_train.shape[1]


# ---------------------------------------------------------------------------
# reference shares
# ---------------------------------------------------------------------------


@torch.no_grad()
def reference_shares(
    f: Callable[[torch.Tensor], torch.Tensor],
    d: int,
    pairs: Sequence[Tuple[int, int]],
    n_mc: int = 2000,
    n_bins: int = 8,
    tree_depth: int = 3,
    n_cand: int = 192,
    n_leaf_keep: int = 48,
    seed: int = 0,
) -> Dict[str, object]:
    """Decompose ``f`` along the reference chain W1 < W2 < W3."""
    rng = torch.Generator().manual_seed(seed)
    X = torch.randn(n_mc, d, generator=rng) * 1.0 + 0.0
    # the generator's inputs are standard normal in every family, so the
    # reference sample uses the same distribution
    y = f(X)
    y = (y - y.mean()) / y.std().clamp_min(1e-6)
    total = float((y ** 2).mean())

    # quantile bins as in the model
    qs = torch.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = torch.quantile(X, qs, dim=0)
    B = torch.stack([torch.bucketize(X[:, j].contiguous(), edges[:, j].contiguous()) for j in range(d)], 1)
    B = B.clamp(0, n_bins - 1)
    oh = torch.nn.functional.one_hot(B, n_bins).float()

    # W1
    A1 = oh.reshape(n_mc, -1)
    f1, r1 = _project(y, A1)
    # W2
    if pairs:
        A2 = torch.cat([A1] + [torch.einsum("na,nb->nab", oh[:, j], oh[:, k]).reshape(n_mc, -1) for j, k in pairs], 1)
    else:
        A2 = A1
    f2_full, r2 = _project(y, A2)
    f2 = f2_full - f1
    # W3: boosted CART fit (depth tree_depth) of the remainder - the reference
    # estimator of the axis-aligned piecewise-constant class.
    resid = r2.clone()
    tree_pred = torch.zeros_like(resid)
    for _ in range(10):
        leaves = _cart_leaf_indicators(B, resid, depth=tree_depth, min_leaf=10,
                                       max_leaf_keep=n_leaf_keep)
        if leaves.shape[1] == 0:
            break
        fit, resid = _project(resid, leaves)
        tree_pred = tree_pred + fit
    f3 = tree_pred
    r3 = r2 - f3
    f4 = r3

    shares = torch.tensor([float((fk ** 2).mean()) for fk in (f1, f2, f3, f4)]) / total
    shares = shares.clamp_min(0.0)
    shares = shares / shares.sum().clamp_min(EPS)
    sobol1 = float((f1 ** 2).mean() / total)
    sobol2 = float((f2 ** 2).mean() / total)
    return {
        "rho_star": [float(v) for v in shares],
        "sobol_main": sobol1,
        "sobol_pair": sobol2,
        "trend": [float(v) for v in shares],
    }


# ---------------------------------------------------------------------------
# task factory
# ---------------------------------------------------------------------------


def make_task(
    family: str = "hybrid",
    n_train: int = 2000,
    n_test: int = 1000,
    d: int = 10,
    noise: float = 0.1,
    seed: int = 0,
    max_pairs: int = 8,
    n_bins: int = 8,
    n_mc: int = 1500,
    compute_shares: bool = True,
) -> OracleTask:
    """Build one structured regression task with ground-truth shares."""
    assert family in FAMILIES, family
    rng = torch.Generator().manual_seed(seed)
    X_all = torch.randn(n_train + n_test, d, generator=rng)
    comps, feats, gen_pairs = _design_matrix(rng, d, family, n_active=min(max(3, d // 2), d))

    def f(x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(x.shape[0])
        for c in comps:
            out = out + c(x)
        return out

    f_all = f(X_all)
    f_all = (f_all - f_all.mean()) / f_all.std().clamp_min(1e-6)
    y_all = f_all + noise * torch.randn(n_train + n_test, generator=rng)

    X_train, X_test = X_all[:n_train], X_all[n_train:]
    y_train, y_test = y_all[:n_train], y_all[n_train:]
    f_test = f_all[n_train:]

    pairs = select_pairs(X_train, y_train, max_pairs=max_pairs, n_bins=4, seed=seed)
    if compute_shares:
        ref = reference_shares(f, d, pairs, n_mc=n_mc, n_bins=n_bins, seed=seed)
        rho = ref["rho_star"]
        sm, sp = ref["sobol_main"], ref["sobol_pair"]
    else:  # cheap approximation used during meta-training
        rho = [0.25, 0.25, 0.25, 0.25]
        sm, sp = 0.25, 0.25

    return OracleTask(
        name=f"{family}-d{d}-n{n_train}-s{seed}",
        family=family,
        seed=seed,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        f_test=f_test,
        rho_star=[float(v) for v in rho],
        sobol_main=float(sm),
        sobol_pair=float(sp),
        sigma2=float(noise ** 2),
        pairs=pairs,
        feature_support=feats,
        pair_support=gen_pairs,
        meta={"n_active": float(len(comps)), "noise": noise},
    )


def task_grid(
    families: Sequence[str] = FAMILIES,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    n_train: int = 2000,
    d: int = 10,
    noise: float = 0.1,
    **kwargs,
) -> List[OracleTask]:
    """The 5-family x 5-seed benchmark grid."""
    tasks = []
    for fam in families:
        for s in seeds:
            tasks.append(make_task(fam, n_train=n_train, d=d, noise=noise, seed=s, **kwargs))
    return tasks


# ---------------------------------------------------------------------------
# meta-learning task sampler
# ---------------------------------------------------------------------------


def sample_meta_task(seed: int, n_min: int = 400, n_max: int = 1600, d_min: int = 5, d_max: int = 12,
                     max_pairs: int = 6, n_bins: int = 8, n_mc: int = 800):
    """Sample one synthetic task for amortized profiler pre-training."""
    rng = torch.Generator().manual_seed(seed)
    fam = FAMILIES[int(torch.randint(0, len(FAMILIES), (1,), generator=rng))]
    n = int(torch.randint(n_min, n_max + 1, (1,), generator=rng))
    d = int(torch.randint(d_min, d_max + 1, (1,), generator=rng))
    noise = float(torch.exp(torch.rand((), generator=rng) * math.log(0.5) + math.log(0.02)))
    task = make_task(
        fam, n_train=n, n_test=max(n // 3, 200), d=d, noise=noise, seed=seed,
        max_pairs=max_pairs, n_bins=n_bins, n_mc=n_mc,
    )
    return task

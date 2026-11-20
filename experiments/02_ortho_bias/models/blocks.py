"""Heterogeneous inductive-bias blocks and the orthogonalization machinery.

Block inventory
---------------
H1  Additive main effects on a quantile-binned basis, with structural zero-mean
    centering (exact under the induced discrete measure, hence batch independent).
H2  Pairwise interactions on a tensor product of two binned bases, with structural
    bi-centering (exact zero marginals, hence orthogonal to H1 in L^2).
H3  Soft oblivious trees: axis-aligned, piecewise-constant function class with a
    differentiable (entmax) feature selector and annealed split temperature.
H4  Residual MLP: smooth high-order function class.

Orthogonalization
-----------------
``FWLResidualizer`` removes the component of a block output that lies in the span
of a basis of the *previous* blocks.  By the Frisch-Waugh-Lovell theorem the
residualized output is the part of the block that is unexplained by the earlier
blocks, so nested residualization realizes the chain of nested orthogonal
projections  W1 < W2 < W3 < W4  discussed in ``docs/theory.md``.

Two modes are provided:

``mode="batch"``
    The projection coefficients are obtained from the current mini-batch.  This
    is the *exact* (population-free) projection on that batch and is used for
    diagnostics only: it makes a prediction depend on the other rows in the
    batch.
``mode="running"``
    The coefficients come from exponential-moving-average Gram statistics, so
    they are shared by every row and the model stays an ordinary function of a
    single input.  This is the deployment mode (and the one used in benchmarks).

All centering operations are *structural* (they act on parameters, not on data),
which keeps the blocks batch-independent at inference time.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8

# ---------------------------------------------------------------------------
# sparse differentiable feature selection
# ---------------------------------------------------------------------------


def entmax15(z: torch.Tensor, dim: int = -1, n_iter: int = 40) -> torch.Tensor:
    """1.5-entmax (Peters, Niculae & Martins, 2019).

    ``p_i = max(0, z_i / 2 - tau)^2`` with ``tau`` chosen so that ``sum_i p_i = 1``.
    The solution is found by bisection on ``tau``; the threshold is treated as a
    constant for autograd (the standard straight-through practice for entmax).
    Exact zeros are produced, which is what makes the selected support axis
    aligned.
    """
    zmax = z.max(dim=dim, keepdim=True).values
    lo = zmax * 0.5 - 8.0
    hi = zmax * 0.5
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        p = torch.clamp(0.5 * z - mid, min=0.0) ** 2
        s = p.sum(dim=dim, keepdim=True)
        too_big = s > 1.0
        lo = torch.where(too_big, mid, lo)
        hi = torch.where(too_big, hi, mid)
    tau = 0.5 * (lo + hi)
    return torch.clamp(0.5 * z - tau, min=0.0) ** 2


# ---------------------------------------------------------------------------
# quantile-binned basis (spans the H1 and H2 hypothesis classes exactly)
# ---------------------------------------------------------------------------


class BinnedBasis(nn.Module):
    """Quantile-binned indicator basis.

    Each feature is split into ``n_bins`` bins with (approximately) equal
    probability mass, so that under the induced measure every bin has mass
    ``1 / n_bins``.  This makes the zero-mean (H1) and zero-marginal (H2)
    centering operators *exact* rather than estimated, and therefore identical at
    training and at inference time.
    """

    def __init__(self, n_bins: int = 8, n_features: Optional[int] = None):
        super().__init__()
        self.n_bins = int(n_bins)
        self.n_features = n_features
        self.register_buffer("edges", torch.zeros(0), persistent=False)

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> "BinnedBasis":
        x = x.detach().float()
        n, d = x.shape
        qs = torch.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
        e = torch.quantile(x, qs, dim=0)                     # (M-1, d)
        # Guarantee strict monotonicity even for heavily tied features.
        span = (x.max(dim=0).values - x.min(dim=0).values).clamp_min(1e-6)
        jitter = torch.arange(e.shape[0], device=x.device, dtype=x.dtype)[:, None]
        e = e + jitter * 1e-6 * span[None, :]
        self.edges = e.T.contiguous()                        # (d, M-1)
        self.n_features = d
        return self

    def indices(self, x: torch.Tensor) -> torch.Tensor:
        """Bin index per (row, feature); integer tensor of shape (n, d)."""
        x = x.detach().float()
        if self.edges.numel() == 0 or self.edges.shape[0] != x.shape[1]:
            self.fit(x)
        outs = [
            torch.searchsorted(self.edges[j], x[:, j].contiguous(), right=False)
            for j in range(x.shape[1])
        ]
        idx = torch.stack(outs, dim=1)
        return idx.clamp_(0, self.n_bins - 1).long()

    def onehot(self, x: torch.Tensor) -> torch.Tensor:
        """(n, d, M) one-hot indicator of the bin of every feature value."""
        return F.one_hot(self.indices(x), self.n_bins).float()

    def flat(self, x: torch.Tensor) -> torch.Tensor:
        """(n, d*M) flattened indicator basis - spans the additive class."""
        return self.onehot(x).reshape(x.shape[0], -1)

    def pair_flat(self, x: torch.Tensor, pairs: Sequence[Tuple[int, int]]) -> torch.Tensor:
        """(n, K*M*M) tensor-product indicators for the given feature pairs."""
        oh = self.onehot(x)                                  # (n, d, M)
        cols = []
        for (j, k) in pairs:
            cols.append(torch.einsum("na,nb->nab", oh[:, j], oh[:, k]).reshape(x.shape[0], -1))
        if not cols:
            return torch.zeros(x.shape[0], 0, device=x.device)
        return torch.cat(cols, dim=1)

    @property
    def dim_flat(self) -> int:
        return int(self.n_features) * self.n_bins

    @property
    def dim_pairs(self) -> int:
        return self.n_pairs * self.n_bins * self.n_bins

    n_pairs: int = 0


# ---------------------------------------------------------------------------
# H1: additive main effects
# ---------------------------------------------------------------------------


class H1Additive(nn.Module):
    """Additive piecewise-constant main-effect block.

    ``f1(x) = sum_j theta[j, b_j(x_j)] - mean_b theta[j, b]``.
    The centering is applied to the parameters, so ``E[f1] = 0`` exactly under the
    quantile (uniform-mass) binning and the block depends on a single row only.
    """

    block_id = 0

    def __init__(self, basis: BinnedBasis):
        super().__init__()
        self.basis = basis
        d, m = basis.n_features, basis.n_bins
        self.theta = nn.Parameter(torch.zeros(d, m))

    def table(self) -> torch.Tensor:
        return self.theta - self.theta.mean(dim=1, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = self.basis.indices(x)                          # (n, d)
        d = idx.shape[1]
        flat = (torch.arange(d, device=x.device)[None, :] * self.basis.n_bins + idx).reshape(-1)
        vals = self.table().reshape(-1)[flat].reshape(idx.shape)
        return vals.sum(dim=1)

    def n_real_params(self) -> float:
        return float(self.theta.numel() - self.basis.n_features)


# ---------------------------------------------------------------------------
# H2: pairwise interactions
# ---------------------------------------------------------------------------


class H2Pairwise(nn.Module):
    """Pairwise-interaction block with structural bi-centering.

    For every selected pair ``(j, k)`` the table ``T in R^{M x M}`` is replaced by

        T~ = T - mean_a T[a, :] - mean_b T[:, b] + mean T

    which has zero row and column means.  Hence ``E[T~ | x_j] = 0`` and
    ``E[T~ | x_k] = 0``, so the block is orthogonal in ``L^2`` to every function
    of ``x_j`` alone or of ``x_k`` alone - in particular to the whole H1 class -
    and pair tables with disjoint (or partially shared) supports are mutually
    orthogonal.  See Lemma 2 in ``docs/theory.md``.
    """

    block_id = 1

    def __init__(self, basis: BinnedBasis, pairs: Sequence[Tuple[int, int]], pair_scale: float = 1.0):
        super().__init__()
        self.basis = basis
        self.pairs = list(pairs)
        basis.n_pairs = len(self.pairs)
        m = basis.n_bins
        k = len(self.pairs)
        self.theta = nn.Parameter(torch.zeros(k, m, m))
        if pair_scale != 1.0:
            with torch.no_grad():
                self.theta.mul_(pair_scale)

    def tables(self) -> torch.Tensor:
        t = self.theta
        t = t - t.mean(dim=2, keepdim=True)                  # zero row means (over b)
        t = t - t.mean(dim=1, keepdim=True)                  # zero column means (over a)
        return t                                             # (K, M, M)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.pairs) == 0:
            return torch.zeros(x.shape[0], device=x.device)
        idx = self.basis.indices(x)
        m = self.basis.n_bins
        vals = self.tables().reshape(-1)
        out = torch.zeros(x.shape[0], device=x.device)
        for p, (j, k) in enumerate(self.pairs):
            flat = p * m * m + idx[:, j] * m + idx[:, k]
            out = out + vals[flat]
        return out

    def n_real_params(self) -> float:
        m = self.basis.n_bins
        return float(len(self.pairs) * (m - 1) * (m - 1))


# ---------------------------------------------------------------------------
# H3: soft oblivious trees (axis-aligned, piecewise constant)
# ---------------------------------------------------------------------------


class SoftObliviousTrees(nn.Module):
    """Ensemble of differentiable axis-aligned oblivious trees.

    Each of the ``n_trees`` trees has depth ``depth``; the internal node at level
    ``l`` owns a sparse feature selector ``p = entmax15(phi / T_feat)`` and a
    threshold ``theta``.  The gate of a node is

        s_n(x) = sigmoid( (p_n . x~ - theta_n) / T_split )

    with ``x~`` the standardised input.  Because ``entmax`` yields exact zeros the
    selector collapses to a single feature during training, so the tree is truly
    axis aligned; the split temperature is annealed so that ``s_n`` approaches a
    hard indicator.  Leaf activations are the products of the gates on the path
    from the root, and sum to one across the leaves of a tree, i.e. the class is
    that of piecewise-constant functions on an axis-aligned partition.
    """

    block_id = 2

    def __init__(
        self,
        n_features: int,
        n_trees: int = 8,
        depth: int = 2,
        init_temp: float = 1.0,
        feat_temp: float = 0.5,
        x_mean: Optional[torch.Tensor] = None,
        x_std: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        assert depth >= 1
        self.n_features = int(n_features)
        self.n_trees = int(n_trees)
        self.depth = int(depth)
        self.n_nodes = 2 ** depth - 1
        self.n_leaves = 2 ** depth
        self.init_temp = float(init_temp)
        self.feat_temp = float(feat_temp)
        self.feat_temp0 = float(feat_temp)

        self.feat_logits = nn.Parameter(0.5 * torch.randn(n_trees, self.n_nodes, n_features))
        self.thresholds = nn.Parameter(0.25 * torch.randn(n_trees, self.n_nodes))
        self.leaf_weights = nn.Parameter(0.1 * torch.randn(n_trees, self.n_leaves))

        self.register_buffer("x_mean", torch.zeros(n_features) if x_mean is None else x_mean.clone())
        self.register_buffer("x_std", torch.ones(n_features) if x_std is None else x_std.clone())
        self.register_buffer("temp", torch.tensor(float(init_temp)))

    # -- temperature schedule -------------------------------------------------
    def set_progress(self, frac: float) -> None:
        """Anneal split temperature geometrically, ``frac`` in [0, 1]."""
        f = min(max(float(frac), 0.0), 1.0)
        lo, hi = math.log(0.08), math.log(self.init_temp)
        self.temp.fill_(math.exp(hi + (lo - hi) * f))
        # the feature selector is annealed too: at the start it may mix features,
        # by the end entmax collapses to a single coordinate per node
        self.feat_temp = float(math.exp(math.log(self.feat_temp0) + (math.log(0.05) - math.log(self.feat_temp0)) * f))

    # -- internals ------------------------------------------------------------
    def gates(self, x: torch.Tensor) -> torch.Tensor:
        """Node gates, shape (T, n, n_nodes)."""
        xn = (x - self.x_mean) / self.x_std.clamp_min(1e-6)
        p = entmax15(self.feat_logits / self.feat_temp)      # (T, nodes, d)
        z = torch.einsum("tcd,nd->tcn", p, xn)               # (T, nodes, n)
        z = z.permute(0, 2, 1)                               # (T, n, nodes)
        return torch.sigmoid((z - self.thresholds[:, None, :]) / self.temp.clamp_min(1e-3))

    def leaf_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Leaf memberships, shape (T, n, 2^depth); they sum to 1 over leaves."""
        g = self.gates(x)                                    # (T, n, nodes)
        t, n, _ = g.shape
        a = torch.ones(t, n, 1, device=x.device, dtype=x.dtype)
        for level in range(self.depth):
            start = 2 ** level - 1
            g_l = g[:, :, start:start + 2 ** level]          # (T, n, 2^level)
            a = torch.cat([a * g_l, a * (1.0 - g_l)], dim=-1)
        return a

    def basis_columns(self, x: torch.Tensor) -> torch.Tensor:
        """Flattened leaf activations: a basis of the span of H3 outputs."""
        a = self.leaf_activations(x)                         # (T, n, L)
        return a.permute(1, 0, 2).reshape(x.shape[0], -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.leaf_activations(x)                         # (T, n, L)
        return torch.einsum("tnl,tl->n", a, self.leaf_weights)

    def selected_features(self) -> torch.Tensor:
        """Hard feature choice per node (argmax of the selector), (T, nodes)."""
        return (self.feat_logits / self.feat_temp).argmax(dim=-1)

    def n_real_params(self) -> float:
        return float(self.leaf_weights.numel() + self.n_trees * self.n_nodes * 2)


# ---------------------------------------------------------------------------
# H4: residual MLP (smooth high-order)
# ---------------------------------------------------------------------------


class ResidualMLP(nn.Module):
    """Small residual network used as the smooth high-order residual block."""

    block_id = 3

    def __init__(
        self,
        n_features: int,
        hidden: int = 128,
        n_blocks: int = 3,
        x_mean: Optional[torch.Tensor] = None,
        x_std: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.register_buffer("x_mean", torch.zeros(n_features) if x_mean is None else x_mean.clone())
        self.register_buffer("x_std", torch.ones(n_features) if x_std is None else x_std.clone())
        self.inp = nn.Linear(n_features, hidden)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, hidden),
                )
                for _ in range(n_blocks)
            ]
        )
        self.out = nn.Linear(hidden, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp((x - self.x_mean) / self.x_std.clamp_min(1e-6))
        h = F.gelu(h)
        for blk in self.blocks:
            h = h + 0.5 * blk(h)
        return self.out(F.gelu(h)).squeeze(-1)

    def n_real_params(self) -> float:
        return float(sum(p.numel() for p in self.parameters()))


# ---------------------------------------------------------------------------
# Hard-concrete gate (Louizos, Ullrich & Welling, 2018)
# ---------------------------------------------------------------------------

GAMMA = -0.1
ZETA = 1.1
BETA = 2.0 / 3.0
#: alpha threshold below which the *deterministic* test-time gate is exactly zero:
#: z(alpha) = clip(sigmoid(alpha) * (ZETA - GAMMA) + GAMMA, 0, 1) vanishes iff
#: sigmoid(alpha) <= -GAMMA / (ZETA - GAMMA) = 1/12, i.e. alpha <= log(1/11) ~= -2.3979
ALPHA_PRUNE = math.log(-GAMMA / (ZETA - GAMMA - (-GAMMA)))  # log(1/11)
ALPHA_ZERO = ALPHA_PRUNE  # backwards-compatible alias


class HardConcreteGate(nn.Module):
    """Per-block hard-concrete gate.

    Training:    u ~ U(0,1),  s = sigmoid((logit(u) + alpha) / BETA),
                 s~ = s * (ZETA - GAMMA) + GAMMA,  z = min(1, max(0, s~))
    Evaluation:  z = min(1, max(0, sigmoid(alpha) * (ZETA - GAMMA) + GAMMA))

    The evaluation gate is *exactly* zero whenever ``alpha <= log(1/11)``, so a
    pruned block contributes nothing (not merely "almost nothing").  The expected
    number of open gates used as the L0 penalty is
    ``sum_k (1 - sigmoid(alpha_k - BETA * log(-GAMMA / ZETA)))``.
    """

    def __init__(self, n_blocks: int, init_alpha: float = 1.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((n_blocks,), float(init_alpha)))

    def expected_open(self) -> torch.Tensor:
        # P(z > 0) = sigmoid(alpha - BETA * log(-GAMMA / ZETA))
        return torch.sigmoid(self.alpha - BETA * math.log(-GAMMA / ZETA)).sum()

    def forward(self, hard: Optional[bool] = None) -> torch.Tensor:
        if hard is None:
            hard = not self.training
        if hard:
            z = torch.sigmoid(self.alpha) * (ZETA - GAMMA) + GAMMA
            return z.clamp(0.0, 1.0)
        u = torch.rand_like(self.alpha).clamp(1e-6, 1 - 1e-6)
        s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + self.alpha) / BETA)
        return (s * (ZETA - GAMMA) + GAMMA).clamp(0.0, 1.0)

    @torch.no_grad()
    def pruned(self) -> torch.Tensor:
        """Boolean mask of blocks whose *evaluation* gate is exactly zero."""
        z = torch.sigmoid(self.alpha) * (ZETA - GAMMA) + GAMMA
        return z.clamp(0.0, 1.0) <= 0.0


# ---------------------------------------------------------------------------
# FWL residualizer: hierarchical orthogonal projection
# ---------------------------------------------------------------------------


def _equilibrated_ridge_solve(G: torch.Tensor, C: torch.Tensor, ridge: float) -> torch.Tensor:
    """Ridge solve with column equilibration.

    ``G`` is a (possibly badly scaled) Gram matrix: the binned indicator columns
    have norm ~1 while a tree-output column may have norm ~1e-2.  Without
    equilibration a scalar ridge shrinks the small columns to zero.  We solve in
    the normalised coordinates ``G / (s s^T)`` and rescale the solution back.
    Degenerate columns (``s ~ 0``) get a coefficient of exactly zero.
    """
    g = G.double()
    c = C.double()
    d = torch.diagonal(g).clamp_min(0.0)
    alive = d > (d.max().clamp_min(EPS) * 1e-10)
    s = d.sqrt().clamp_min(1e-8)
    gs = g / (s[:, None] * s[None, :])
    cs = c / s
    a = gs + ridge * torch.eye(g.shape[0], device=g.device, dtype=g.dtype)
    try:
        gamma = torch.linalg.solve(a, cs)
    except Exception:  # pragma: no cover - numerical fallback
        gamma = torch.linalg.lstsq(a, cs.unsqueeze(-1)).solution.squeeze(-1)
    beta = gamma / s
    return torch.where(alive, beta, torch.zeros_like(beta))


class FWLResidualizer(nn.Module):
    """Removes the projection of a block output onto a given basis.

    Given a basis ``Phi(x) in R^m`` (spanning the hypothesis class of the blocks
    that come *before* the current one) and a block output ``h(x)``, the
    residualised output is

        h_perp(x) = h(x) - Phi(x)^T beta,    beta = (E[Phi Phi^T])^{-1} E[Phi h].

    ``beta`` is the population least-squares coefficient; by the
    Frisch-Waugh-Lovell theorem ``h_perp`` is uncorrelated with every element of
    ``span(Phi)``, i.e. ``h_perp`` is the component of ``h`` that the earlier
    blocks cannot express.  In ``running`` mode ``beta`` is estimated from
    exponential-moving-average second moments (batch-independent, deployment
    safe); in ``batch`` mode it is solved on the current batch (exact, for
    diagnostics).
    """

    def __init__(
        self,
        dim: int,
        ridge: float = 1e-4,
        momentum: float = 0.95,
        mode: str = "running",
        accumulate_every: int = 1,
    ):
        super().__init__()
        assert mode in ("running", "batch", "batch_train", "none")
        self.dim = int(dim)
        self.ridge = float(ridge)
        self.momentum = float(momentum)
        self.mode = mode
        self.accumulate_every = int(accumulate_every)
        self.register_buffer("G", torch.zeros(self.dim, self.dim), persistent=False)
        self.register_buffer("C", torch.zeros(self.dim), persistent=False)
        self.register_buffer("beta", torch.zeros(self.dim), persistent=False)
        self.register_buffer("seen", torch.zeros(1), persistent=False)
        self._acc = 0
        self.frozen = False

    # -- statistics -----------------------------------------------------------
    @torch.no_grad()
    def observe(self, phi: torch.Tensor, h: torch.Tensor) -> None:
        """Accumulate (EMA) the second moments used to estimate ``beta``."""
        if self.mode in ("batch", "none") or self.frozen:
            return
        self._acc += 1
        if self._acc % self.accumulate_every != 0:
            return
        phi = phi.detach()
        h = h.detach()
        g = phi.T @ phi / phi.shape[0]
        c = phi.T @ h / phi.shape[0]
        if float(self.seen) < 0.5:
            self.G.copy_(g)
            self.C.copy_(c)
        else:
            a = self.momentum
            self.G.mul_(a).add_(g, alpha=1.0 - a)
            self.C.mul_(a).add_(c, alpha=1.0 - a)
        self.seen.add_(1.0)

    @torch.no_grad()
    def refresh(self) -> torch.Tensor:
        """Re-solve the projection coefficients from the running statistics."""
        if self.mode == "batch" or float(self.seen) < 0.5:
            return self.beta
        beta = _equilibrated_ridge_solve(self.G, self.C, self.ridge)
        self.beta.copy_(beta.to(self.beta.dtype))
        return self.beta

    # -- projection -----------------------------------------------------------
    def forward(self, phi: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if phi.shape[1] == 0 or self.mode == "none":
            return h
        if self.mode == "batch" or (self.mode == "batch_train" and self.training):
            with torch.no_grad():
                g = (phi.double().T @ phi.double()) / phi.shape[0]
                c = (phi.double().T @ h.detach().double()) / phi.shape[0]
                beta = _equilibrated_ridge_solve(g, c, self.ridge).to(phi.dtype)
            return h - phi @ beta
        beta = self.beta
        if float(beta.abs().sum()) == 0.0:
            return h
        return h - phi @ beta

    def freeze(self) -> None:
        """Stop updating the statistics (the calibration operator is fixed)."""
        self.frozen = True

    def unfreeze(self) -> None:
        self.frozen = False

    @torch.no_grad()
    def calibrate(self, phi: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Fit ``beta`` directly on a calibration sample (exact projection)."""
        if phi.shape[1] == 0:
            return self.beta
        nbar = max(phi.shape[0], 1)
        self.G.copy_(phi.T @ phi / nbar)
        self.C.copy_(phi.T @ h / nbar)
        self.seen.fill_(1.0)
        return self.refresh()

    @torch.no_grad()
    def residual_correlation(self, phi: torch.Tensor, h: torch.Tensor) -> float:
        """Max |correlation| between a raw block output and each basis column.

        A diagnostic: in ``batch`` mode it should be ~0 after projection.
        """
        if phi.shape[1] == 0:
            return 0.0
        hc = h - h.mean()
        pc = phi - phi.mean(dim=0, keepdim=True)
        num = (pc * hc[:, None]).mean(dim=0)
        den = pc.pow(2).mean(dim=0).sqrt() * hc.pow(2).mean().sqrt() + EPS
        return float((num / den).abs().max())


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def variance(v: torch.Tensor, unbiased: bool = False) -> torch.Tensor:
    return v.var(dim=0, unbiased=unbiased)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + EPS))


def covariance_matrix(outputs: torch.Tensor) -> torch.Tensor:
    """``outputs`` of shape (n, k) -> (k, k) covariance matrix."""
    c = outputs - outputs.mean(dim=0, keepdim=True)
    return c.T @ c / outputs.shape[0]


def choose_pairs(d: int, max_pairs: int, method: str = "all") -> List[Tuple[int, int]]:
    pairs = [(j, k) for j in range(d) for k in range(j + 1, d)]
    if len(pairs) <= max_pairs:
        return pairs
    if method == "first":
        return pairs[:max_pairs]
    return pairs[:max_pairs]

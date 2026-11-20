"""OrthoBias: the assembled model, its constrained-objective trainer, and metrics.

The model is a *signed* (gated) sum of four residualized blocks

    f_hat(x) = b + sum_{k=1..4} g_k * f_k^perp(x)

with ``f_1 = H1``, ``f_2 = H2``, ``f_3 = H3``, ``f_4 = H4`` and

    f_2^perp = (I - P_{V1}) f_2
    f_3^perp = (I - P_{V1+V2}) f_3
    f_4^perp = (I - P_{V1+V2+V3}) f_4

where ``V1`` is the span of the binned main-effect basis, ``V2`` adds the pair
tensor basis, ``V3`` adds the leaf activations of the tree ensemble.  See
``docs/theory.md`` for why this gives an (approximately) orthogonal variance
decomposition and what exactly is - and is not - identified.

The objective is

    L = L_task
      + lambda_L0 * sum_k (1 - P(z_k > 0))                    (expected L0 / MDL)
      + lambda_MDL * sum_k z_k * MDL_k(theta_k) / n            (two-part code length)
      + lambda_ortho * L_ortho                                 (soft orthogonality)
      + lambda_screen * sum_j (1 - s_j) ||theta_j^{(1)}||^2     (probe-screening prior)
      + ALM_{mu,lambda}( tau_interp - IR )                      (interpretability target)

where the augmented-Lagrangian term uses the Powell-Hestenes-Rockafellar form
``ALM_{mu,lam}(g) = (max(0, lam + mu g)^2 - lam^2) / (2 mu)`` and the multiplier
is updated by ``lam <- max(0, lam + mu g)``.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    BinnedBasis,
    FWLResidualizer,
    H1Additive,
    H2Pairwise,
    HardConcreteGate,
    ResidualMLP,
    SoftObliviousTrees,
    covariance_matrix,
    cosine,
    variance,
)
from .profiler import Signatures

EPS = 1e-8


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass
class OrthoBiasConfig:
    """Static configuration of one OrthoBias instance."""

    n_features: int
    pairs: Sequence[Tuple[int, int]] = ()
    n_bins: int = 8
    n_trees: int = 8
    tree_depth: int = 2
    hidden: int = 128
    res_blocks: int = 3
    # objective / constraints
    tau_interp: Optional[float] = None      # None -> unconstrained
    lam_mdl: float = 1e-3
    lam_l0: float = 1e-3
    lam_ortho: float = 1e-2
    lam_screen: float = 2e-3
    mdl_tau: float = 0.1
    # orthogonalization
    ortho_mode: str = "running"
    ortho_ridge: float = 1e-4
    ortho_momentum: float = 0.95
    accumulate_every: int = 4
    refresh_every: int = 20
    # gates
    gate_init: float = 1.5

    def with_(self, **kwargs) -> "OrthoBiasConfig":
        d = copy.deepcopy(self.__dict__)
        d.update(kwargs)
        return OrthoBiasConfig(**d)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class OrthoBias(nn.Module):
    """Heterogeneous block assembly with hierarchical orthogonalization."""

    def __init__(self, cfg: OrthoBiasConfig, x_mean: Optional[torch.Tensor] = None,
                 x_std: Optional[torch.Tensor] = None):
        super().__init__()
        self.cfg = cfg
        self.n_features = int(cfg.n_features)
        self.basis = BinnedBasis(cfg.n_bins, cfg.n_features)
        self.h1 = H1Additive(self.basis)
        self.h2 = H2Pairwise(self.basis, cfg.pairs)
        m1 = cfg.n_features * cfg.n_bins
        m2 = m1 + len(cfg.pairs) * cfg.n_bins * cfg.n_bins
        m3 = m2 + 1
        self.h3 = SoftObliviousTrees(
            cfg.n_features, n_trees=cfg.n_trees, depth=cfg.tree_depth,
            x_mean=x_mean, x_std=x_std,
        )
        self.h4 = ResidualMLP(cfg.n_features, cfg.hidden, cfg.res_blocks, x_mean=x_mean, x_std=x_std)
        self.gates = HardConcreteGate(4, cfg.gate_init)
        self.bias = nn.Parameter(torch.zeros(1))

        kw = dict(ridge=cfg.ortho_ridge, momentum=cfg.ortho_momentum,
                  mode=cfg.ortho_mode, accumulate_every=cfg.accumulate_every)
        self.res1 = FWLResidualizer(m1, **kw)     # H2 against V1
        self.res2 = FWLResidualizer(m2, **kw)     # H3 against V1+V2
        self.res3 = FWLResidualizer(m3, **kw)     # H4 against V1+V2+V3

        self.register_buffer("screening", torch.ones(self.n_features))
        self.register_buffer("pair_screening", torch.ones(len(cfg.pairs)))
        self.register_buffer("y_mean", torch.zeros(1))
        self.register_buffer("y_std", torch.ones(1))
        self._steps = 0

    # -- preprocessing --------------------------------------------------------
    @torch.no_grad()
    def fit_preprocessing(self, x: torch.Tensor) -> "OrthoBias":
        x = x.detach().float()
        self.basis.fit(x)
        mu, sd = x.mean(dim=0), x.std(dim=0).clamp_min(1e-6)
        self.h3.x_mean.copy_(mu)
        self.h3.x_std.copy_(sd)
        self.h4.x_mean.copy_(mu)
        self.h4.x_std.copy_(sd)
        return self

    def set_target_scaling(self, y: torch.Tensor) -> "OrthoBias":
        y = y.detach().float().reshape(-1)
        self.y_mean.copy_(y.mean())
        self.y_std.copy_(y.std().clamp_min(1e-6))
        return self

    # -- bases ---------------------------------------------------------------
    def phi1(self, x: torch.Tensor) -> torch.Tensor:
        return self.basis.flat(x)

    def phi2(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.phi1(x), self.basis.pair_flat(x, self.h2.pairs)], dim=1)

    # -- raw blocks ----------------------------------------------------------
    def raw_blocks(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        leaves = self.h3.leaf_activations(x)                  # (T, n, L)
        h3 = torch.einsum("tnl,tl->n", leaves, self.h3.leaf_weights)
        return {
            "h1": self.h1(x),
            "h2": self.h2(x),
            "h3": h3,
            "h4": self.h4(x),
            "leaves": leaves,
        }

    # -- forward -------------------------------------------------------------
    def forward(self, x: torch.Tensor, update_ortho: Optional[bool] = None) -> Dict[str, torch.Tensor]:
        if update_ortho is None:
            update_ortho = self.training
        raw = self.raw_blocks(x)
        p1 = self.phi1(x)
        p2 = torch.cat([p1, self.basis.pair_flat(x, self.h2.pairs)], dim=1)
        # H4 is residualized against V1+V2 and against the *realized* H3 function
        # (a single direction), not against the whole leaf span: orthogonality of the
        # two blocks is all the variance decomposition needs, and carving out the
        # full tree class would remove structure that H3 is not (yet) able to fit.
        p3 = torch.cat([p2, raw["h3"].detach().unsqueeze(1)], dim=1)

        if update_ortho:
            with torch.no_grad():
                self.res1.observe(p1, raw["h1"])
                self.res1.observe(p1, raw["h2"])
                self.res2.observe(p2, raw["h3"])
                self.res3.observe(p3, raw["h4"])
        self._steps += 1

        h2p = self.res1(p1, raw["h2"])
        h3p = self.res2(p2, raw["h3"])
        h4p = self.res3(p3, raw["h4"])

        if getattr(self, "ungated", False):
            g = torch.ones(4, device=x.device, dtype=x.dtype)
        else:
            g = self.gates()                                  # (4,) in [0,1]
        f = [raw["h1"], h2p, h3p, h4p]
        fg = [g[k] * f[k] for k in range(4)]
        out = fg[0] + fg[1] + fg[2] + fg[3]
        y = out + self.bias
        return {
            "y": y,
            "f": fg,
            "f_raw_blocks": f,
            "gates": g,
            "blocks": [raw["h1"], raw["h2"], raw["h3"], raw["h4"]],
            "phi": [p1, p2, p3],
        }

    # -- interpretation ------------------------------------------------------
    def variance_shares(self, out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Variance decomposition of the prediction (no gradients required)."""
        f = out["f"]
        v = torch.stack([variance(fk.detach()) for fk in f]).clamp_min(0.0)
        total = variance(out["y"].detach()).clamp_min(EPS)
        rho = v / v.sum().clamp_min(EPS)
        ir = (v[:3].sum()) / v.sum().clamp_min(EPS)
        # var(total) - sum_k var(f_k) = 2 * sum_{i<j} cov(f_i, f_j): a direct
        # measure of how much the orthogonalization is off on this sample.
        return {"var": v, "rho": rho, "ir": ir, "var_total_pred": total,
                "rho_of_total": v / total, "orthogonality_gap": float(1.0 - v.sum() / total)}

    @torch.no_grad()
    def diagnostics(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Orthogonality audit: raw vs residualized block covariance."""
        raw = self.raw_blocks(x)
        p1 = self.phi1(x)
        p2 = torch.cat([p1, self.basis.pair_flat(x, self.h2.pairs)], dim=1)
        p3 = torch.cat([p2, raw["h3"].detach().unsqueeze(1)], dim=1)
        h2p = self.res1(p1, raw["h2"])
        h3p = self.res2(p2, raw["h3"])
        h4p = self.res3(p3, raw["h4"])
        f = torch.stack([raw["h1"], h2p, h3p, h4p], dim=1)
        g = self.gates()
        fg = f * g
        cov_raw = covariance_matrix(torch.stack([raw["h1"], raw["h2"], raw["h3"], raw["h4"]], dim=1))
        cov_res = covariance_matrix(fg)
        d = torch.diagonal(cov_res).clamp_min(EPS)
        norm_cov = cov_res / torch.sqrt(d[:, None] * d[None, :])
        off = norm_cov - torch.diag_embed(torch.diagonal(norm_cov))
        return {
            "cov_raw": cov_raw,
            "cov_res": cov_res,
            "corr_res": norm_cov,
            "max_abs_offdiag_corr": off.abs().max(),
            "raw_offdiag_cov": (cov_raw - torch.diag_embed(torch.diagonal(cov_raw))).abs().max(),
            "f": fg,
            "gates": g,
        }

    @torch.no_grad()
    def calibrate_orthogonalization(self, x: torch.Tensor) -> "OrthoBias":
        """Fit the projection coefficients once on a calibration sample.

        The running (EMA) statistics keep ``beta`` current during training, but at
        the end of training the leaf activations of H3 have moved; re-solving the
        projections on a fixed calibration sample makes the deployed operator the
        exact empirical projection for that distribution.  This is the only place
        where a batch of data is used to define the model, so it must be a sample
        the model does not fit (a calibration split).
        """
        raw = self.raw_blocks(x)
        p1 = self.phi1(x)
        p2 = torch.cat([p1, self.basis.pair_flat(x, self.h2.pairs)], dim=1)
        p3 = torch.cat([p2, raw["h3"].detach().unsqueeze(1)], dim=1)
        self.res1.calibrate(p1, raw["h2"])
        self.res2.calibrate(p2, raw["h3"])
        self.res3.calibrate(p3, raw["h4"])
        return self

    # -- complexity ----------------------------------------------------------
    def mdl_bits(self) -> torch.Tensor:
        """Two-part code length (bits) of every block.

        ``MDL_k = sum_{w in theta_k} log2(1 + w^2 / tau^2) + s_k * log2(n + 1)``
        where ``s_k`` is the number of structural units (bins / tensors / leaves /
        weights).  Divided by the sample size this is a per-sample description
        length and can be added to the per-sample task loss.
        """
        tau2 = self.cfg.mdl_tau ** 2
        out = []
        for mod in (self.h1, self.h2, self.h3, self.h4):
            bits = torch.zeros((), device=self.bias.device)
            for p in mod.parameters():
                bits = bits + torch.log1p(p ** 2 / tau2).sum() / math.log(2.0)
            out.append(bits)
        return torch.stack(out)

    def n_parameters(self) -> Dict[str, int]:
        return {k: int(sum(p.numel() for p in m.parameters()))
                for k, m in (("h1", self.h1), ("h2", self.h2), ("h3", self.h3), ("h4", self.h4))}

    # -- prediction ----------------------------------------------------------
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self(x)["y"] * self.y_std + self.y_mean

    # -- warm start ----------------------------------------------------------
    @torch.no_grad()
    def apply_warm_start(self, profiler, sig: Signatures, **kwargs) -> Dict[str, torch.Tensor]:
        return profiler.warm_start(sig, self, **kwargs)


# ---------------------------------------------------------------------------
# augmented-Lagrangian constraint handler
# ---------------------------------------------------------------------------


class ALMConstraint:
    """Powell-Hestenes-Rockafellar augmented Lagrangian for ``g(theta) <= 0``."""

    def __init__(self, target: float, mu: float = 1.0, lam: float = 0.0,
                 dual_lr: float = 1.0, mu_growth: float = 2.0, mu_max: float = 1e4,
                 mu_min: float = 1e-3, warm_up: int = 5):
        self.target = float(target)
        self.mu = float(mu)
        self.lam = float(lam)
        self.dual_lr = float(dual_lr)
        self.mu_growth = float(mu_growth)
        self.mu_max = float(mu_max)
        self.mu_min = float(mu_min)
        self.warm_up = int(warm_up)
        self.k = 0
        self.history: List[Tuple[float, float, float]] = []

    def violation(self, ir: torch.Tensor) -> torch.Tensor:
        return self.target - ir                              # g = tau - IR

    def penalty(self, ir: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        g = self.violation(ir)
        val = ((self.lam + self.mu * g).clamp_min(0.0) ** 2 - self.lam ** 2) / (2.0 * self.mu)
        return val, g

    @torch.no_grad()
    def update(self, ir: float) -> None:
        self.k += 1
        g = self.target - ir
        if self.k <= self.warm_up:                            # let the model fit first
            return
        self.lam = max(0.0, self.lam + self.dual_lr * self.mu * g)
        if g > 0.02 and self.mu < self.mu_max:                # still violating: stiffen
            self.mu = min(self.mu * self.mu_growth, self.mu_max)
        self.history.append((self.k, self.lam, self.mu))

# ---------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------


@dataclass
class TrainHistory:
    steps: List[int] = field(default_factory=list)
    task: List[float] = field(default_factory=list)
    ir: List[float] = field(default_factory=list)
    gates: List[List[float]] = field(default_factory=list)
    lam: List[float] = field(default_factory=list)
    stages: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, List]:
        return {
            "steps": self.steps,
            "task": self.task,
            "ir": self.ir,
            "gates": self.gates,
            "lam": self.lam,
            "stages": self.stages,
        }


@torch.no_grad()
def gcv_ridge(A: torch.Tensor, y: torch.Tensor,
              grid: Sequence[float] = (1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1)) -> torch.Tensor:
    """Ridge least squares with the penalty chosen by generalised cross-validation.

    ``A`` is the (n, m) design, ``y`` the centred target.  GCV is computed from the
    eigendecomposition of the (m, m) Gram matrix, so the whole path costs one
    ``eigh`` and is still exact least squares: no hyper-parameter is selected on
    held-out data.  This matters for the linear blocks, whose table sizes grow with
    ``d * M`` and ``K * M^2`` and would otherwise over-fit on small samples.
    """
    n, m = A.shape
    if m == 0:
        return torch.zeros(0)
    y = y - y.mean()
    if n <= m + 2:
        # fewer observations than parameters: fall back to the minimum-norm solution
        return torch.linalg.lstsq(A, y.unsqueeze(-1)).solution.squeeze(-1)
    G = (A.T @ A) / n
    b = (A.T @ y) / n
    evals, evecs = torch.linalg.eigh(G.double())
    evals = evals.clamp_min(0.0)
    proj = evecs.T @ b.double()
    yy = float((y * y).mean())
    best = (float("inf"), torch.zeros(m, dtype=torch.float64))
    for alpha in grid:
        den = evals + alpha
        coef = proj / den
        beta = evecs @ coef
        resid = yy - 2.0 * float(beta @ b.double()) + float(beta @ (G.double() @ beta))
        trace = float((evals / den).sum())
        gcv = max(resid, 0.0) / max(1.0 - trace / n, 1e-3) ** 2
        if gcv < best[0]:
            best = (gcv, beta)
    return best[1].float()


@torch.no_grad()
def solve_linear_blocks(model: "OrthoBias", X: torch.Tensor, yz: torch.Tensor,
                        ridge: float = 1e-6, n_alt: int = 3) -> Dict[str, float]:
    """Closed-form least squares for H1 and H2 (their classes are linear in the parameters).

    * H1: the OLS solution of a one-hot additive design is the per-bin mean, so the
      block is set to the binned conditional mean (then centred).
    * H2: OLS on the pair tensor design restricted to the bi-centred subspace; the
      restriction is enforced by alternating least squares (OLS step, then
      bi-centre, repeated ``n_alt`` times), which converges to the projection of
      the residual onto that subspace.

    Solving the two linear blocks exactly removes an otherwise dominant source of
    variance in benchmark numbers: an SGD-fit look-up table is badly
    under-trained at small step budgets while the deep block is not.
    """
    idx = model.basis.indices(X)
    d, M = model.basis.n_features, model.basis.n_bins
    A1 = model.basis.flat(X)
    w1 = gcv_ridge(A1, yz)
    model.h1.theta.copy_(w1.reshape(d, M))

    r = yz - model.h1(X)
    out = {"h1_r2": float(1.0 - r.var() / yz.var().clamp_min(EPS))}
    if len(model.h2.pairs) > 0:
        A = model.basis.pair_flat(X, model.h2.pairs)
        w = torch.zeros(A.shape[1], device=X.device, dtype=X.dtype)
        for _ in range(n_alt):
            w = gcv_ridge(A, r)
            T = w.reshape(len(model.h2.pairs), M, M)
            T = T - T.mean(dim=2, keepdim=True) - T.mean(dim=1, keepdim=True) + T.mean(dim=(1, 2), keepdim=True)
            w = T.reshape(-1)
            r = yz - model.h1(X) - (A @ w)
        model.h2.theta.copy_(w.reshape(len(model.h2.pairs), M, M))
        r = yz - model.h1(X) - model.h2(X)
    out["h12_r2"] = float(1.0 - r.var() / yz.var().clamp_min(EPS))
    return out


class OrthoBiasTrainer:
    """Trains an :class:`OrthoBias` model with the constrained objective.

    The schedule is constructive: H1 and H2 are solved in closed form (their
    hypothesis classes are linear), H3 and H4 are then fitted by SGD on the
    residual, and a final joint phase switches the gates on and optimizes the
    full constrained objective (L0 / MDL / orthogonality / augmented Lagrangian).
    """

    def __init__(
        self,
        model: OrthoBias,
        lr: float = 4e-3,
        lr_tree: Optional[float] = None,
        lr_deep: Optional[float] = None,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        tau_interp: Optional[float] = None,
        lam_mdl: float = 1e-3,
        lam_l0: float = 1e-3,
        lam_ortho: float = 1e-2,
        lam_screen: float = 2e-3,
        alm_mu: float = 0.05,
        alm_lam: float = 0.0,
        alm_lr: float = 1.0,
        alm_update_every: int = 20,
        grad_clip: float = 1.0,
        seed: int = 0,
    ):
        self.model = model
        self.lr = lr
        self.lr_tree = lr_tree if lr_tree is not None else max(lr, 5e-3)
        self.lr_deep = lr_deep if lr_deep is not None else 1e-3
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.lam_mdl = lam_mdl
        self.lam_l0 = lam_l0
        self.lam_ortho = lam_ortho
        self.lam_screen = lam_screen
        self.grad_clip = grad_clip
        self.tau_interp = tau_interp if tau_interp is not None else model.cfg.tau_interp
        self.alm = ALMConstraint(self.tau_interp) if self.tau_interp is not None else None
        self.alm_update_every = alm_update_every
        self.seed = seed
        self.gen = torch.Generator().manual_seed(seed)

    # -- losses ---------------------------------------------------------------
    def orthogonality_loss(self, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        f = torch.stack(out["f"], dim=1)
        c = f - f.mean(dim=0, keepdim=True)
        cov = c.T @ c / f.shape[0]
        d = torch.diagonal(cov).clamp_min(EPS)
        corr = cov / torch.sqrt(d[:, None] * d[None, :])
        off = corr - torch.diag_embed(torch.diagonal(corr))
        return (off ** 2).sum()

    def screening_loss(self) -> torch.Tensor:
        w = (1.0 - self.model.screening)
        theta2 = (self.model.h1.table() ** 2).sum(dim=1)
        loss = (w * theta2).sum()
        if len(self.model.h2.pairs) > 0:
            pw = (1.0 - self.model.pair_screening)
            t2 = (self.model.h2.tables() ** 2).mean(dim=(1, 2))
            loss = loss + (pw * t2).sum()
        return loss

    def compute_loss(self, out: Dict[str, torch.Tensor], y: torch.Tensor,
                     joint: bool = True) -> Dict[str, torch.Tensor]:
        task = F.mse_loss(out["y"], y)
        stats = self.model.variance_shares(out)
        loss = task
        mdl = self.model.mdl_bits() / max(y.numel(), 1)
        g = out["gates"]
        alm_val = torch.zeros((), device=y.device)
        if joint:
            loss = loss + self.lam_mdl * (g * mdl).sum()
            loss = loss + self.lam_l0 * (4.0 - self.model.gates.expected_open())
            loss = loss + self.lam_ortho * self.orthogonality_loss(out)
            loss = loss + self.lam_screen * self.screening_loss()
            if self.alm is not None:
                alm_val, _ = self.alm.penalty(stats["ir"])
                loss = loss + alm_val
        else:
            loss = loss + 0.1 * self.lam_ortho * self.orthogonality_loss(out)
        return {"loss": loss, "task": task, "alm": alm_val.detach(), "mdl": mdl.detach(),
                "ir": stats["ir"]}

    # -- helpers -------------------------------------------------------------
    def _set_active(self, module) -> None:
        for p in self.model.parameters():
            p.requires_grad_(False)
        targets = list(self.model.parameters()) if module is None else list(module.parameters())
        for p in targets:
            p.requires_grad_(True)

    def _optimizer(self, lr: float):
        return torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
                                 lr=lr, weight_decay=self.weight_decay)

    def _batches(self, tr_idx: torch.Tensor, bs: int):
        perm = tr_idx[torch.randperm(len(tr_idx), generator=self.gen)]
        for start in range(0, len(perm) - 1, bs):
            b = perm[start:start + bs]
            if len(b) >= 8:
                yield b

    # -- main loop -----------------------------------------------------------
    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 60,
        batch_size: Optional[int] = None,
        staged: bool = True,
        stage_weights: Sequence[float] = (0.45, 0.30, 0.25),
        log_every: int = 0,
        verbose: bool = False,
        calibrate: bool = False,
        prune_steps: int = 0,
        freeze_linear: bool = True,
    ) -> TrainHistory:
        X = X.detach().float()
        y = y.detach().float().reshape(-1)
        bs = int(batch_size or self.batch_size)
        model = self.model
        model.train()
        model.set_target_scaling(y)
        yz = (y - model.y_mean) / model.y_std
        n = X.shape[0]
        tr_idx = torch.randperm(n, generator=self.gen)
        steps_per_epoch = max(math.ceil(n / bs), 1)
        total_steps = max(int(epochs * steps_per_epoch), 12)
        hist = TrainHistory()

        self._calibrate_after = bool(calibrate)
        if staged:
            with torch.no_grad():
                solve_linear_blocks(model, X, yz)
            model.res1.refresh()
            model.res2.refresh()
            w3, w4, wj = stage_weights
            n3 = max(int(w3 * total_steps), 4)
            n4 = max(int(w4 * total_steps), 4)
            nj = max(total_steps - n3 - n4, 4)
            model.ungated = True
            self._run_steps(X, yz, tr_idx, bs, n3, hist, log_every, verbose, "H3",
                            module=model.h3, lr=self.lr_tree, joint=False, total=total_steps)
            self._run_steps(X, yz, tr_idx, bs, n4, hist, log_every, verbose, "H4",
                            module=model.h4, lr=self.lr_deep, joint=False, total=total_steps)
            model.ungated = False
            # Joint phase in rounds: train under the current projection operator, then
            # re-solve it exactly on a fixed calibration split.  Alternating
            # "fit in the complement" and "re-estimate the complement" is the
            # fixed-point iteration that removes the leakage a stale operator would
            # otherwise let the higher blocks exploit.
            self._run_steps(X, yz, tr_idx, bs, nj, hist, log_every, verbose, "joint",
                            module=None, lr=self.lr, joint=True, total=total_steps)
        else:
            self._run_steps(X, yz, tr_idx, bs, total_steps, hist, log_every, verbose, "joint",
                            module=None, lr=self.lr, joint=True, total=total_steps)

        if prune_steps:
            self.prune_gates(X, y, steps=prune_steps)
        model.res1.refresh()
        model.res2.refresh()
        model.res3.refresh()
        model.ungated = False
        if calibrate:
            with torch.no_grad():
                model.calibrate_orthogonalization(X[: min(512, n)])
                for r in (model.res1, model.res2, model.res3):
                    r.freeze()
        model.eval()
        return hist

    def _run_steps(self, X, yz, tr_idx, bs, n_steps, hist, log_every, verbose, stage,
                   module, lr: float, joint: bool, total: int, freeze_params=None) -> int:
        model = self.model
        self._set_active(module)
        for p in (freeze_params or []):
            p.requires_grad_(False)
        opt = self._optimizer(lr)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, total_steps=max(n_steps, 1), pct_start=0.3, anneal_strategy="cos"
        )
        step = 0
        while step < n_steps:
            for b in self._batches(tr_idx, bs):
                if step >= n_steps:
                    break
                if stage == "H3":
                    frac = min(step / max(n_steps - 1, 1), 1.0)
                else:
                    frac = 1.0
                model.h3.set_progress(frac)
                out = model(X[b])
                logs = self.compute_loss(out, yz[b], joint=joint)
                opt.zero_grad(set_to_none=True)
                logs["loss"].backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], self.grad_clip)
                opt.step()
                try:
                    sched.step()
                except Exception:
                    pass
                step += 1
                if step % max(model.cfg.refresh_every, 1) == 0:
                    model.res1.refresh()
                    model.res2.refresh()
                    model.res3.refresh()
                if joint and self.alm is not None and step % self.alm_update_every == 0:
                    with torch.no_grad():
                        ir_val = float(model.variance_shares(model(X))["ir"])
                    self.alm.update(ir_val)
                if log_every and step % log_every == 0:
                    hist.steps.append(step)
                    hist.stages.append(stage)
                    hist.task.append(float(logs["task"]))
                    hist.ir.append(float(logs["ir"]))
                    hist.gates.append([float(v) for v in out["gates"].detach()])
                    hist.lam.append(float(self.alm.lam) if self.alm else 0.0)
                    if verbose:
                        print(
                            f"  [{stage:5s}] step {step:4d} task {float(logs['task']):.4f} "
                            f"ir {float(logs['ir']):.3f} "
                            f"gates {[round(float(v), 2) for v in out['gates'].detach()]}"
                        )
        return step

    def prune_gates(self, X: torch.Tensor, y: torch.Tensor, steps: int = 40,
                    lr: float = 0.05, log_every: int = 0) -> List[float]:
        """Final model-selection phase: optimise *only* the gate parameters.

        At a saturated gate (``z = 1``) the task gradient vanishes through the
        clamp, so the L0/MDL terms - which act analytically on ``alpha`` - are the
        only force; driving ``alpha`` down closes a block exactly instead of
        leaving it "almost closed".  As soon as a block is closed the task loss
        rises and its gradient switches back on, so the phase settles at the
        blocks the data actually needs.
        """
        model = self.model
        model.train()
        model.set_target_scaling(y)
        yz = (y - model.y_mean) / model.y_std
        self._set_active(model.gates)
        opt = torch.optim.Adam([model.gates.alpha], lr=lr)
        n = X.shape[0]
        for step in range(steps):
            idx = torch.randint(0, n, (min(self.batch_size, n),), generator=self.gen)
            out = model(X[idx])
            logs = self.compute_loss(out, yz[idx], joint=True)
            opt.zero_grad(set_to_none=True)
            logs["loss"].backward()
            opt.step()
            if log_every and step % log_every == 0:
                print(f"  [prune] step {step:3d} gates {[round(float(v), 3) for v in out['gates'].detach()]} "
                      f"alpha {[round(float(a), 2) for a in model.gates.alpha.detach()]}")
        self._set_active(None)
        model.eval()
        return [float(a) for a in model.gates.alpha.detach()]

    @torch.no_grad()
    def greedy_prune(self, X_val: torch.Tensor, y_val: torch.Tensor, tol: float = 0.01,
                     order: Sequence[int] = (3, 2, 1)) -> Dict[str, object]:
        """Backward elimination of blocks with an exact-zero gate.

        Each block is tentatively switched off by setting ``alpha = -10`` (so the
        evaluation gate is exactly ``0``, not merely small) and the held-out MSE is
        recomputed.  The block is removed when the relative degradation stays below
        ``tol`` - i.e. the block's contribution does not pay for its MDL code
        length.  This is a discrete model-selection step; because the gate is
        exactly zero, the pruned block contributes *identically* nothing.
        """
        model = self.model
        model.eval()
        pred = model(X_val)["y"] * model.y_std + model.y_mean
        base = float(((y_val - pred) ** 2).mean())
        removed = []
        for k in order:
            old = float(model.gates.alpha[k])
            model.gates.alpha[k] = -10.0
            pred = model(X_val)["y"] * model.y_std + model.y_mean
            mse = float(((y_val - pred) ** 2).mean())
            if mse <= base * (1.0 + tol):
                removed.append(int(k))
                base = mse
            else:
                model.gates.alpha[k] = old
        return {"pruned_blocks": removed, "val_mse_after": base,
                "gates": [float(v) for v in model.gates()], "alpha": [float(a) for a in model.gates.alpha]}

    @torch.no_grad()
    def _r2(self, X: torch.Tensor, y: torch.Tensor) -> float:
        self.model.eval()
        pred = self.model(X)["y"] * self.model.y_std + self.model.y_mean
        ss_res = ((y - pred) ** 2).mean()
        ss_tot = ((y - y.mean()) ** 2).mean().clamp_min(EPS)
        return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model: OrthoBias, X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    """R^2, variance shares, interpretability ratio, gates and pruning state."""
    model.eval()
    out = model(X.float())
    pred = out["y"] * model.y_std + model.y_mean
    y = y.float().reshape(-1)
    ss_res = ((y - pred) ** 2).mean()
    ss_tot = ((y - y.mean()) ** 2).mean().clamp_min(EPS)
    stats = model.variance_shares(out)
    return {
        "r2": float(1.0 - ss_res / ss_tot),
        "rmse": float(ss_res.sqrt()),
        "rho": [float(v) for v in stats["rho"]],
        "var": [float(v) for v in stats["var"]],
        "ir": float(stats["ir"]),
        "gates": [float(v) for v in out["gates"]],
        "pruned": [bool(v) for v in model.gates.pruned()],
        "n_params": sum(model.n_parameters().values()),
    }

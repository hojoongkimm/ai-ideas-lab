"""Baselines for the benchmark.

Two groups:

* neural, implemented here so that they use the same optimiser, batching and
  early-stopping protocol as OrthoBias (a fair comparison of *architectures*, not
  of training budgets);
* library models (scikit-learn / XGBoost) when available, wrapped behind the same
  ``fit`` / ``predict`` interface.

Every model exposes ``name``, ``fit(X, y)``, ``predict(X)`` and ``fit_time``.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualMLP, SoftObliviousTrees

EPS = 1e-8


class _TorchRegressor:
    """Shared training loop for the neural baselines."""

    name = "torch"
    epochs = 150
    lr = 3e-3
    batch_size = 256

    def _build(self, d: int) -> nn.Module:  # pragma: no cover - overridden
        raise NotImplementedError

    def fit(self, X: torch.Tensor, y: torch.Tensor, seed: int = 0) -> "._TorchRegressor":
        torch.manual_seed(seed)
        X = X.detach().float()
        y = y.detach().float().reshape(-1)
        self.mu = y.mean()
        self.sd = y.std().clamp_min(1e-6)
        yz = (y - self.mu) / self.sd
        self.net = self._build(X.shape[1])
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        g = torch.Generator().manual_seed(seed)
        n = X.shape[0]
        steps_per_epoch = max(math.ceil(n / self.batch_size), 1)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.lr, total_steps=self.epochs * steps_per_epoch,
            pct_start=0.3, anneal_strategy="cos",
        )
        t0 = time.time()
        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            for s in range(0, n - 1, self.batch_size):
                b = perm[s:s + self.batch_size]
                if len(b) < 8:
                    continue
                loss = F.mse_loss(self.net(X[b]).reshape(-1), yz[b])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                try:
                    sched.step()
                except Exception:
                    pass
        self.fit_time = time.time() - t0
        self.net.eval()
        return self

    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        return self.net(X.float()).reshape(-1) * self.sd + self.mu


class MLPBaseline(_TorchRegressor):
    name = "MLP"

    def __init__(self, hidden: int = 128, epochs: int = 150):
        self.hidden = hidden
        self.epochs = epochs

    def _build(self, d: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(d, self.hidden), nn.GELU(), nn.Linear(self.hidden, self.hidden),
            nn.GELU(), nn.Linear(self.hidden, 1),
        )


class ResNetBaseline(_TorchRegressor):
    """H4 alone: the same residual network without any other block."""

    name = "ResNet(H4)"

    def __init__(self, hidden: int = 128, blocks: int = 3, epochs: int = 150):
        self.hidden, self.blocks, self.epochs = hidden, blocks, epochs

    def _build(self, d: int) -> nn.Module:
        mu = torch.zeros(d)
        sd = torch.ones(d)
        return ResidualMLP(d, self.hidden, self.blocks, mu, sd)


class NAMBaseline(_TorchRegressor):
    """Neural additive model: one MLP per feature, summed (Agarwal et al., 2021)."""

    name = "NAM"

    def __init__(self, hidden: int = 32, epochs: int = 200):
        self.hidden, self.epochs = hidden, epochs

    def _build(self, d: int) -> nn.Module:
        return _NAM(d, self.hidden)


class _NAM(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.nets = nn.ModuleList(
            [nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
             for _ in range(d)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = 0.0
        for j, net in enumerate(self.nets):
            out = out + net(x[:, j:j + 1]).squeeze(-1)
        return out


class SoftTreeBaseline(_TorchRegressor):
    """H3 alone: a soft oblivious tree ensemble (gradient-trained, axis aligned)."""

    name = "SoftTree(H3)"

    def __init__(self, n_trees: int = 16, depth: int = 3, epochs: int = 200):
        self.n_trees, self.depth, self.epochs = n_trees, depth, epochs

    def fit(self, X: torch.Tensor, y: torch.Tensor, seed: int = 0):
        X = X.detach().float()
        mu, sd = X.mean(dim=0), X.std(dim=0).clamp_min(1e-6)
        self._mu_x, self._sd_x = mu, sd
        self.y_mu = y.mean()
        self.y_sd = y.std().clamp_min(1e-6)
        yz = (y - self.y_mu) / self.y_sd
        net = SoftObliviousTrees(X.shape[1], self.n_trees, self.depth, x_mean=mu, x_std=sd)
        opt = torch.optim.AdamW(net.parameters(), lr=5e-3, weight_decay=1e-4)
        g = torch.Generator().manual_seed(seed)
        n = X.shape[0]
        t0 = time.time()
        for ep in range(self.epochs):
            net.set_progress(ep / max(self.epochs - 1, 1))
            perm = torch.randperm(n, generator=g)
            for s in range(0, n - 1, 256):
                b = perm[s:s + 256]
                if len(b) < 8:
                    continue
                loss = F.mse_loss(net(X[b]), yz[b])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
        self.fit_time = time.time() - t0
        self.net = net.eval()
        return self

    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        return self.net(X.float()) * self.y_sd + self.y_mu


# ---------------------------------------------------------------------------
# library baselines
# ---------------------------------------------------------------------------


class _SklearnWrapper:
    name = "sklearn"

    def __init__(self, model=None, name: str = "sklearn"):
        self.model = model
        self.name = name

    def fit(self, X, y, seed: int = 0):
        t0 = time.time()
        try:
            self.model.set_params(random_state=seed)
        except Exception:
            pass
        self.model.fit(X.numpy(), y.numpy())
        self.fit_time = time.time() - t0
        return self

    def predict(self, X):
        import numpy as np
        return torch.from_numpy(self.model.predict(X.numpy())).float()


def library_baselines(seed: int = 0) -> List[object]:
    """HistGradientBoosting and RandomForest when scikit-learn is importable."""
    out: List[object] = []
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor  # noqa

        out.append(_SklearnWrapper(HistGradientBoostingRegressor(random_state=seed), "HistGBM"))
        out.append(_SklearnWrapper(RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=seed), "RF"))
    except Exception:
        pass
    try:
        from xgboost import XGBRegressor  # noqa

        out.append(_SklearnWrapper(XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                                                subsample=0.9, n_jobs=2), "XGBoost"))
    except Exception:
        pass
    return out


def neural_baselines(seed: int = 0) -> List[object]:
    return [MLPBaseline(), ResNetBaseline(), NAMBaseline(), SoftTreeBaseline()]

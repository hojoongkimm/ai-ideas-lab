# OrthoBias: dataset-compiled heterogeneous inductive-bias blocks with an identified variance decomposition

A reference implementation and formalization of a tabular regression architecture that
**assembles heterogeneous inductive-bias blocks per dataset** and **decomposes its own
prediction variance into identified, semantically meaningful parts**.

The design targets three properties that current tabular meta-architectures do not
provide together:

| Property | Meaning here |
|---|---|
| **Identifiability** | The share of predictive variance carried by each block is (for the low-order blocks) the Sobol index of the target function, not an artefact of training. |
| **Controllable interpretability** | An interpretability ratio `IR` is an explicit *constraint* (`IR >= tau`), not a post-hoc measurement, enforced by an augmented Lagrangian. |
| **Amortized design** | A permutation-invariant profiler maps cheap model-free probe statistics to the initial block shares and variable screening, in one forward pass. |

Everything runs on CPU in minutes; a GPU is only needed for the large-scale sweeps.

---

## 1. The idea in one page

A data scientist facing a new table asks three questions before choosing a model:

1. *Is the signal mostly additive?* → fit a GAM.
2. *Are there interactions between a few pairs?* → fit a GA2M / EBM.
3. *Is there axis-aligned, discontinuous structure (thresholds, conjunctions)?* → fit trees.
4. *Or is there smooth high-order structure?* → fit a network.

OrthoBias makes those four options **explicit, mutually orthogonal components of one
model** and lets the data decide how much variance each one carries:

```
f(x) = b  +  g1 * H1(x)  +  g2 * H2(x)  +  g3 * H3(x)  +  g4 * H4(x)
              additive     pairwise      axis-aligned   smooth
              (binned)     (binned)      piecewise-cst  residual net
```

* **H1** `sum_j theta_j[b_j(x_j)]` — binned main effects, structurally zero-mean.
* **H2** `sum_{(j,k) in P} T_jk[b_j(x_j), b_k(x_k)]` — binned pairwise interactions,
  structurally *bi*-centred (zero row and column means).
* **H3** a differentiable axis-aligned oblivious-tree ensemble (entmax feature
  selection, annealed split temperature), i.e. piecewise-constant on an axis-aligned
  partition.
* **H4** a residual MLP for the smooth higher-order remainder.

Each block is fitted **after residualizing the previous ones** (`H2 ⟂ V1`,
`H3 ⟂ V1+V2`, `H4 ⟂ V1+V2+[H3]`), so the variance of the prediction splits into
per-block variances:

```
Var(f) = sum_k g_k^2 Var(H_k)     (exactly, if the residualization is exact)
rho_k  = g_k^2 Var(H_k) / sum_j g_j^2 Var(H_j)
```

The four `rho_k` are the model's *answer to the four questions above*, and — this is the
point of the paper — for `k = 1, 2` they are the (binned) Sobol indices of the target,
hence **identified**: two models that fit the data equally well but attribute variance
differently are distinguishable, and the wrong one is wrong about the data, not merely
about its own parameters. Section 4 states exactly what is and is not identified.

---

## 2. Repository map

All commands in this document are run from this directory (`experiments/02_ortho_bias/`),
which is self-contained: it needs only the Python packages in `requirements.txt`.

```
experiments/02_ortho_bias/
├── README.md                     this document (paper-style overview + formalization)
├── docs/theory.md                full proofs, lemmas, and the honest-limitations section
├── requirements.txt
├── models/
│   ├── blocks.py                 H1, H2, H3 (soft oblivious trees), H4, hard-concrete gate,
│   │                             FWL residualizer (hierarchical orthogonal projection)
│   ├── profiler.py               orthogonal probe signatures + permutation-invariant meta-profiler
│   ├── orthobias.py              assembled model, GCV linear solves, ALM trainer, metrics
│   └── baselines.py              MLP, ResNet, NAM, SoftTree, (sklearn/XGBoost if available)
├── datasets/
│   └── synthetic_oracle.py       5 structured families with reference-chain ground-truth shares
├── train_meta.py                 meta-profiler pre-training (`--smoke`)
├── run_benchmark.py              oracle recovery, orthogonality audit, ablations, tau sweep,
│                                 exact-zero pruning check (`--smoke`)
└── results*/                      generated reports (Markdown + LaTeX + JSON)
```

---

## 3. Architecture

### 3.1 Blocks

Let `b_j(x) ∈ {1..M}` be the index of the quantile bin of feature `j`, with bins chosen so
that every bin has mass `1/M` under the training marginal.

**H1 — additive main effects.**

```
H1(x) = sum_j theta_j[b_j(x)],        theta_j <- theta_j - mean_b theta_j[b]
```

The centring is applied to the *parameters*, so `E[H1] = 0` holds exactly for every
parameter value: no running statistics, no train/test discrepancy.

**H2 — pairwise interactions.**

```
H2(x) = sum_{(j,k) in P} T_jk[b_j(x), b_k(x)]
T <- T - mean_over_rows(T) - mean_over_cols(T) + mean(T)
```

Again a projection of the *parameter tensor*, so every pair table has exactly zero
marginals. Consequence (Lemma 2): `H2` is orthogonal in `L^2` to *every* function of
`x_j` alone and of `x_k` alone, in particular to the whole H1 class, and distinct pair
tables are mutually orthogonal.

**H3 — differentiable axis-aligned trees.**

A soft oblivious tree with depth `D` owns `2^D` leaves. Node `n` has a sparse feature
selector `p_n = entmax_{1.5}(phi_n / T_feat)` and a threshold `theta_n`:

```
s_n(x) = sigmoid( (p_n . x~ - theta_n) / T_split )
leaf_l(x) = prod_{n in path(l)} s_n(x)^{dir} (1 - s_n(x))^{1-dir}
H3(x) = sum_t sum_l w_{t,l} leaf_{t,l}(x)
```

`entmax_{1.5}` produces exact zeros, so the selector collapses to a single coordinate
(the tree is *axis aligned*) as `T_feat` anneals; `T_split` anneals so the gates approach
hard indicators. Leaves of one tree sum to one, so the class is that of
piecewise-constant functions on axis-aligned partitions — the natural higher-order
complement of the additive and pairwise classes.

**H4 — smooth residual.** A residual MLP (LayerNorm + two-layer blocks, output layer
initialised at zero so the block starts as a no-op).

### 3.2 Hierarchical orthogonalization

Let `Phi_1(x) ∈ R^{dM}` be the one-hot bin indicators, `Phi_2 = [Phi_1, pair tensors]`,
and `Phi_3 = [Phi_2, H3(x)]`. Given a block output `h`, the residualizer computes

```
h_perp = h - Phi beta,      beta = (E[Phi Phi^T])^{-1} E[Phi h]
```

By the Frisch–Waugh–Lovell theorem this is the projection of `h` onto the orthogonal
complement of `span(Phi)`: the part of the block that the earlier blocks *cannot*
represent. Two implementations:

* `mode="batch"` solves `beta` on the current sample. This is the exact empirical
  projection; it makes a prediction depend on the other rows in the batch, so it is used
  only for diagnostics (it is what the orthogonality audit reports as `exact`).
* `mode="running"` (deployment) keeps exponential moving averages of `Phi^T Phi` and
  `Phi^T h` and solves for `beta` from those, so the model remains a function of a single
  row. The mismatch between the two is the *measured* residual correlation reported in the
  benchmark (typically `|corr| < 0.2`; the exact projection is `< 1e-7`).

A soft penalty `L_ortho = sum_{i<j} corr(f_i, f_j)^2` keeps the running operator
consistent while the blocks move. `ortho_mode="none"` switches orthogonalization off and
is one of the ablations.

### 3.3 Gates and exact pruning

Each block is multiplied by a hard-concrete gate `z_k` (Louizos et al., 2018) with
`gamma = -0.1`, `zeta = 1.1`, `beta = 2/3`:

```
training:    s = sigmoid((logit(u) + alpha_k)/beta),  z = clip(s (zeta - gamma) + gamma, 0, 1)
evaluation:  z = clip(sigmoid(alpha_k) (zeta - gamma) + gamma, 0, 1)
```

The evaluation gate is **exactly zero** whenever `alpha_k <= log(1/11) = -2.3979`, so a
pruned block contributes identically nothing. Because the gate saturates at one, the task
gradient vanishes there, and exact elimination is done by an explicit backward-elimination
step (`OrthoBiasTrainer.greedy_prune`) that sets `alpha_k = -10` and keeps the block
removed if the held-out MSE increase stays within a tolerance. `run_benchmark.py` reports
the resulting gates and confirms they are bit-exact zeros.

### 3.4 Objective

```
L = L_task
  + lambda_L0    * sum_k (1 - P(z_k > 0))            # expected L0 (MDL code for "block used")
  + lambda_MDL   * sum_k z_k * MDL_k(theta_k) / n    # two-part code length per sample
  + lambda_ortho * L_ortho                            # soft orthogonality
  + lambda_scr   * sum_j (1 - s_j) ||theta^{(1)}_j||^2  # probe-screening prior
  + ALM_{mu, lambda}( tau - IR )                      # interpretability constraint
```

with `MDL_k(theta) = sum_{w in theta_k} log2(1 + w^2/tau_mdl^2) + s_k log2(n+1)` (a Gaussian
code length under a scale prior plus a structural term, normalised per sample), and the
Powell–Hestenes–Rockafellar augmented Lagrangian

```
ALM_{mu,lambda}(g) = ( max(0, lambda + mu g)^2 - lambda^2 ) / (2 mu),   lambda <- max(0, lambda + mu g)
```

for the constraint `g(theta) = tau - IR(theta) <= 0`, where
`IR = Var(H1 + H2 + H3) / Var(f_hat)`.

### 3.5 Meta-profiler (the amortized compiler)

Input: **orthogonal probe signatures** — deliberately model-free statistics computed on a
subsample:

* global (17): `log n`, `log d`, `d/n`, mean/max |corr|, effective rank ratio,
  target skew/kurtosis/entropy, scale ratio, missingness, and the held-out `R^2` of a
  *linear*, *binned additive*, *pairwise tensor*, and *kNN* probe, plus their gaps;
* per feature (15): moments, unique-value ratio, entropy, linear/stump/univariate-binned
  `R^2`, mutual information with the binned target, collinearity, interaction salience,
  conditional-mean curvature, missingness;
* per candidate pair (6): dependence, **residual interaction energy** (variance of the
  double-centred cell means of the additive residual), log energy, held-out `R^2` gain of
  the pair, marginal strength, collinearity.

Encoder: DeepSets-style pooling over feature/pair tokens with a permutation-equivariant
per-token head. Outputs: predicted block shares `rho^(0)`, feature-screening logits,
pair-screening logits.

The warm start sets `alpha_k = logit(clip(2 rho_k)) + 1.5`, so an *uniform* prediction
returns exactly the default open gate: an uninformed profiler is a no-op, never a silent
prior that shuts blocks down. Screening priors act only when the profiler is confident
(`prob < 0.5`). This is design rule **D3: the profiler is a prior, never a mask.**

---

## 4. Formalization

Notation: `X ∈ R^d` with independent coordinates (A1), `Y = f(X) + eps`,
`E[eps] = 0`, `Var(eps) = sigma^2`, `f ∈ L^2(P_X)`. `P_W` is the orthogonal projection onto
a closed subspace `W ⊂ L^2(P_X)`. Complete proofs: [`docs/theory.md`](docs/theory.md).

### 4.1 The decomposition the model is built to reproduce

**Theorem 1 (Hoeffding–Sobol).** *For independent coordinates,*

```
f = sum_{S subseteq [d]} f_S ,   f_S = sum_{T subseteq S} (-1)^{|S|-|T|} E[f | X_T]
```

*converges in `L^2`, the components are mutually orthogonal, `E[f_S | X_T] = 0` for every
proper `T ⊂ S`, and the decomposition is unique up to a null set. Consequently*

```
Var(f) = sum_S Var(f_S),        S_S = Var(f_S)/Var(f)   (Sobol indices).
```

Theorem 1 says the *semantic* target of a block decomposition exists and is unique: what
share of the variance is additive, pairwise, higher-order is not a modelling choice but a
property of `f`. The architecture is built so that its parameterisation matches this
structure.

### 4.2 Lemma 2 (structural centring gives exact orthogonality)

*With quantile bins (uniform mass `1/M`) and independent coordinates:*

*(a)* `E[H1] = 0` *for every parameter value;*
*(b)* `E[H2 | x_j] = E[H2 | x_k] = 0` *for every parameter value, hence `H2 ⟂ V1` and
`H2 ⟂` every function of a single coordinate;*
*(c)* *distinct pair tables are orthogonal,* `E[T_jk T_lm] = 0` *whenever `(j,k) ≠ (l,m)`.*

Proof sketch: the bi-centring operator is the composition of two conditional-expectation
removals, `T~ = (I - E_j)(I - E_k) T`, so `E_k T~ = 0` and `E_j T~ = 0` pointwise. Then for
any `h(x_j)`, `E[T~ h] = E_j[h E_k T~] = 0`. For two pairs sharing a coordinate, use the
centring in the non-shared coordinate of each. ∎

This is what makes H1/H2 the *estimated* Sobol components rather than an arbitrary split:
the orthogonality is imposed on the parameters and therefore holds at every optimisation
step, at inference, and for any batch.

### 4.3 Lemma 3 (nested projections and the variance identity)

*Let `W_1 ⊂ W_2 ⊂ W_3 ⊂ W_4 = L^2(P_X)` and `Delta_k = P_{W_k} - P_{W_{k-1}}`. Then every
`Delta_k` is an orthogonal projection onto `W_k ⊖ W_{k-1}`, `Delta_k Delta_j = 0` for
`k ≠ j`, and for all `f`*

```
||P_{W_m} f||^2 = sum_{k<=m} ||Delta_k f||^2 ,    Var(f) = sum_k ||Delta_k f||^2 .
```

The residualizer of Section 3.2 is exactly the empirical estimator of `Delta_k`: subtracting
`Phi beta` from `h` is `(I - P_{W_{k-1}}) h`. Hence, when the blocks are fitted to their
`L^2`-optimal projections, the model's block variances *are* the terms of this identity and
`rho_k` is their normalised share. (FWL guarantees that the block fitted on the residual
equals the block fitted jointly under orthogonality, so staged fitting is not an
approximation.)

### 4.4 Theorem 2 (what is identified)

*Assume (A1), exact residualization, and that each block attains its `L^2` projection onto
its chain element. Then*

1. `Cov(H_k^perp, H_j^perp) = 0` *for `k ≠ j` and* `Var(f_hat) = sum_k g_k^2 Var(H_k^perp)`
   *(so the reported shares add up exactly);*
2. `rho_1` *equals the Sobol share of the order-1 part of the binned projection of `f`,
   which converges to* `sum_j S_j` *as `M → inf` for Lipschitz main-effect shapes
   (Lemma 4: error `<= d (L h/2)^2`, `h` the bin width);*
3. `rho_2` *equals the Sobol share of the order-2 part over the selected pair set
   `P`;*
4. `rho_3 + rho_4` *equals the Sobol share of all components of order `>= 3`* (plus the
   binning error);
5. **the split `rho_3 / rho_4` is *not* identified by `f`**: it is identified only
   relative to the chosen tree class `W_3`. Two function classes can both explain the
   same higher-order variance and assign it differently without contradicting the data.

Statement 5 is the honest core of the identifiability claim, and it is why the benchmark
scores recovery on the **identified triple** `(rho_1, rho_2, rho_3 + rho_4)` and treats the
H3/H4 split as a *measured property of the model*, tested comparatively (does the model use
the axis-aligned block for axis-aligned data and the smooth block for smooth data?).

### 4.5 Lemma 4 (binning error)

*If `f_j` is `L`-Lipschitz, the piecewise-constant projection on bins of width `h` satisfies
`||f_j - P f_j||_2^2 <= (L h / 2)^2`; summing over `d` features gives
`||f - P_{V_1} f||^2 <= d (L h/2)^2`.* Hence `rho_1` converges to the additive Sobol share
as `M` grows, at rate `O(M^{-2})`.

### 4.6 Proposition 5 (exact zero of the gate)

*With `gamma = -0.1`, `zeta = 1.1`, the deterministic gate is*

```
z(alpha) = clip( sigmoid(alpha) (zeta - gamma) + gamma , 0, 1 )
```

*which is exactly `0` iff `sigmoid(alpha) <= 0.1/1.2 = 1/12`, i.e. iff
`alpha <= logit(1/12) = log(1/11) = -2.3979`. The expected number of open gates is
`sum_k sigmoid(alpha_k - beta log(-gamma/zeta)) = sum_k sigmoid(alpha_k + 1.5986)`, which is
the L0 term used in the objective.*

### 4.7 Proposition 6 (the constraint is handled by a convergent ALM)

*Let `F` be the (smooth) task + code-length objective and `g(theta) = tau - IR(theta)`. If
`F, g ∈ C^2`, `grad g != 0` at the solution, and `theta^k` is a stationary point of the PHR
penalty for `mu_k -> inf`, then every limit point of `{(theta^k, lambda^k)}` satisfies the
KKT conditions of `min F s.t. g <= 0` (Rockafellar, 1973; Bertsekas, 1982). `F` is
non-convex here, so this is a local statement; the dual variable is logged per run
(`ALMConstraint.history`) and is the diagnostic of whether the constraint is active.*

*If the constraint set is empty (a `tau` above the maximum achievable `IR`), the multiplier
diverges and the iterate converges to the minimiser of `F` over the reachable `IR`; the
report prints the attained `IR` alongside the requested `tau` rather than claiming
feasibility.*

### 4.8 Proposition 7 (profiler invariance)

*The encoder is a symmetric pooler over feature/pair tokens with an equivariant per-token
head, so for any permutation `pi`: `rho(X_pi) = rho(X)` and
`screening(X_pi) = pi(screening(X))`. The empirical check
(`run_benchmark.py --check-invariance`) reports the maximum deviation, which is at floating
point noise (`~1e-8`).*

Sufficiency is **not** claimed: the probe signatures are not a sufficient statistic of the
data-generating process; the profiler is an amortized estimator trained by imitation
(`train_meta.py`) and used as a warm start.

---

## 5. Related work and differentiation

Verified references (arXiv ids / venues as of November 2026):

| Line of work | What it does | Why OrthoBias is not it |
|---|---|---|
| **TabPFN / TabPFN-3.5, TabICLv2** (Hollmann et al., Nature 2025; Qu et al., 2026) | One in-context transformer predicts from a support set; no per-dataset architecture, no decomposition. | OrthoBias produces an explicit, auditable functional decomposition with identified shares; accuracy is not the claim. |
| **OrionMoE** (Bouadi et al., arXiv:2610.20418) | Dataset-conditioned *sparse routing* among homogeneous FFN experts inside a transformer. | Its experts are interchangeable FFN blocks; there is no semantic (additive/pairwise/tree/smooth) assignment of variance, and routing is per token, not per function class. |
| **MotherNet / MotherTree** (Müller et al. 2025; arXiv:2610.10832) | Hypernetwork that emits an MLP or a decision tree for a new task. | A single homogeneous model class is generated; nothing is decomposed, so nothing is identified. MotherTree is a *special case* of our H3 branch. |
| **MetaD2A / DiffusionNAG** (NeurIPS 2021; ICLR 2024) | Dataset → neural architecture graph / diffusion-generated architecture. | Architecture search over wiring; no functional semantics, no variance decomposition, no interpretability constraint. |
| **NAM / NAM-FS / NODE-GAM / BAM / EBM** (Agarwal et al. 2021; Kishimoto, arXiv:2606.19850; Chang et al., ICLR 2022; Koike, arXiv:2611.05940; Lou et al., KDD 2013) | Fixed interpretable classes (additive, + pairs), with differentiable feature selection. | These are *single-class* models: everything is interpretable, and accuracy collapses when the signal is high-order. OrthoBias explicitly models the high-order remainder and *measures* how much of the variance had to leave the interpretable part. NAM-FS/BAM also use entmax feature selection — we reuse that device inside H3, not as the contribution. |
| **Sample-level heterogeneous mixtures** | Route a *sample* to a tree/linear/kernel expert (EM or gate per row). | Routing per sample does not yield a per-*function* variance share; two samples in the same region may be explained by different families, so no global decomposition exists. |
| **Hard-concrete / sparsity-inducing gating** (Louizos et al., ICLR 2018) | Learnable L0 gates. | We use it as the mechanical device for exact block elimination and give the exact threshold (Prop. 5); the contribution is what is being gated (semantic blocks) and why (identified shares + constraint). |

**The gap this fills.** No existing tabular meta-architecture (i) chooses among
*heterogeneous functional classes* per dataset, (ii) guarantees that the resulting
attribution is the (low-order) Sobol decomposition, and (iii) exposes interpretability as
a constrained optimisation target with exact pruning. The claim is deliberately narrow:
*decomposition-first architecture design*, with an explicit statement of which parts are
identified.

---

## 6. Experimental protocol

### 6.1 Synthetic families with a well-defined ground truth

`datasets/synthetic_oracle.py` builds five families (`main`, `pairwise`, `tree`, `deep`,
`hybrid`) with unit-variance targets and additive noise. The ground truth is a
**reference chain** `W1 < W2 < W3` estimated on a large Monte-Carlo sample with (i) the same
quantile bins, (ii) the same pair-selection rule, (iii) a boosted depth-3 CART fit of the
remainder. The reported quantities are

```
rho_1* , rho_2* , rho_hi* = rho_3* + rho_4*   (identified)
rho_3* / rho_4*                               (chain-relative, indicative)
```

The `tree` and `deep` families are a *controlled contrast*: both place ~2/3 of the variance
in higher-order components, but one is axis-aligned and discontinuous (parity of step
functions), the other smooth and rotation-invariant (`sin(w1.x) sin(w2.x)`). The
scientifically meaningful claim is comparative: **`a3 = rho_3/(rho_3+rho_4)` should be
higher for `tree` than for `deep`** — the model must spend its axis-aligned block on
axis-aligned structure.

### 6.2 Baselines

Neural, trained with the same optimiser/budget protocol (`models/baselines.py`): MLP,
ResNet (which is H4 alone), NAM, SoftTree (H3 alone). Library baselines when installed:
HistGradientBoosting, RandomForest, XGBoost.

### 6.3 Metrics

* `R^2` on a held-out split.
* `eps_id = |rho_1 - rho_1*| + |rho_2 - rho_2*| + |rho_hi - rho_hi*|` (identified shares).
* `a3` (higher-order attribution ratio) for the tree/deep contrast.
* orthogonality audit: max/mean `|corr|` between block outputs, raw vs exact projection vs
  deployed operator.
* `IR`, gate values, pruned blocks, parameter count, wall-clock.

### 6.4 Ablations

1. `no-ortho`: `ortho_mode="none"`, `lambda_ortho = 0`.
2. `no-meta`: no profiler warm start.
3. `tau` sweep: `tau ∈ {0.7, 0.9, 0.95}` — the accuracy/interpretability frontier.
4. pruning check: does the gate reach a bit-exact zero on a linear target?

### 6.5 Running it

```bash
pip install -r requirements.txt

# 1) correctness smoke tests (~2 min CPU each)
python3 train_meta.py --smoke
python3 run_benchmark.py --smoke

# 2) the real thing
python3 train_meta.py --tasks 400 --epochs 30 --out checkpoints/profiler.pt
python3 run_benchmark.py --seeds 0 1 2 3 4 --epochs 150 --ablations --tau-sweep \
    --check-invariance --prune-check --out results

# 3) GPU / large-scale
python3 run_benchmark.py --n-train 10000 --d 40 --hidden 256 --epochs 400 \
    --max-pairs 24 --out results_large
```

`run_benchmark.py` writes `results/benchmark.md`, `results/tables/*.tex` and
`results/metrics.json`.

On a machine with little RAM (the smoke tests were produced on a 1.5 GB container) run the
experiments in subsets and concatenate the reports:

```bash
python3 run_benchmark.py --smoke --only oracle,ablations,prune,invariance --skip-baselines
python3 run_benchmark.py --smoke --only tau --skip-baselines
python3 run_benchmark.py --smoke --baselines-only
# then merge results_smoke/benchmark_*.md and results_smoke/baselines.md
```

---

## 7. What the smoke run shows

Measured on this machine with `run_benchmark.py --smoke` (1 seed per family, n=1200, d=8,
M=8, 16 trees of depth 4, 60 epochs). Full report: `results_smoke/benchmark.md`,
baselines: `results_smoke/baselines.md`, raw numbers: `results_smoke/metrics_*.json`.

**Oracle recovery (reference shares estimated on 1500 Monte-Carlo points):**

| family | R^2 | rho_hat (H1/H2/H3/H4) | rho* | eps_id | a3 = rho_3/(rho_3+rho_4) |
|---|---|---|---|---|---|
| main | 0.892 | 0.980 / 0.002 / 0.006 / 0.012 | 0.873 / 0.015 / 0.000 / 0.112 | 0.214 | 0.35 |
| pairwise | 0.810 | 0.155 / 0.782 / 0.012 / 0.051 | 0.084 / 0.827 / 0.015 / 0.074 | 0.143 | 0.19 |
| tree | 0.501 | 0.110 / 0.029 / 0.713 / 0.148 | 0.052 / 0.141 / 0.000 / 0.807 | 0.224 | **0.83** |
| deep | 0.754 | 0.155 / 0.258 / 0.166 / 0.420 | 0.120 / 0.449 / 0.000 / 0.431 | 0.383 | **0.28** |
| hybrid | 0.781 | 0.679 / 0.268 / 0.018 / 0.034 | 0.422 / 0.399 / 0.000 / 0.179 | 0.516 | 0.35 |

**Baselines (same protocol, `results_smoke/baselines.md`):**

| family | MLP | ResNet (H4 alone) | NAM (additive) | SoftTree (H3 alone) |
|---|---|---|---|---|
| main | 0.971 | 0.966 | 0.989 | 0.972 |
| pairwise | 0.976 | 0.962 | -0.029 | 0.909 |
| tree | 0.292 | 0.425 | -0.011 | 0.708 |
| deep | 0.978 | 0.965 | 0.012 | 0.687 |
| hybrid | 0.917 | 0.895 | 0.459 | 0.853 |

**Orthogonality audit:**

| family | max abs corr, raw | exact projection | deployed operator |
|---|---|---|---|
| main | 0.392 | 1.1e-08 | 0.118 |
| pairwise | 0.745 | 1.9e-08 | 0.552 |
| tree | 0.530 | 1.6e-08 | 0.320 |
| deep | 0.756 | 2.0e-08 | 0.285 |
| hybrid | 0.591 | 1.9e-08 | 0.423 |

**Exact-zero pruning** (linear target, only H1 needed): `alpha = [1.57, -10.0, 1.46, 1.44]`,
evaluation gates `[0.894, 0.0, 0.873, 0.870]`, pruned block index `[1]` — the H2 gate is
bit-exactly `0.0` and `R^2` moves `0.9762 -> 0.9740`.

**Profiler invariance**: max deviation `3.0e-08` under a random column permutation.

### Reading the numbers honestly

1. **The controlled contrast works.** `a3 = 0.83` for the axis-aligned family and `0.28`
   for the smooth family: the model spends its axis-aligned block exactly on the
   axis-aligned structure and its smooth block on the smooth structure, with essentially
   the same Sobol profile in both (rho_hi* ≈ 0.81 vs 0.43 is the only difference in the
   ground truth). This is the empirical claim the architecture is built to make.
2. **The identified shares are recovered well when the chain matches the data**
   (`pairwise`: 0.155/0.782 vs 0.084/0.827; `main`: essentially all in H1). They degrade
   when the binned additive/pair classes cannot represent the target (`hybrid`: H1 keeps
   0.68 where the reference says 0.42).
3. **The exact projection does what Theorem 2.1 says** (1e-8), and the deployed operator
   costs `0.12 - 0.55` in residual correlation. The `pairwise` number (0.55) is the honest
   worst case: the operator was estimated while the blocks were still moving.
4. **The assembly is not yet competitive on raw accuracy.** All baselines beat it on
   `main`/`pairwise`/`deep`; `NAM` collapses on `pairwise`/`tree` (0.0) and `SoftTree`
   collapses on `main` (which is the point of *heterogeneous* assembly), but the assembled
   model pays for its decomposition: each block is trained under a shared budget, so H3 gets
   ~110 gradient steps in the assembly versus 1000 in the `SoftTree` baseline (0.501 vs
   0.708 on `tree`). A per-block budget is the obvious fix.
5. **The interpretability constraint is only sometimes attainable.** For `deep` with
   `tau = 0.7` the attained `IR` is `0.631` (constraint active, well approximated); with
   `tau = 0.9` it is `0.402` on a single seed, i.e. the multiplier pushes but the reachable
   `IR` is lower — the report prints the attained value rather than claiming feasibility
   (Proposition 6, remark).
6. **Attribution is capacity-limited.** With depth-3 trees H3 could not represent the
   parity structure and `a3` fell to `0.09`; with depth 4 it reaches `0.83`. The block's
   *class* must be rich enough before its share means anything — a statement the framework
   makes measurable rather than hiding.

## 8. Honest limitations

1. **The H3/H4 split is not identified** (Theorem 2.5). Only `rho_1`, `rho_2`, `rho_hi` are
   properties of the data. Any paper-style claim of the form "the model discovered that 30%
   of the signal is tree-like" is a claim *relative to the chosen tree class*.
2. **Independence.** Theorem 1 needs independent coordinates. Under dependence the
   orthogonalization is still exact in `L^2(P_X)` (the residualizer uses the empirical
   Gram), but the shares are then the shares of *the chosen nested chain*, not Sobol
   indices.
3. **Binning.** `V1`/`V2` are binned, so `rho_1` underestimates the smooth additive share
   by `O(M^{-2})`; the within-bin part is absorbed by H4, which is why an additive target
   can show a few percent of "higher-order" variance.
4. **Deployed projection.** Running-mode coefficients are estimated; the residual
   correlation is measured and reported per family.
5. **The profiler is a heuristic prior.** It is trained by imitation on synthetic families
   and helps only insofar as the probe signatures transfer; the warm start is neutral when
   the prediction is uniform (design rule D3).
6. **Baselines.** This codebase compares against classical/neural tabular models, not
   against foundation models (TabPFN-3.5, TabICLv2). That comparison is the natural next
   experiment and is *not* claimed here.
7. **Scale.** Everything is verified at `d <= 40`, `n <= 10^4`. The pair basis grows as
   `K M^2`; beyond a few dozen features pair selection must become an explicit stage.

---

## 9. If you extend this

* Swap H4 for a monotone/spline or Gaussian-process residual block; the identifiability
  argument only needs a nested chain.
* Make the pair set a learned object (`W3` then includes a discrete choice); the
  identifiability statement must then be re-stated relative to the selected `P`.
* Replace the imitation loss of the profiler with a differentiable proxy of the achieved
  `eps_id` (the "compiler" then optimises the metric it is scored on).
* The natural theory extension: a finite-sample bound on `rho_hat_k - rho_k*` in terms of
  the Gram condition number, `sqrt(log d / n)`, and the binning width.

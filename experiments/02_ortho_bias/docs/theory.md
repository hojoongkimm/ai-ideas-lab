# Theory appendix

Complete statements and proofs for the claims used in `README.md`. Notation follows the
README: `X ∈ R^d` with independent coordinates (A1), `Y = f(X) + eps`, `E[eps] = 0`,
`Var(eps) = sigma^2`, `f ∈ L^2(P_X)`; `P_W` is the orthogonal projection onto a closed
subspace `W ⊂ L^2(P_X)`; `E_S[·] = E[· | X_S]` for `S ⊆ [d]`.

---

## A1. The Hoeffding–Sobol decomposition

**Theorem 1.** Under (A1), define for every `S ⊆ [d]`

```
f_S(x_S) = sum_{T ⊆ S} (-1)^{|S| - |T|} E_T[f](x_T).
```

Then `f = sum_S f_S` in `L^2(P_X)`, the summands are mutually orthogonal,
`E[f_S | X_T] = 0` for all `T ⊊ S`, and the decomposition is unique up to a `P_X`-null set.
Consequently `Var(f) = sum_S Var(f_S)`.

*Proof.* Orthogonality: for `S ⊄ T`, pick `i ∈ S \ T`. Then
`E[f_S | X_T] = 0` (the alternating sum over `i ∈` or `∉` the subset cancels), so
`⟨f_S, g(X_T)⟩ = 0` for every `g ∈ L^2`. Taking `g = f_T` gives `⟨f_S, f_T⟩ = 0` whenever
`S ⊄ T`; if `T ⊄ S` the same argument with roles exchanged applies, and if `S = T` the inner
product is `Var(f_S) ≥ 0`. Completeness: for any `F ⊂ [d]`, `sum_{S ⊆ F} f_S = E_F[f]`
(inclusion–exclusion), so with `F = [d]` the sum is `E[f | X] = f` a.s. ∎

Uniqueness is what makes `S_S = Var(f_S)/Var(f)` a property of the target rather than of a
model. The estimator of `f_S` is what the blocks H1 and H2 compute: `f_{j}` is a
zero-mean function of `x_j`, `f_{jk}` is a function of `(x_j, x_k)` with zero
one-dimensional marginals.

---

## A2. Lemma 2 (structural centring ⇒ exact orthogonality)

Let `b_j : R → {1..M}` be quantile bins with `P(b_j = m) = 1/M`.

Define the parameter maps

```
(C1 theta)_j[m] = theta_j[m] - (1/M) sum_{m'} theta_j[m']
(C2 Theta)_jk[a,b] = Theta_jk[a,b] - (1/M) sum_{a'} Theta[a',b]
                      - (1/M) sum_{b'} Theta[a,b'] + (1/M^2) sum_{a',b'} Theta[a',b']
```

**Lemma 2.** *For any parameters, with `H1(x) = sum_j (C1 theta)_j[b_j(x_j)]` and
`H2(x) = sum_{jk} (C2 Theta)_jk[b_j(x_j), b_k(x_k)]`:*

*(a)* `E[H1] = 0`;
*(b)* `E[H2 | x_j] = 0` and `E[H2 | x_k] = 0` for every pair `(j,k)` in the sum, hence
`⟨H2, h(x_j)⟩ = 0` for every `h ∈ L^2(P_{X_j})` and `⟨H1, H2⟩ = 0`;
*(c)* `⟨T_jk, T_lm⟩ = 0` for distinct pairs `(j,k) ≠ (l,m)`;
*(d)* `⟨H1, H1'⟩` for two different parameter values, and the analogous cross terms, are
the ordinary inner products of the corresponding additive functions — i.e. `C1` is an
orthogonal projection of the parameter space, and so is `C2`.

*Proof.* (a) `E[H1] = sum_j (1/M) sum_m (C1 theta)_j[m] = 0` by construction of `C1`.

(b) `C2 = (I - E_j)(I - E_k)` acting on the table, where `E_j` averages over `a` with
weights `1/M`. Hence `E_{x_k}[H2 | x_j] = sum_jk (I-E_j)(I-E_k)Theta = 0` (apply `E_k` to
the second factor), and symmetrically. For any `h(x_j)`,
`⟨H2, h⟩ = E_{x_j}[ h(x_j) · E_{x_k}[H2 | x_j] ] = 0`. Since `H1` is a sum of functions of
single coordinates, `⟨H1, H2⟩ = 0`.

(c) For pairs sharing exactly one coordinate, say `(1,2)` and `(1,3)`:
`E[T_12 T_13] = E_{x_1}[ E_{x_2}[T_12] · E_{x_3}[T_13] ] = 0`. For disjoint pairs use
`E_{x_2}T_12 = 0`. Identical pairs are handled by (d).

(d) `C1` and `C2` are orthogonal projections with respect to the (uniform) counting inner
product on the tables, since they are compositions of conditional-expectation removals.
∎

**Corollary.** Under (A1), `H1` and `H2` are orthogonal in `L^2(P_X)` at *every* parameter
value, in particular at any point of the optimisation trajectory, and independently of the
batch used to evaluate them. This is the property that makes the low-order shares
estimable without post-hoc attribution.

*Remark (dependence).* If the coordinates are dependent, (b) still holds as a statement
about the *conditional* marginals under the uniform bin measure, but the mixed moments are
no longer zero; the empirical residualizer (Section A4) then restores orthogonality with
respect to the empirical measure, and the shares become those of the chosen chain rather
than Sobol indices.

---

## A3. Lemma 3 (nested projections)

**Lemma 3.** Let `W_1 ⊆ W_2 ⊆ ... ⊆ W_m` be closed subspaces and `Delta_k = P_{W_k} - P_{W_{k-1}}`
with `P_{W_0} = 0`. Then `Delta_k` is the orthogonal projection onto `W_k ⊖ W_{k-1}`,
`Delta_k Delta_j = 0` for `k ≠ j`, `sum_k Delta_k = P_{W_m}`, and for all `f`

```
||P_{W_m} f||^2 = sum_{k <= m} ||Delta_k f||^2 .
```

*Proof.* For `x ∈ W_k`, `P_{W_k} x = x` and `P_{W_{k-1}} x ∈ W_{k-1} ⊂ W_k`, so
`Delta_k x = x - P_{W_{k-1}} x ⟂ W_{k-1}`, i.e. `Delta_k x ∈ W_k ⊖ W_{k-1}`. For
`y ∈ W_k ⊖ W_{k-1}`: `P_{W_k} y = y` and `P_{W_{k-1}} y = 0`, so `Delta_k y = y`.
`Delta_k` is self-adjoint and idempotent, hence an orthogonal projection onto
`W_k ⊖ W_{k-1}`. Images of `Delta_k` and `Delta_j` (`k<j`) are orthogonal because
`W_k ⊖ W_{k-1} ⊆ W_{k-1}^⊥`... more directly, for `x ∈ W_k`, `Delta_j x = P_{W_j}x - P_{W_{j-1}}x`
and both terms lie in `W_j`, so `Delta_j x ∈ W_j`, while `Delta_k x ∈ W_k ⊖ W_{k-1}`; and
`⟨Delta_j x, Delta_k x⟩ = ⟨x, Delta_j Delta_k x⟩` with `Delta_j Delta_k = 0` since
`Delta_k(W) ⊂ W_k ⊆ W_{j-1}` for `k < j` and `Delta_j` annihilates `W_{j-1}`. The norm
identity follows from Pythagoras. ∎

**Frisch–Waugh–Lovell.** For `W_1 ⊆ W_2`, `P_{W_2} = P_{W_1} + P_{W_1^⊥ ∩ W_2}`. Hence for
any `h`, `h^perp := h - P_{W_1} h` satisfies `P_{W_2} h = P_{W_1} h + P_{W_2} h^perp`, i.e.
the *increment* of the projection equals the projection of the residualized block. This is
why the residualizer in `models/blocks.py` implements `Delta_k` when `Phi` spans `W_{k-1}`
and `h` is block `k`.

**Corollary (staged fitting is exact, not a heuristic).** If the blocks' design operators
`Phi_k` satisfy `⟨Phi_i, Phi_j⟩ = 0` (`i ≠ j`), the normal equations of the joint least
squares problem decouple into the per-block normal equations, so fitting block `k` on the
residual of blocks `< k` attains the joint optimum. This is the justification for the
staged schedule in `OrthoBiasTrainer.fit`.

---

## A4. Theorem 2 (identifiability) and its limit

**Setting.** Let `V_1 = span{1[b_j = m]}` (binned additive class), `V_2 = V_1 + span{1[b_j=a]1[b_k=b] :
(j,k) ∈ P}`, `V_3` a fixed tree class, `W = (V_1, V_2, V_2 + V_3, L^2)`, and let `Delta_k` be
as in Lemma 3. Define `rho_k = ||Delta_k f||^2 / ||f||^2`.

**Theorem 2.**

1. `Delta_k f` are mutually orthogonal and `sum_k ||Delta_k f||^2 = ||f||^2` (Lemma 3).
2. `Delta_1 f = P_{V_1} f` is the best additive (binned) approximation of `f`, so
   `||Delta_1 f||^2 = sum_j ||P_{V_1} f_j||^2 + O(sqrt(d)||f|| ||f - P_{V_1}f||)` and
   `rho_1 -> sum_j S_j` as `M -> inf` for Lipschitz shapes (Lemma 4).
3. `Delta_2 f = P_{V_2} f - P_{V_1} f` is the best binned pairwise increment; `rho_2` is the
   sum of the order-2 Sobol shares over the selected pairs, up to binning error.
4. `rho_3 + rho_4 = 1 - rho_1 - rho_2` is the higher-order share and is a property of `f`
   (given the bins and the pair set).
5. `rho_3 / rho_4` is **not** a property of `f`. Given `f` and `V_3`, `rho_3` is determined;
   but different admissible `V_3` give different values with the same
   `rho_1, rho_2, rho_3 + rho_4`.

*Proof of 5.* Let `h = f - P_{V_2} f` be the higher-order residual, `||h||^2 > 0`, and let
`V_3, V_3'` be two tree classes with `P_{V_3} h = h` and `P_{V_3'} h = 0` respectively
(e.g. `V_3` contains `h` — possible only if `h` is axis-aligned piecewise constant — and
`V_3'` is a class whose span is orthogonal to `h`, e.g. trees on a different, disjoint
feature set). Then `rho_3 = ||h||^2, rho_4 = 0` for the chain with `V_3`, and
`rho_3 = 0, rho_4 = ||h||^2` for the chain with `V_3'`, while `(rho_1, rho_2, rho_3+rho_4)`
is identical. Since both satisfy the identity of item 1, no data can distinguish them. ∎

**Consequence for the paper.** The only four-number statement the data can support is
`(rho_1, rho_2, rho_hi)`; the H3/H4 split must be reported as *relative to the tree class*
and validated comparatively (does the axis-aligned block take the high-order variance on
axis-aligned data?). `run_benchmark.py` scores `eps_id` on the identified triple and
reports `a3` separately.

**Remark (what would identify more).** Fixing `V_3` *and* requiring `Delta_3 f` to be the
`L^2` projection onto `V_3 ⊖ V_2` makes `rho_3` a well-defined functional of `(f, V_3)`.
Our estimator approximates exactly that, with two error sources: (i) `V_3` is a
*parameterised* class whose optimum is found by SGD, not exactly; (ii) the projection
operator is estimated from data (Section A5).

---

## A5. Finite-sample behaviour of the residualizer

Let `G = E[Phi Phi^T]` (`m x m`), `G_hat` its empirical estimate from `n_c` calibration
rows, and `P = Phi^T G^{-1} Phi` the population projection. With `||G_hat - G|| <= eps` and
`lambda_min(G) >= c > 0`, the Davis–Kahan/Wedin perturbation bound gives

```
||P_hat - P|| <= 2 eps / (c - eps),
```

so the residual correlation of a block with the basis is `O(eps/c) = O(sqrt(m/n_c))` for
sub-Gaussian rows. This is exactly what the benchmark measures: with `m ≈ 180` and
`n_c = 400` the observed deployed `|corr|` is `~0.1`; the exact per-sample projection is
`< 1e-7`. Two consequences are implemented:

* the design uses a **calibration split** for the coefficients (never the evaluation set),
  and reports the residual correlation rather than assuming it is zero;
* the soft penalty `L_ortho` fights the drift that a stale operator would otherwise allow.

**Why the coefficients cannot be a fixed pre-computed matrix.** `beta = G^{-1} E[Phi h]`
depends on the block's *function* `h`, which changes during training. A pre-computed matrix
can only remove a fixed function, which is why an early calibration (before the block has
taken shape) fails: the block then re-grows into the removed direction. The running-EMA
estimate tracks `h`; the exact projection (`mode="batch"`) is available for diagnostics but
makes the prediction depend on the batch, so it is not deployable.

---

## A6. Lemma 4 (binning error)

**Lemma 4.** Let `f_j : R → R` be `L`-Lipschitz and let `P` project onto the binned additive
class with bin width `h`. Then `||f_j - P f_j||_2^2 <= (Lh/2)^2`, and consequently
`||f - P_{V_1} f||^2 - sum_j ||f_j - P f_j||^2 = O(||f - sum_j f_j|| · ||f - P_{V_1}f||)`.

*Proof.* On a bin `[a, a+h)`, the best constant approximation is the conditional mean, and
`|f_j(x) - E[f_j | bin]| <= L h / 2` by Lipschitz continuity. Squaring and taking
expectations gives the bound; summing over `d` independent coordinates gives the additive
error, and the cross terms are bounded by Cauchy–Schwarz. ∎

Hence `rho_1(M) -> sum_j S_j` at rate `O(M^{-2})`; in practice `M = 8` leaves a few percent,
which is why an additive target shows a small non-zero `rho_hi` (reported in the smoke run).

---

## A7. Proposition 5 (exact-zero gate)

With `gamma = -0.1`, `zeta = 1.1` the deterministic gate is

```
z(alpha) = clip( sigmoid(alpha)(zeta - gamma) + gamma, 0, 1 )
```

`z(alpha) = 0` iff `sigmoid(alpha) · 1.2 - 0.1 <= 0` iff `sigmoid(alpha) <= 1/12` iff
`alpha <= log((1/12)/(11/12)) = log(1/11) ≈ -2.3979`. The code stores this boundary as
`blocks.ALPHA_PRUNE`; the additional check `HardConcreteGate.pruned()` tests `z(alpha) <= 0`
directly, so the two always agree.

*Proof of the expected L0.*
`P(z > 0) = P(s > -gamma/(zeta - gamma)) = P(s > 1/12)` with
`s = sigmoid((logit(u) + alpha)/beta)`. Solving gives
`P(z>0) = sigmoid(alpha - beta log(-gamma/zeta)) = sigmoid(alpha + 1.5986)`, so
`E[#open] = sum_k sigmoid(alpha_k + 1.5986)`, which is the L0 term of the objective. ∎

---

## A8. Proposition 6 (augmented Lagrangian)

**Proposition 6.** Let `F : R^p → R` and `g : R^p → R` be `C^2`, consider
`min F(theta) s.t. g(theta) <= 0`, and let `L_{mu,lambda}` be the PHR augmented Lagrangian
of Section 3.4. Suppose `{theta^k}` are stationary points of `L_{mu_k, lambda_k}` with
`mu_k -> inf` and `lambda^{k+1} = max(0, lambda^k + mu_k g(theta^k))`. If `grad g(theta*) ≠ 0`
for every limit point `theta*`, then `theta*` is a KKT point and
`lambda* = lim lambda^k >= 0` is the associated multiplier.

*Proof.* Standard (Rockafellar 1973, Thm. 4.2; Bertsekas 1982, Prop. 4.2.1): under the
linear independence constraint qualification the PHR multiplier iteration is locally
convergent, and `grad_theta L = grad F + lambda max(0, lambda + mu g)/... ` yields the KKT
system in the limit. ∎

**Remark (infeasible target).** If `{theta : IR(theta) >= tau}` is empty (tau above the
achievable interpretability ratio), `lambda^k -> inf` and `g(theta^k)` converges to
`max_g g < 0`, i.e. the iterates approach the *largest achievable* `IR`. The benchmark
reports the attained `IR` next to `tau` and never claims the constraint was met.

---

## A9. Proposition 7 (profiler invariance)

Let the encoder be `h = T( mean_j e(x_j), max_j e(x_j), g )` and the per-feature head
`logit_j = psi(e(x_j), h)`. For any permutation `pi` of the columns the multiset
`{e(x_j)}` is unchanged, so `h` is unchanged (mean and max are symmetric), and
`logit_{pi(j)}(x_pi) = logit_j(x)`. Hence `rho(X_pi) = rho(X)` and the screening output is
equivariant. The empirical check reports the floating-point residual (`~1e-8`). ∎

No sufficiency claim: `rho^(0)` is an amortized estimate whose error is measured on held-out
synthetic tasks (`results/meta_pretrain.md`), not a function of a sufficient statistic.

---

## A10. Why the objective has the form it has

* **Why MDL and not just weight decay.** A block's *value* depends on both its fit and its
  code length; `log2(1 + w^2/tau^2)` is the code length of a Gaussian code with scale `tau`,
  so `z_k MDL_k / n` is a per-sample description length and is commensurable with the
  per-sample task loss.
* **Why the L0 term is separate from MDL.** Pruning a block is a structural decision; the
  expected L0 makes the *decision* differentiable while MDL controls the *size* of blocks
  that remain open.
* **Why `IR` is constrained rather than penalised.** A penalty gives a single point on the
  accuracy/interpretability frontier for a fixed weight; the constraint gives the frontier
  point the *user asked for*, with a dual variable that reports whether it was attainable.

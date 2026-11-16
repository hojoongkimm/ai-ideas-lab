# 🧪 Experiment 01: DOENet (Design of Experiments Neural Network)

> **DOENet: Differentiable Orthogonal Factor Screening and Functional ANOVA for High-Dimensional Deep Learning**  
> **Target Venues:** NeurIPS, ICML, KDD, IEEE Transactions on Knowledge and Data Engineering (TKDE)

---

## 📌 1. Theoretical Motivation & Research Gap

### 🔍 Background & Motivation
In high-dimensional tabular, genomic, and industrial cyber-physical sensor data ($P \gg N$ or $P \ge 500$), deep neural networks encounter severe theoretical and empirical failure modes:
1. **Shortcut Learning & Spurious Memorization:** Overparameterized deep networks fit spurious correlations from collinear noise features rather than discovering invariant causal factors.
2. **Multicollinear Variance Inflation:** In the presence of correlated predictors ($\Sigma_{ij} \ne 0$), gradient-based feature attribution (e.g., Integrated Gradients, DeepSHAP) spreads importance across redundant groups, diluting sparse signal identification.
3. **Black-Box Opacity & Interaction Confounding:** Existing deep architectures fail to mathematically decouple main additive effects from complex multiplicative interactions.

### ⚔️ Comprehensive Comparison with Prior Art

| Method | Feature Selection Principle | Factor Orthogonality | Functional ANOVA | Heredity Constraint | High-Dim Scalability ($P \ge 500$) | End-to-End Differentiable |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard MLP** | ❌ None | ❌ None | ❌ None | ❌ None | ⚠️ High Overfitting | ✅ Yes |
| **LassoNet (JMLR 2021)** | ✅ Hierarchical L1 Skip | ❌ None | ❌ None | ❌ None | ⚠️ Collinear Redundancy | ✅ Yes |
| **TabNet (AAAI 2021)** | ✅ Sequential Sparsemax | ❌ None | ❌ None | ❌ None | ⚠️ Mask Drift | ✅ Yes |
| **GAMI-Net (PR 2021)** | ⚠️ Marginal Filtering | ❌ None | ✅ Spline ANOVA | ⚠️ Heuristic | ❌ $O(P^2)$ Combinatorial Explosion | ❌ Multi-stage |
| **NAM (NeurIPS 2021)** | ❌ None | ❌ None | ⚠️ Additive Only | ❌ No Interactions | ⚠️ Parameter Overhead | ✅ Yes |
| **DOENet (Ours)** | **✅ Hard Concrete Gate** | **✅ Orthogonal Arrays** | **✅ Functional ANOVA** | **✅ Strong Heredity Gate** | **✅ $O(K^2)$ Scalable ($K \ll P$)** | **✅ Fully End-to-End** |

---

## 📐 2. Mathematical Formulation & Architecture

### A. Differentiable Hard Concrete Factor Screening
For an input vector $x \in \mathbb{R}^P$, we assign a continuous relaxation of binary feature inclusion gates $z_i \in \{0, 1\}$ using the Hard Concrete distribution (Louizos et al., ICLR 2018):

$$
u_i \sim \text{Uniform}(0, 1)
$$

$$
s_i = \sigma\left( \frac{\log u_i - \log(1 - u_i) + \log \alpha_i}{\tau} \right)
$$

$$
\bar{s}_i = \text{clamp}\left( s_i (\zeta - \gamma) + \gamma, 0, 1 \right)
$$

$$
\tilde{x} = x \odot \bar{s}
$$

where $\gamma = -0.1$ and $\zeta = 1.1$ stretch the support of the distribution to include exact $0$ and $1$ with non-zero probability mass.

**Expected $L_0$ Norm Regularization:**  
The expected number of active factors has an exact closed-form differentiable expression:

$$
\mathcal{L}_{L_0}(\log \alpha) = \sum_{j=1}^P \mathbb{P}(\bar{s}_j > 0) = \sum_{j=1}^P \sigma\left( \log \alpha_j - \tau \log(-\gamma / \zeta) \right)
$$

An exponential temperature annealing schedule $\tau_t = \max(\tau_{\min}, \tau_0 \cdot \exp(-\kappa t))$ guarantees continuous gradient propagation early in training and discrete factor pruning upon convergence.

### B. DOE Orthogonal Factor Regularization
In traditional Design of Experiments (DOE), **Orthogonal Arrays** ($L_N(s^k)$) ensure that factor columns have zero covariance, enabling independent estimation of main effects without confounding. To instill this principle into deep networks:

$$
\mathcal{L}_{\text{ortho}}(W_{\text{proj}}) = \frac{1}{K} \| W_{\text{proj}} W_{\text{proj}}^\top - I_K \|_F^2
$$

where $W_{\text{proj}} \in \mathbb{R}^{K \times P}$ projects high-dimensional inputs onto $K$ orthogonal factor subspaces. This penalty penalizes co-selection of collinear features and forces the network to retain non-redundant informative dimensions.

### C. Hierarchical Functional ANOVA with Strong Heredity
Under Functional ANOVA decomposition (Sobol, 1993), an arbitrary integrable function $f(x)$ is uniquely decomposed as:

$$
f(x) = f_0 + \sum_{i} f_i(x_i) + \sum_{i < j} f_{ij}(x_i, x_j) + f_{\text{residual}}(x)
$$

subject to the orthogonality and zero-mean conditions: $\mathbb{E}[f_i(X_i)] = 0$ and $\mathbb{E}[f_{ij}(X_i, X_j) \mid X_i] = 0$.

In DOENet:

1. **Main Effects:** Modeled via parallel 1D nonlinear subnets with batch zero-mean centering:

$$
\hat{f}_i(x_i) = f_i(\tilde{x}_i) - \frac{1}{B}\sum_{b=1}^B f_i(\tilde{x}_{b,i})
$$

2. **Strong Heredity Gating:** In factorial design theory (Chipman, 1996; Wu & Hamada, 2009), an interaction effect $x_i x_j$ can only be active if **both** parent factors have significant main effects: $m_{ij} = \bar{s}_i \cdot \bar{s}_j$. We parameterize pairwise interactions via low-rank bilinear factorization scaled by the heredity mask:

$$
f_{\text{inter}}(\tilde{x}) = \sum_{r=1}^R (u_r^\top \tilde{x})(v_r^\top \tilde{x}) \cdot (\bar{s} \bar{s}^\top)
$$

This bounds interaction exploration from $O(P^2)$ down to $O(K^2)$ where $K = \sum_j \mathbf{1}(\bar{s}_j > 0) \ll P$.

3. **Deep Residual Head:** A deep MLP captures multi-way higher-order interactions ($>2$) on the screened representation.

### D. Unified End-to-End Objective

$$
\min_{\Theta} \mathcal{L}_{\text{task}}(y, \hat{y}) + \lambda_{\text{sparse}} \mathcal{L}_{L_0}(\log \alpha) + \lambda_{\text{ortho}} \mathcal{L}_{\text{ortho}}(W_{\text{proj}})
$$

---

## 📊 3. Rigorous Benchmark Protocol

### Synthetic Data Generation ($P=500, N=2500$)
- **True Factors (0..9):** Non-linear main effects (sinusoidal, quadratic, exponential) + XOR pairwise interactions.
- **Multicollinear Confounders (10..49):** High correlation ($\rho = 0.75$) with true factors to test collinear robustness.
- **Pure Noise (50..499):** $450$ uninformative Gaussian noise features.

### Statistical Validation Methodology
- **Multi-Seed Evaluation:** All models evaluated across $5$ independent random seeds with identical data splits.
- **Statistical Significance Testing:** Paired Student's t-test and Wilcoxon signed-rank test to verify $p < 0.01$ improvements.
- **Metrics Reported (Mean $\pm$ Std):**
  - Predictive Accuracy: Test RMSE, Test $R^2$
  - Screening Fidelity: Precision, Recall, F1-Score, False Discovery Proportion (FDP)
  - Variance Attribution: ANOVA Main % vs. Interaction % vs. Residual %

---

## 🚀 4. Execution Guide

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete 5-seed benchmark with statistical hypothesis testing
python train_benchmark.py --dim 500 --samples 2500 --epochs 80 --num_seeds 5
```


---

## 🔧 5. Implementation Notes (branch `hj-benchmark`)

Fixes applied before the real-data benchmark (see commit history for diffs):

| Component | Issue | Fix |
| :--- | :--- | :--- |
| `OrthogonalFactorProjection` | $W_{\text{proj}}$ was never used in `forward`, so $\mathcal{L}_{\text{ortho}}$ did not affect predictions | The $K$ orthogonal factors $W_{\text{proj}}\tilde{x}$ now feed the deep residual head (`use_ortho_proj=True`, default). `use_ortho_proj=False` keeps the original behaviour for ablation |
| Main-effect centering | Batch-mean centering was also applied at inference, so predictions depended on test batch size/composition | Running mean $\hat{\mathbb{E}}[f_i(X_i)]$ is tracked during training and used at inference |
| Multi-class tasks | Only regression / binary output | `out_features = C` with cross-entropy on logits |
| Feature ranking | Clamped gates tie at 1.0, making top-$k$ order arbitrary | `get_feature_importance` ranks by $\log\alpha$ |
| `GatedMLP_Ablation` | No temperature annealing (unlike DOENet), binary only | Same annealing schedule as DOENet, multi-class support |
| `train_benchmark.py` | `.numpy()` on tensors requiring grad crashed the GatedMLP / L1 paths | Feature extraction runs under `torch.no_grad()` |

---

## 🧪 6. Real-Data Feature-Selection Benchmark

Standard supervised feature-selection benchmark shared by CAE (ICML'19), LassoNet (JMLR'21) and Sequential Attention (ICLR'23): **Mice Protein, MNIST, Fashion-MNIST, ISOLET, COIL-20, Activity**, $k = 50$ selected features, seeds 1–5.

- **Data & splits:** `bench/prepare_sa_data.py` re-implements the Sequential Attention data pipeline (same loaders, preprocessing, train/val/test splits per seed).
- **Evaluation:** every method ranks all features; the top-50 are given to the *same* downstream classifier (input BN → 67 ReLU → softmax, SA batch sizes/epochs), trained from scratch; test accuracy is reported as mean ± SD over 5 seeds.
- **Methods:** DOENet, DOENet without orthogonal projection, GatedMLP ablation, STG, LassoNet, TabNet, random-$k$, all features.
- **Full protocol** (fixed before running, incl. deviations from the reference): [`bench/PROTOCOL.md`](./bench/PROTOCOL.md)

```bash
pip install -r requirements.txt -r requirements-benchmark.txt

# 1) download + preprocess the 6 datasets (seeds 1-5)
python bench/prepare_sa_data.py --out_dir DATA_DIR

# 2) run the benchmark (resumable; one JSON line per dataset x method x seed)
python run_fs_benchmark.py --data_dir DATA_DIR --out results.jsonl \
    --datasets mice,coil20,isolet,activity,mnist,fashion --seeds 1 2 3 4 5
```

Status: full run in progress. Results will be added after independent verification.

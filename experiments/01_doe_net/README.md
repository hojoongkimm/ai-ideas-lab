# 🧪 Experiment 01: DOENet (Design of Experiments Neural Network)

> **DOENet: Differentiable Orthogonal Factor Screening and Functional ANOVA for High-Dimensional Deep Learning**  
> **Authors:** Ho-joong Kim (`@hojoongkimm`) & Pickle  
> **Target Venues:** NeurIPS, ICML, KDD, IEEE Transactions on Knowledge and Data Engineering (TKDE) / IEEE T-ASE

---

## 📌 1. Research Motivation & Novelty Analysis

### 🔍 Research Gap
In high-dimensional tabular and industrial sensor data ($P \gg N$ or $P \ge 500$), deep neural networks suffer severely from:
1. **Shortcut Learning & Overfitting:** Models memorize spurious correlations from noise features.
2. **Multicollinear Redundancy:** Correlated noise variables dilute attention and gradient signals.
3. **Black-box Opacity:** Inability to statistically attribute variance to individual main effects vs. complex interactions.

### ⚔️ Comparison with Prior Art

| Method | Feature Selection | Orthogonality (DOE) | ANOVA Decomposition | Heredity Principle | High-Dim Scale ($P \ge 500$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Standard MLP** | ❌ None | ❌ None | ❌ None | ❌ None | ⚠️ High Overfitting |
| **LassoNet (JMLR 2021)** | ✅ Linear Skip L1 | ❌ None | ❌ None | ❌ None | ⚠️ Collinear Redundancy |
| **TabNet (AAAI 2021)** | ✅ Sequential Attention | ❌ None | ❌ None | ❌ None | ⚠️ Heuristic Masking |
| **GAMI-Net / NAM (NeurIPS 2021)** | ⚠️ Limited | ❌ None | ✅ Additive Splines | ⚠️ Manual Selection | ❌ $O(P^2)$ Explosion |
| **DOENet (Ours)** | **✅ Hard Concrete Gate** | **✅ Orthogonal Arrays** | **✅ Functional ANOVA** | **✅ Strong Heredity Gate** | **✅ End-to-End Scalable** |

---

## 📐 2. Mathematical Formulation

### A. Differentiable Hard Concrete Screening
We assign a continuous relaxation of binary gates $z_i \in [0, 1]$ to each feature $i \in \{1, \dots, P\}$:
$$\bar{s}_i = \text{clamp}\left( \sigma\left(\frac{\log u - \log(1-u) + \log \alpha_i}{\tau}\right)(\zeta - \gamma) + \gamma, 0, 1 \right)$$
$$\tilde{x} = x \odot \bar{s}$$

### B. DOE Orthogonality Regularization
To mimic the non-redundant factor exploration of Orthogonal Arrays in Design of Experiments:
$$\mathcal{L}_{\text{ortho}} = \frac{1}{K} \| W_{\text{proj}} W_{\text{proj}}^\top - I_K \|_F^2$$
This drives the network to select linearly independent, maximally informative factors rather than clusters of collinear variables.

### C. Hierarchical Functional ANOVA with Strong Heredity
The prediction is structurally decomposed into:
$$\hat{y} = f_0 + \sum_{i \in \mathcal{S}} f_i(\tilde{x}_i) + \sum_{i < j \in \mathcal{S}} f_{ij}(\tilde{x}_i, \tilde{x}_j) \cdot (\bar{s}_i \bar{s}_j) + f_{\text{residual}}(\tilde{x})$$
- **Main Effects:** 1D nonlinear feature mappings.
- **Pairwise Interactions:** Low-rank bilinear representation gated by $\bar{s}_i \bar{s}_j$ (Strong Heredity: an interaction is active if and only if both factors are active).
- **Deep Residual Head:** Handles higher-order multi-way nonlinearities.

### D. Unified End-to-End Objective
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}}(y, \hat{y}) + \lambda_{\text{sparse}} \mathcal{L}_{L0}(\log \alpha) + \lambda_{\text{ortho}} \mathcal{L}_{\text{ortho}}(W_{\text{proj}})$$

---

## 📊 3. Benchmark Experiment Protocol

- **Dataset:** Synthetic high-dimensional benchmark ($N=2500, P=500$).
  - **Features 0..9:** True causal factors (linear, quadratic, sinusoidal, exponential, XOR interaction).
  - **Features 10..49:** Collinear confounding variables ($75\%$ correlated with true features).
  - **Features 50..499:** Pure Gaussian noise ($450$ distractor features).
- **Compared Models:**
  1. `DOENet` (Full Architecture)
  2. `GatedMLP` (Ablation without Orthogonality and ANOVA)
  3. `L1SparseMLP` (LassoNet-style baseline)
  4. `StandardMLP` (Vanilla Deep MLP)
- **Evaluation Metrics:**
  - Prediction Quality: Test RMSE, Test $R^2$
  - Screening Accuracy: Precision, Recall, F1-Score, False Discovery Proportion (FDP)
  - Explainability: ANOVA Variance Attribution ($SS_{\text{main}} / SS_{\text{total}}$, $SS_{\text{inter}} / SS_{\text{total}}$)

---

## 🚀 4. How to Run

### Command-Line Benchmark
```bash
# In the experiment directory
cd experiments/01_doe_net
pip install -r requirements.txt

# Run benchmark across all 4 models
python train_benchmark.py --dim 500 --epochs 80 --batch_size 64
```

### Interactive Jupyter / VS Code Notebook
Open `doe_net_benchmark.ipynb` via VS Code Remote-SSH on your GPU server to run step-by-step visualizations and inspect ANOVA plots.

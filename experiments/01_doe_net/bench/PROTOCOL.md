# DOENet — Supervised Feature-Selection Benchmark Protocol (fixed before running)

Written and committed before any full-run result was seen. Any later change is listed
under "Change log" with its reason and date; results produced under an earlier version
are kept and labelled.

## 1. Question
On the standard neural feature-selection benchmark (CAE ICML'19, LassoNet JMLR'21,
Sequential Attention ICLR'23), how good are the k = 50 features selected by DOENet
compared with neural feature-selection / tabular baselines, when every method's
features are evaluated by the same downstream classifier?

## 2. Data and splits — identical to the Sequential Attention (SA) reference code
`bench/prepare_sa_data.py` re-implements `google-research/sequential_attention/experiments/
datasets/{data_loader,dataset}.py` with numpy and reproduces its numpy random-call order.

| dataset | P | C | train / val / test | split |
|---|---|---|---|---|
| Mice Protein | 77 | 8 | 756 / 108 / 216 | seed-dependent 80/20 shuffle, then val = 12.5 % of train |
| MNIST | 784 | 10 | 52,500 / 7,500 / 10,000 | official train/test, val = 12.5 % of train |
| Fashion-MNIST | 784 | 10 | 52,500 / 7,500 / 10,000 | official train/test |
| ISOLET | 617 | 26 | 5,459 / 779 / 1,559 | isolet1-4 / isolet5 |
| COIL-20 | 400 (20x20) | 20 | 1,008 / 144 / 288 | seed-dependent 80/20 |
| Activity (UCI HAR) | 561 | 6 | 6,433 / 919 / 2,947 | official train/test |

Seeds: 1, 2, 3, 4, 5 (SA `experiments.py`). k = 50 for every dataset.
Kept reference quirks (for comparability with published numbers): Mice imputation includes
the -100000 sentinel (noted by IP-CAE, ICML'24); min-max scaling is fit on train+test.

## 3. Methods (each ranks all P features using train, val for early stopping only)
| id | ranking score | settings |
|---|---|---|
| `doenet` | gate `log_alpha` | this repo, branch `hj-benchmark`: orthogonal projection wired into residual head; repo defaults (epochs 80, batch 64, AdamW lr 2e-3 wd 1e-4, cosine, λ_sparse 2e-3, λ_ortho 1e-2, K=24 factors, hidden 64, rank 16, temperature 0.66→0.1); checkpoint = best val accuracy |
| `doenet_noortho` | gate `log_alpha` | same, `use_ortho_proj=False` (original architecture: W_proj unused in forward) |
| `gated_mlp` | gate `log_alpha` | repo ablation (Hard Concrete gate + MLP), same training as DOENet |
| `stg` | gate μ | stg 0.1.2 (Yamada et al. ICML'20) defaults: hidden [400,200], ReLU, σ 0.5, λ 0.1, batch 100, wd 1e-3; lr 1e-3 (library default 1e-5 is a placeholder), 100 epochs, per-epoch shuffling |
| `lassonet` | `feature_importances_` (λ at which a feature leaves the path) | lassonet 0.0.22 defaults: hidden (100,), M 10, full path, val set for early stopping |
| `tabnet` | `feature_importances_` | pytorch-tabnet 4.1.0 defaults, max 100 epochs, patience 10, batch 1024 |
| `random` | random permutation (seed 1000+s) | lower reference |
| `all_features` | — (all P features) | upper reference |

No method is tuned on test data. Baselines use library defaults; DOENet uses repo defaults.

## 4. Downstream evaluation (identical for every method)
SA `SparseModel`: input BatchNorm → Dense(67, ReLU) → Dense(C, softmax), Adam, cross-entropy.
Batch size and epochs per dataset from SA `experiments.py` / `hyperparams_sa.py`
(batch: mice 16, mnist 349, fashion 391, isolet 23, coil 451, activity 183;
epochs: 250, 50, 54, 80, 166, 134). Last incomplete batch dropped (as in the reference).
Learning rate ∈ {1e-3, 3e-3, 1e-2}, chosen by validation accuracy of the final-epoch model;
the reported number is the test accuracy of that final-epoch model. Runs on CPU (1 thread).

## 5. Metrics and analysis (pre-specified)
- Primary: downstream test accuracy with the top-50 features; mean ± sample SD (ddof=1) over 5 seeds.
- Secondary: each trainable method's own test accuracy with all features; number of active gates
  (gate > 0.1) for DOENet; ANOVA variance shares for DOENet; selection wall time.
- Per dataset: paired t-test over the 5 seeds, `doenet` vs each other method (two-sided);
  report p-values without multiplicity claims (n = 5 per test).
- Across datasets: average rank of each method (rank by mean accuracy per dataset, ties averaged);
  Wilcoxon signed-rank test over the 6 dataset means, `doenet` vs each baseline.
- Ablation: `doenet` vs `doenet_noortho` vs `gated_mlp`.
- Published numbers (SA Table 2, LassoNet Table 1, IP-CAE Table 2) are shown as external
  references only and are not mixed into the statistical tests: their downstream training,
  repetitions and (for LassoNet/IP-CAE) data sizes differ.

## 6. Known deviations from the SA reference
1. Downstream learning rate: SA's tuned per-dataset lr/decay are not used because
   `decay_steps = 0` (mice, fashion) makes TF's ExponentialDecay undefined; replaced by the
   val-selected grid above, applied identically to all methods.
2. Full per-epoch shuffling instead of `tf.data.shuffle(100)`.
3. PyTorch instead of TensorFlow; COIL-20 resized with Pillow 12.0.0 default resampling.

## 7. Failure handling
A failed cell (exception) is retried once with the same settings; if it fails again it is
reported as missing with the error. Nothing is silently dropped.

## Change log
- (none yet)

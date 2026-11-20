# OrthoBias benchmark report

configuration: {
  "smoke": true,
  "families": [
    "main",
    "pairwise",
    "tree",
    "deep",
    "hybrid"
  ],
  "seeds": [
    0
  ],
  "n_train": 1200,
  "d": 8,
  "noise": 0.1,
  "epochs": 60,
  "batch": 256,
  "lr": 0.003,
  "lr_deep": 0.001,
  "calibrate": false,
  "n_bins": 8,
  "trees": 16,
  "tree_depth": 4,
  "hidden": 96,
  "res_blocks": 2,
  "max_pairs": 4,
  "lam_l0": 0.001,
  "lam_mdl": 0.001,
  "lam_ortho": 0.01,
  "taus": [
    0.7,
    0.9
  ],
  "no_staging": false,
  "ablations": true,
  "tau_sweep": true,
  "check_invariance": true,
  "prune_check": true,
  "prune_tol": 0.3,
  "skip_baselines": true,
  "only": "oracle,ablations,prune,invariance",
  "baselines_only": false,
  "profiler": "checkpoints/profiler_smoke.pt",
  "out": "results_smoke"
}



## 1. Oracle recovery and predictive accuracy

| family | R2 OrthoBias | err(ident) | err(main) | err(pair) | err(high) | attribution H3 |
|---|---|---|---|---|---|---|
| main | 0.892 | 0.214 | 0.107 | 0.013 | 0.094 | 0.354 |
| pairwise | 0.810 | 0.143 | 0.071 | 0.045 | 0.026 | 0.194 |
| tree | 0.501 | 0.224 | 0.058 | 0.112 | 0.054 | 0.828 |
| deep | 0.754 | 0.383 | 0.036 | 0.192 | 0.156 | 0.284 |
| hybrid | 0.781 | 0.516 | 0.258 | 0.131 | 0.127 | 0.348 |

Ground-truth reference shares (first seed of each family):

| family | rho* (H1/H2/H3/H4) | rho_hat |
|---|---|---|
| main | [0.873, 0.015, 0.0, 0.112] | [0.98, 0.002, 0.006, 0.012] |
| pairwise | [0.084, 0.827, 0.015, 0.074] | [0.155, 0.782, 0.012, 0.051] |
| tree | [0.052, 0.141, 0.0, 0.807] | [0.11, 0.029, 0.713, 0.148] |
| deep | [0.12, 0.449, 0.0, 0.431] | [0.155, 0.258, 0.166, 0.42] |
| hybrid | [0.422, 0.399, 0.0, 0.179] | [0.679, 0.268, 0.018, 0.034] |

## 2. Orthogonality audit (|corr| between block outputs)

| family | raw max | exact-projection max | deployed max | raw mean | exact mean | deployed mean |
|---|---|---|---|---|---|---|
| main | 0.392 | 1.1e-08 | 0.118 | 0.179 | 6.0e-09 | 0.065 |
| pairwise | 0.745 | 1.9e-08 | 0.552 | 0.429 | 8.9e-09 | 0.194 |
| tree | 0.530 | 1.6e-08 | 0.320 | 0.130 | 6.6e-09 | 0.134 |
| deep | 0.756 | 2.0e-08 | 0.285 | 0.379 | 6.6e-09 | 0.134 |
| hybrid | 0.591 | 1.9e-08 | 0.423 | 0.361 | 9.3e-09 | 0.133 |

## 3. Ablations

| family | variant | R2 | err(ident) | IR | max block corr |
|---|---|---|---|---|---|
| deep | full | 0.693 | 0.290 | 0.732 | 0.231 |
| deep | no-ortho | 0.659 | 0.212 | 0.821 | 0.779 |
| deep | no-meta | 0.787 | 0.635 | 0.644 | 0.244 |
| hybrid | full | 0.783 | 0.492 | 0.970 | 0.328 |
| hybrid | no-ortho | 0.765 | 0.499 | 0.975 | 0.665 |
| hybrid | no-meta | 0.828 | 0.033 | 0.932 | 0.281 |
| main | full | 0.878 | 0.227 | 0.995 | 0.168 |
| main | no-ortho | 0.867 | 0.231 | 0.998 | 0.289 |
| main | no-meta | 0.918 | 0.044 | 0.924 | 0.265 |
| pairwise | full | 0.763 | 0.160 | 0.976 | 0.400 |
| pairwise | no-ortho | 0.827 | 0.075 | 0.945 | 0.853 |
| pairwise | no-meta | 0.863 | 0.098 | 0.975 | 0.213 |
| tree | full | 0.508 | 0.218 | 0.935 | 0.314 |
| tree | no-ortho | 0.513 | 0.216 | 0.678 | 0.718 |
| tree | no-meta | 0.562 | 0.319 | 0.970 | 0.332 |

## 5. Exact-zero block pruning (linear target, only H1 needed)

- gate parameters alpha: [1.55, -10.0, 1.45, 1.43]
- evaluation gates: [0.889, 0.0, 0.872, 0.868]
- exactly zero gates: [False, True, False, False]
- pruned blocks (index): [1]
- R2 0.9793 -> 0.9771, IR 0.971 -> 0.971, tolerance 0.3

## 6. Invariance check

- `profiler_rho_maxdiff` = 2.98e-08
- `profiler_block_logit_maxdiff` = 7.45e-09
- `profiler_screening_setdiff` = 1.86e-09

## 7. Baseline models (same optimiser/budget protocol)

| family | MLP | ResNet(H4) | NAM | SoftTree(H3) |
|---|---|---|---|---|
| main | 0.971 | 0.966 | 0.989 | 0.972 |
| pairwise | 0.976 | 0.962 | -0.029 | 0.909 |
| tree | 0.292 | 0.425 | -0.011 | 0.708 |
| deep | 0.978 | 0.965 | 0.012 | 0.687 |
| hybrid | 0.917 | 0.895 | 0.459 | 0.853 |

## 4. Interpretability constraint sweep

| family | tau | R2 | IR | pruned blocks |
|---|---|---|---|---|
| deep | 0.70 | 0.724 | 0.631 | 0.000 |
| deep | 0.90 | 0.808 | 0.402 | 0.000 |
| hybrid | 0.70 | 0.746 | 0.992 | 0.000 |
| hybrid | 0.90 | 0.774 | 0.980 | 0.000 |
| main | 0.70 | 0.888 | 0.990 | 0.000 |
| main | 0.90 | 0.876 | 0.996 | 0.000 |
| pairwise | 0.70 | 0.754 | 0.981 | 0.000 |
| pairwise | 0.90 | 0.808 | 0.952 | 0.000 |
| tree | 0.70 | 0.512 | 0.804 | 0.000 |
| tree | 0.90 | 0.447 | 0.477 | 0.000 |

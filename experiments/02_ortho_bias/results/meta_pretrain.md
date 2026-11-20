# Meta-profiler pre-training

- tasks: 10 train / 4 val, epochs 6
- checkpoint: `checkpoints/profiler_smoke.pt`

| metric | untrained | trained |
|---|---|---|
| rho_l1 | 0.3097 | 0.0571 |
| rho_corr_block1 | 0.9601 | -0.8484 |
| rho_corr_block2 | 0.3076 | -0.7799 |
| rho_corr_high | 0.0909 | 0.4734 |
| feature_screen_acc | 0.5152 | 0.8182 |
| pair_screen_acc | 0.9583 | 0.9583 |

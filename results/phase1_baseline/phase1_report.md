# Phase 1 Report — Baseline Reproduction

**Generated:** 2026-09-10 10:31:24

## Exit Criteria Checklist

- [x] DQN baseline runs end-to-end without errors across 5 seeds
- [x] Fixed-Time and Max Pressure reference numbers recorded (5/5 seeds)
- [x] All metrics logged as mean ± std across all 5 seeds
- [x] Final Phase 1 Report verified
- [x] Frozen-policy evaluation run (epsilon=0.0) and cross-checked against training-curve numbers (within ~5%, consistent direction)

## Results Table (mean ± std across 5 seeds)

| Metric | DQN (Training, eps 91-100) | DQN (Frozen Eval, eps=0.0) | Fixed-Time | Max Pressure |
|--------|-----------------------------|----------------------------|------------|--------------|
| Avg Waiting Time (s/step) | 30.3389 ± 1.7735 | 28.9320 ± 2.4285 | 236.6550 ± 8.3341 | 924.3029 ± 137.0757 |
| Avg Delay (s/step) | 11.7105 ± 1.1792 | 11.0567 ± 1.3232 | 190.6523 ± 6.5282 | 883.0683 ± 137.0032 |
| Avg Queue Length (veh) | 2.3599 ± 0.1461 | 2.2726 ± 0.2370 | 22.7890 ± 0.8025 | 90.0537 ± 13.7591 |
| Sum Negative Reward | -4916.8600 ± 483.5883 | -4693.4000 ± 660.9231 | -99139.2000 ± 3394.6767 | -464607.6000 ± 74137.4830 |

## Notes

- DQN trained for 100 episodes per seed (5 seeds total)
- DQN metrics are from last 10 episodes (post-convergence)
- Baselines run for 1 episode per seed (deterministic, no training)
- State: 80-cell binary vector (lane occupancy)
- Action space: 4 green phases
- Reward: -Δ(cumulative waiting time)
- Traffic demand: Weibull-distributed arrivals, 1000 cars/episode
- Max steps: 5400 steps/episode

## Files Produced

```
results/phase1_baseline/
  dqn_seed_{1..5}.csv                    — per-episode DQN metrics per seed
  dqn_frozen_eval.csv                    — 5-seed deterministic frozen-policy evaluation metrics
  checkpoints/seed_{1..5}/trained_model.pt — saved PyTorch model weights per seed
  summary_dqn.csv                        — DQN aggregated mean/std
  fixedtime_seeds.csv                    — Fixed-Time results
  maxpressure_seeds.csv                  — Max Pressure results
  summary_baselines.csv                  — Baseline aggregated mean/std
  summary_all.csv                        — Combined summary
  environment.txt                        — Software versions
  phase1_report.md                       — This document
  plots/
    dqn_learning_curves.png
    controller_comparison.png
```

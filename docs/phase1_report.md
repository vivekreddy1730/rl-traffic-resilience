# Phase 1 Report (v2) — Baseline Reproduction & Max-Pressure Correction

**Date:** September 29, 2026  
**Status:** COMPLETE (Fully verified against disk evidence)  
**Target Directory:** `results/phase1_baseline_v2/`

---

## 1. Executive Summary & Purpose of v2
Phase 1 establishes the baseline performance of multi-seed Deep Q-Network (DQN) reinforcement learning against two traditional reference controllers: Fixed-Time and Max-Pressure.

In v1 of Phase 1, an anomaly was discovered: Max-Pressure performed ~3.9x worse than Fixed-Time (924.30s vs 236.66s waiting time). Diagnostic tracing on Seed 1 revealed a severe bug: `ACTION_TO_EDGES` aggregated queues at the edge level rather than lane level, creating identical pressures for straight (actions 0 & 2) and left-turn (actions 1 & 3) phases. Python's built-in `max()` dictionary tie-breaking always picked the first encountered key (0 or 2), resulting in **Actions 1 and 3 being selected 0 times across the entire episode**. Left-turning vehicles (25% of all traffic) were completely starved, remaining permanently trapped in the intersection.

In Phase 1 v2:
1. `MaxPressureController` was corrected in `baselines/run_baselines.py` to evaluate pressure at the lane level (`ACTION_TO_LANES`) and break ties cyclically/fairly.
2. Baselines were re-run across all 5 seeds into `results/phase1_baseline_v2/` without touching DQN training code or overwriting historical v1 evidence.
3. With the bug resolved, Max-Pressure waiting time dropped from 924.30s to 172.95s (an 81.3% improvement), properly outperforming Fixed-Time (236.66s).

---

## 2. Exit Criteria Checklist

- [x] DQN baseline runs end-to-end without errors across 5 seeds (`dqn_seed_1.csv` to `dqn_seed_5.csv`, 101 lines each)
- [x] Fixed-Time reference numbers recorded across 5 seeds (`fixedtime_seeds.csv`, 6 lines)
- [x] Max-Pressure reference numbers recorded with verified lane-level evaluation across 5 seeds (`maxpressure_seeds.csv`, 6 lines)
- [x] Every baseline controller actively uses all 4 actions/phases at least once
- [x] All metrics logged as mean ± std across all 5 seeds (`summary_all.csv`, `summary_baselines.csv`)
- [x] Frozen-policy evaluation run ($\epsilon = 0.0$) and cross-checked against training-curve numbers (28.93s vs 30.34s, consistent within 4.6%)
- [x] Verification script `verify_phase1_v2.py` passes all automated checks and returns exit code 0

---

## 3. Results Table (Mean ± Std Across 5 Seeds)

| Metric | DQN (Training, eps 91-100) | DQN (Frozen Eval, $\epsilon=0.0$) | Fixed-Time Baseline | Corrected Max-Pressure (v2) | Historical Max-Pressure (v1 Buggy) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Avg Waiting Time (s/step)** | 30.3389 ± 1.7735 | 28.9320 ± 2.4285 | 236.6550 ± 8.3341 | **172.9483 ± 7.1827** | 924.3029 ± 137.0757 |
| **Avg Delay (s/step)** | 11.7105 ± 1.1792 | 11.0567 ± 1.3232 | 190.6523 ± 6.5282 | **138.1546 ± 4.5393** | 883.0683 ± 137.0032 |
| **Avg Queue Length (veh)** | 2.3599 ± 0.1461 | 2.2726 ± 0.2370 | 22.7890 ± 0.8025 | **16.7387 ± 0.7309** | 90.0537 ± 13.7591 |
| **Sum Negative Reward** | -4,916.86 ± 483.59 | -4,693.40 ± 660.92 | -99,139.20 ± 3,394.68 | **-72,200.60 ± 2,419.98** | -464,607.60 ± 74,137.48 |

---

## 4. Key Findings & Controller Analysis
1. **DQN vs Baselines**:
   - The learned DQN policy achieves superior performance across all traffic congestion metrics.
   - Frozen DQN ($\epsilon=0.0$) reduces average waiting time by **87.8% relative to Fixed-Time** (28.93s vs 236.66s) and **83.3% relative to Corrected Max-Pressure** (28.93s vs 172.95s).
2. **Corrected Max-Pressure vs Fixed-Time**:
   - Corrected Max-Pressure reduces waiting time by **26.9% relative to Fixed-Time** (172.95s vs 236.66s).
   - This aligns with transport engineering literature: adaptive pressure relief outperforms rigid pre-timed cyclic scheduling under Weibull traffic arrivals.
3. **Action Distribution Verification**:
   - Under corrected Max-Pressure on Seed 1, action counts were: Action 0: 181, Action 1: 117, Action 2: 156, Action 3: 55.
   - All 4 signal phases (including left turns) are actively and dynamically scheduled.

---

## 5. Artifact Inventory in `results/phase1_baseline_v2/`

```
results/phase1_baseline_v2/
  dqn_seed_1.csv .. dqn_seed_5.csv      (101 lines each, per-episode DQN metrics)
  dqn_frozen_eval.csv                   (6 lines, deterministic evaluation across 5 seeds)
  fixedtime_seeds.csv                   (6 lines, Fixed-Time per-seed evaluation)
  maxpressure_seeds.csv                 (6 lines, corrected Max-Pressure per-seed evaluation)
  summary_baselines.csv                 (9 lines, baseline summary table)
  summary_all.csv                       (17 lines, full cross-controller comparison table)
  environment.txt                       (software and environment metadata)
  phase1_report.md                      (this report)
  checkpoints/
    seed_{1..5}/trained_model.pt        (saved PyTorch model weights)
  plots/
    dqn_learning_curves.png             (multi-seed DQN training progression)
    controller_comparison.png           (bar chart comparing all 4 controller configurations)
```

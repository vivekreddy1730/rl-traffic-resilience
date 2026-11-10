# Frozen DQN weights

`seed_N/trained_model.pt` holds the trained Q-network for training seed N (N = 1..5),
saved by `tests/run_phase1_dqn.py` after 100 training episodes (Phase 1).

These are byte-identical copies of `results/phase1_baseline_v2/checkpoints/seed_N/trained_model.pt`
from the original project (MD5 checked). All demos and verification scripts load them with
epsilon = 0 (no exploration, no further training).

The file is a full pickled PyTorch module (Vidali's `torch.save(model)` format), so it is loaded with
`torch.load(..., weights_only=False)` and needs the `tlcs` package on the Python path
(the scripts handle this automatically).

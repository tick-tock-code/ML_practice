# Agent Notes

Use `C:\ProgramData\anaconda3\envs\torch_env2\python.exe` for commands that need PyTorch.

Start with:

- `quant_rnn/cli.py` for available commands.
- `quant_rnn/data.py` for ingestion, features, splits, and scaling.
- `quant_rnn/sequences.py` for sample construction and leakage boundaries.
- `quant_rnn/models.py` for model registry.
- `quant_rnn/training.py` and `quant_rnn/evaluation.py` for stages 2 and 3.
- `quant_rnn/baselines.py` for cash, equal-weight, and momentum baselines.
- `quant_rnn/walk_forward.py` for expanding-window retraining.
- `quant_rnn/aggregation.py` for validation summaries and selection comparisons from stored grid checkpoints.
- `quant_rnn/seed_sweep.py` for repeated fixed-config training across random seeds.
- `scripts/run_weight_decay_grid.py` for the reusable LSTM `1e-5` learning-rate weight-decay grid runner.

Generated artifacts are ignored by git:

- `data/`
- `runs/`
- `*.pt`
- `*.parquet`

Run tests with:

```powershell
C:\ProgramData\anaconda3\envs\torch_env2\python.exe -m unittest discover -s tests
```

The current default universe uses current S&P 500 membership. Keep the survivorship-bias warning unless historical membership support is added.

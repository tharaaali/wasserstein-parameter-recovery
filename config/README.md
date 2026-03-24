# Configuration Files

This directory contains the YAML files used to train and evaluate CaloGAN runs.

- `train.yaml` is the main baseline configuration for the prepared-data WGAN workflow.
- `io.fresh_data_path` and `io.aged_data_path` should point to already-prepared inputs.
- `io.target_aging_path` is optional and is only needed when you want RMSE/MAE/R2 alignment metrics.
- Experiment subdirectories store parameter sweeps and generated config sets.
- `create_cfgs.py` scripts generate batches of related configs for sweep runs.

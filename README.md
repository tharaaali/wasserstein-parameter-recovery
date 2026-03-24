# Calorimeter Aging GAN

This repository contains a script-based CaloGAN WGAN workflow for learning aging factors
from prepared fresh/aged calorimeter datasets.

## Repository Layout

- `config/` holds baseline YAML files, generated sweeps, and config helpers.
- `scripts/` contains the training and evaluation entry points.
- `utils/` contains shared data-loading, model, checkpoint, and evaluation helpers.
- `results/` is where experiment outputs, checkpoints, figures, and logs are written.

## Training

Run the main training entry point with prepared fresh/aged inputs from `config/`:

```bash
python scripts/calogan_train.py --config config/train.yaml
```

Each run creates a timestamped directory inside `results/` and saves the learned
aging-factor tensor as `learned_aging_factors.pt` and `learned_aging_factors.npy`.

## Evaluation

If you provide `io.target_aging_path`, training will also write final alignment plots
and `training_metrics.csv` under the run directory in `results/`.

## Notes

- The training entrypoint no longer generates synthetic aging factors inside the repo.
- If you want alignment metrics, provide `io.target_aging_path` in the YAML config.

## License

This project is released under the LAMBDA LAB License.

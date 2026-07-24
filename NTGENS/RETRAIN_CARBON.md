# Retraining CSPDiffusion on carbon_24 (CNT-native checkpoint)

The CTGEN notebook (`../CTGEN_generation/05_ctgen_generation.ipynb`) generates with
the public **mp_20** diffusion checkpoint by default. For carbon-native generation,
train a fresh CSPDiffusion on the **carbon_24** dataset. Everything below is already
scaffolded in this repo; only the actual training run (GPU, hours–days) is left.

## Already scaffolded

- `data/carbon_24/{train,val,test}.csv` — carbon_24 dataset (CIF strings in the
  `cif` column, `energy_per_atom` property), copied from `../DiffCSP-main/data/carbon_24/`.
- `conf/data/carbon_24.yaml` — hydra data config (`prop: energy_per_atom`,
  `max_atoms: 24`, `train_max_epochs: 4000`), targets `scigen.pl_data.*`.
- `scigen -> ntgent` symlink — the package dir is named `ntgent/`, but all imports,
  hydra `_target_`s, and `setup.py` reference `scigen`. The symlink makes
  `import scigen` (and `python scigen/run.py`) work. If it is missing:
  `ln -s ntgent scigen` from the repo root.

## Prerequisites (GPU machine)

1. Python env with: `torch` (CUDA build), `torch_geometric`, `torch_scatter`,
   `pytorch_lightning`, `hydra-core`, `omegaconf`, `pymatgen`, `p_tqdm`, `wandb`
   (see `requirements.txt` / `env.yml` if present).
2. Environment variables (copy `.env.template` -> `.env` or export):
   - `PROJECT_ROOT` = absolute path of this `NTGEN-edit` directory
   - `HYDRA_JOBS`   = output directory for runs (e.g. `$PROJECT_ROOT/hydra`)
   - `WANDB_DIR`    = wandb scratch dir (e.g. `$PROJECT_ROOT/wandb`)

## Train

```bash
cd models/NTGEN-edit
python scigen/run.py data=carbon_24 model=diffusion_w_type expname=cnt_carbon24
```

Notes:
- `model=diffusion_w_type` selects `CSPDiffusion` with atom-type diffusion
  (`conf/model/diffusion_w_type.yaml`; costs: coord 1.0, lattice 1.0, type 20.0).
- First run preprocesses every CIF (CrystalNN graphs) and caches
  `data/carbon_24/{train,val,test}_ori.pt`; later runs reuse the cache.
- Checkpointing: best `val_loss` (top-1) saved by PyTorch Lightning.
- To resume, rerun the same command — the run dir's newest `epoch=*.ckpt` is picked
  up automatically.

## Outputs

`${HYDRA_JOBS}/singlerun/<yyyy-mm-dd>/cnt_carbon24/` will contain:

- `epoch=*-step=*.ckpt` — model weights
- `hparams.yaml` — resolved config (read back by `load_model_for_inference`)
- `lattice_scaler.pt`, `prop_scaler.pt` — scalers fitted on the training set

That directory is exactly the bundle the generation notebook expects: set
`MODEL_PATH` in `05_ctgen_generation.ipynb` to it and rerun.

## Sanity check without launching training

```bash
python scigen/run.py data=carbon_24 model=diffusion_w_type expname=smoke --cfg job
```

prints the fully-resolved config (dataset paths, model targets) and exits before
any GPU work if you Ctrl-C at the trainer start; verify `data.root_path` points at
`data/carbon_24` and `model._target_` is `scigen.pl_modules.diffusion_w_type.CSPDiffusion`.

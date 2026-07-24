# Configuration: `conf/`, `.env`, and user config scripts

> **Audience:** programmer. **Purpose:** every knob that parameterizes a run without editing model code.
> ⬅ Back to [docs hub](../README.md) · Related: [core-package-ntgent.md](core-package-ntgent.md), [setup.md](../usage/setup.md)

Three configuration mechanisms coexist:
1. **Hydra `conf/`** — drives **training** (`scigen/run.py`).
2. **`.env`** — supplies path env vars used throughout.
3. **Editable Python config scripts** (`config_scigen.py`, `gnn_eval/config_eval.py`) — drive **generation/screening**.

## Hydra `conf/`
```
conf/
├── default.yaml         top-level composition
├── data/  { carbon_24.yaml, mp_20.yaml }
├── logging/default.yaml
├── model/  { diffusion.yaml, diffusion_w_type.yaml, energy.yaml,
│            beta_scheduler/cosine.yaml, decoder/cspnet.yaml,
│            sigma_scheduler/wrapped.yaml }
├── optim/default.yaml
└── train/default.yaml
```

### `default.yaml` composition
```yaml
defaults:
  - data: default        # ⚠ overridden on the CLI — no data/default.yaml exists
  - logging: default
  - model: diffusion     # ⚠ the USED model is diffusion_w_type — override on CLI
  - optim: default
  - train: default
```
It also templates the Hydra output dirs from env vars:
```yaml
hydra.run.dir:  ${oc.env:HYDRA_JOBS}/singlerun/${now:%Y-%m-%d}/${expname}/
```

> **Two defaults you must override.** `conf/data/` contains only `carbon_24.yaml` and `mp_20.yaml` (no `default.yaml`), and the used model is `diffusion_w_type`, not the composed `diffusion`. So every real command passes both, e.g.:
> ```bash
> python scigen/run.py data=mp_20 model=diffusion_w_type expname=<name>
> ```
> See [known-discrepancies.md](../known-discrepancies.md).

### `model/diffusion_w_type.yaml` (the used model)
```yaml
_target_: scigen.pl_modules.diffusion_w_type.CSPDiffusion
time_dim: 256
cost_coord: 1.     cost_lattice: 1.     cost_type: 20.
max_neighbors: 20  radius: 7.           timesteps: 1000
defaults: { decoder: cspnet, beta_scheduler: cosine, sigma_scheduler: wrapped }
```
The `cost_*` weights combine the three training losses (coord / lattice / type). → [diffusion-model.md](diffusion-model.md).

## `.env` (copy from `.env.template`)
```bash
export PROJECT_ROOT=""   # repo/package root, used by common/utils.py
export HYDRA_JOBS=""     # where Hydra writes run/sweep outputs
export WANDB_DIR=""      # Weights & Biases log dir
```
Loaded via `python-dotenv` in `ntgent/common/utils.py`. → [setup.md](../usage/setup.md).

## User config scripts (gitignored)
Copied from templates and edited before running; **gitignored** so credentials/paths stay local.

- **`config_scigen.py`** (from `config_scigen_template.py`) — imported by `gen_mul.py`, `screen_mul.py`, and `script/*`. Fields: `home_dir`, `hydra_dir`, `job_dir`, `out_name`, `gnn_eval_path`, `stab_pred_name_A/B`, `mag_pred_name`, `seedn=42`.
- **`gnn_eval/config_eval.py`** (from `config_eval_template.py`) — `api_key`, `model_dir`/`data_dir`, `seedn`. Only needed to (re)train screening classifiers. → [gnn-screening.md](gnn-screening.md).

## What's gitignored (and why)
The `.gitignore` files exclude local/experimental artifacts: `__pycache__/`, `archive/`, `figures/`, `slurm/`, `*.sh`, `.env`, `config_scigen.py`, `screening/`, `models/`, cached preprocessed tensors (`data/*/*.pt`), `conf/data/alex_2d.yaml` (a hidden 2D-Alexandria config), and a long list of the original author's personal scratch scripts. This is why cloning the repo gives you templates, not filled-in configs — you must copy and fill them.

## Next

- Put it all together → [usage/setup.md](../usage/setup.md)
- Training-loop internals → [core-package-ntgent.md](core-package-ntgent.md)

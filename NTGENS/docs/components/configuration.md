# Configuration: `conf/`, `.env`, and user config scripts

> **Audience:** programmer. **Purpose:** every knob that parameterizes a run without editing model code.
> ⬅ Back to [docs hub](../README.md) · Related: [core-package-ntgent.md](core-package-ntgent.md), [setup.md](../usage/setup.md)

Four configuration mechanisms coexist:
1. **Hydra `conf/`** — drives **training** (`scigen/run.py`).
2. **`.env`** — supplies path env vars used throughout.
3. **Editable Python config scripts** (`config_scigen.py`, `gnn_eval/config_eval.py`) — drive **generation/screening**.
4. **Runtime model attributes** (`model.pin_cfg` / `cyl_cfg` / `dens_cfg` / `ang_cfg` / `geom_pin_cfg`) — set by `script/generation.py` from CLI flags (or by the notebook directly) and read by `sample_scigen`. A pretrained checkpoint's `self.hparams` are frozen, so guidance is configured here rather than in Hydra. Tabulated below.

## Generation-time guidance knobs

Every flag below is a `script/generation.py` CLI arg; the notebook `ntgen_generation.ipynb` exposes the same knobs as UPPER_CASE variables in its config cell. All apply to `sc='shl'` only — `van` and any graph without a real template (`is_alx=0`) are untouched. Mechanism: [technical-foundations.md](../technical-foundations.md) §5b.

**Skeleton/lattice pinning ramp `ψ(t)`** → `model.pin_cfg`

| Flag | Default | Meaning |
|---|---|---|
| `--pin_schedule` | `none` | `none` (binary) \| `linear` \| `sigmoid` |
| `--pin_alpha` | `10.0` | sigmoid steepness |
| `--pin_tmid` | `0.6` | sigmoid inflection, as a fraction of `T` |
| `--pin_psi_start` | `0.0` | ψ at `t=T`. **Keep at 0** — see [known-discrepancies.md](../known-discrepancies.md) §8 |
| `--pin_psi_end` | `1.0` | ψ at `t=0` |

**Geometric-force ramp** (decouples the force gain from the mask) → `model.geom_pin_cfg`

| Flag | Default | Meaning |
|---|---|---|
| `--geom_pin_schedule` | `same` | `same` reuses `--pin_schedule`; otherwise `none`/`linear`/`sigmoid` |
| `--geom_psi_start` / `--geom_psi_end` | `0.0` / `1.0` | force gain at `t=T` / `t=0` |

**Radial band + frame tracking** → `model.cyl_cfg`

| Flag | Default | Meaning |
|---|---|---|
| `--cyl_masking` | `False` | confine atoms to `[r_min, r_max]` |
| `--cyl_margin` | `0.1` | `r_hi = r_max · (1 + margin)` |
| `--cyl_r_lo_pct` | `5.0` | inner band-edge percentile |
| `--cyl_strength` | `1.0` | peak radial pull (× ψ) |
| `--cyl_track_frame` | `True` | rebuild the frame from the **current** lattice each step — the fix for azimuthal clustering |
| `--cyl_scale_lo` / `--cyl_scale_hi` | `0.25` / `4.0` | clamp `s_t` against the linear cell-scale ratio |
| `--cyl_ang_min` | `0.05` | min `sin(angle)` between transverse rows before a cell counts as degenerate |

**Radial density guidance** → `model.dens_cfg`

| Flag | Default | Meaning |
|---|---|---|
| `--density_guidance` | `False` | step `r` up `d/dr log ρ(r)` toward the wall |
| `--density_strength` | `0.05` | radial step scale (× ψ × `s_t`) |
| `--bandwidth_scale` | `1.0` | KDE bandwidth = wall thickness; `<1` sharpens |
| `--density_grid_size` | `96` | force-table resolution |

**Angular dispersion** → `model.ang_cfg`

| Flag | Default | Meaning |
|---|---|---|
| `--ang_spread` | `False` | reduce the circular resultant `R̄` (0 = spread, 1 = one arc) |
| `--ang_strength` | `0.05` | max radians rotated per application (× ψ) |
| `--ang_modes` | `1` | `1` = first moment; `2` also splits antipodal pairs |
| `--ang_mode2_weight` | `0.25` | mode-2 weight when `--ang_modes ≥ 2` |
| `--ang_mode2_floor` | `0.10` | mode-2 deadband — real tubes have genuine n-fold structure at `R̄₂ ≈ 0.10`, so do **not** drive it to 0 |
| `--ang_min_atoms` | `3` | graphs with fewer valid atoms are skipped |
| `--ang_max_dtheta` | `0.5` | hard cap on `|Δθ|` per application (rad) |

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

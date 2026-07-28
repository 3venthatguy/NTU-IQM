# Workflows

> **Audience:** programmer. **Purpose:** runnable, end-to-end commands for each task, with the artifact each produces.
> ⬅ Back to [docs hub](../README.md) · Prereq: [setup.md](setup.md)

Run from the `NTGENS/` root. Assumes the [symlink](../architecture/module-dependencies.md), `.env`, and `config_scigen.py` are in place ([setup.md](setup.md)).

## 0. Pick your route
Two ways to generate:
- **Notebook route** (friendliest) — `../comp_models/NTGEN_generation/*.ipynb`. Colab-oriented; handles checkpoint download. ⚠ these assume an older `NTGEN-edit` folder name — see [known-discrepancies.md](../known-discrepancies.md) §3.
- **CLI route** — edit `gen_mul.py` and run it, or call `script/generation.py` directly. Documented below.

## 1. Build the template cache (once, for `shl`)
```bash
python data/nano_1D/build_templates.py
```
→ produces `data/nano_1D/nanotube_templates.npz`. Skip if you're only doing `van`.

## 2. (Optional) Train a diffusion model
Only needed if you don't use the pretrained `models/mp_20/` checkpoint.
```bash
# general (multi-element) model:
python scigen/run.py data=mp_20 model=diffusion_w_type expname=<name>
# nanotube-native (for shl; fixes the mp_20-is-OOD generation problem) — see RETRAIN_ALX.md:
#   first: build train/val/test.csv from the sources (build_train_csv.py was removed — recreate; see RETRAIN_ALX.md)
python scigen/run.py data=alx_1d model=diffusion_w_type expname=alx_1d
# smoke test (resolve config, no training):
python scigen/run.py data=mp_20 model=diffusion_w_type expname=smoke --cfg job
```
> Always pass both `data=` and `model=diffusion_w_type` — the config defaults (`data: default`, `model: diffusion`) are not what you want. See [configuration.md](../components/configuration.md).

→ writes checkpoints + `lattice_scaler.pt` / `prop_scaler.pt` under `${HYDRA_JOBS}/singlerun/<date>/<expname>/`.

## 3. Generate structures

**Batch driver:** edit the params block at the top of [`gen_mul.py`](../../gen_mul.py), then:
```bash
python gen_mul.py
```
Key params: `dataset` (`'mp_20'` → general multi-element tubes; `'carbon_24'` → all-carbon tubes), `sc_list` (e.g. `['shl']`, `['van']`), `atom_list` (known species used for bond-length sampling, e.g. `['Mn','Fe']`), `batch_size`, `num_batches_to_samples`, `frac_z`, `sc_natm_range`. It loops `sc_list × num_run` and shells out to `generation.py`.

**Direct CLI** (equivalent single run):
```bash
python script/generation.py \
  --model_path models/mp_20 --dataset mp_20 \
  --sc shl --natm_range 24 64 \
  --frac_z 0.5 --label myrun [--save_traj True]
```
→ writes `models/mp_20/eval_gen_myrun.pt` (the [output contract](../components/generation-scripts.md)).

> **Constraint ↔ dataset pairing:** `shl` generates every atom's species, so pair it with a **general** dataset (`mp_20`/`uniform`); use `carbon_24` only when you specifically want all-carbon tubes (it flattens species to carbon). See [extending.md](extending.md) and [known-discrepancies.md](../known-discrepancies.md).

## 4. Export CIF files
```bash
python script/save_cif.py --label myrun
```
→ `.cif` files (+ a zip) and `eval_setting.json` under the job dir. (Or set `save_cif=True` in `gen_mul.py` to do it automatically.)

## 5. Screen the outputs
```bash
python script/eval_screen.py --label myrun [--screen_mag True]
```
→ surviving structures as CIFs after SMACT validity → occupancy ratio → GNN classifier cascade. Requires trained classifier weights referenced in `config_scigen.py`. → [gnn-screening.md](../components/gnn-screening.md).

(Batch equivalent: `python screen_mul.py` — but note its `sc_list` is stale; see [known-discrepancies.md](../known-discrepancies.md) §2.)

## 6. Render a trajectory movie
```bash
python script/traj_movie.py --label myrun --idx_list 0 1 2 --supercell 1 1 3
```
→ GIF(s) under `figures/<job_dir>/…`. Requires `--save_traj True` at generation time (step 3).

## 7. Compute benchmark metrics
```bash
python script/compute_metrics.py --label myrun
```
→ validity / novelty / uniqueness / coverage metrics (CDVAE/DiffCSP-style).

## 8. Inspect outputs (notebook)
For a quick, in-memory sanity check before exporting anything, use the generation notebook directly — see **Section 7 ("Analysis")** of `ntgen_generation.ipynb`: lattice parameter distributions, space-group analysis, and simulated XRD, run straight on the freshly generated `structures` list. Full pattern + nanotube-specific caveats: [inspecting-outputs.md](inspecting-outputs.md).

## Quick reference

| Goal | Command |
|---|---|
| Build `shl` templates | `python data/nano_1D/build_templates.py` |
| Train | `python scigen/run.py data=mp_20 model=diffusion_w_type expname=X` |
| Generate (batch) | edit + `python gen_mul.py` |
| Generate (one) | `python script/generation.py --sc … --label X …` |
| CIFs | `python script/save_cif.py --label X` |
| Screen | `python script/eval_screen.py --label X` |
| Movie | `python script/traj_movie.py --label X --idx_list …` |
| Metrics | `python script/compute_metrics.py --label X` |
| Inspect (notebook) | `ntgen_generation.ipynb` Section 7 — see [inspecting-outputs.md](inspecting-outputs.md) |

## Next

- Add a new constraint → [extending.md](extending.md)
- The pipeline in detail → [../architecture/data-flow.md](../architecture/data-flow.md)

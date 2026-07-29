# Generation & Evaluation Scripts: `script/`

> **Audience:** programmer. **Purpose:** the CLI layer that drives sampling and post-processing.
> ⬅ Back to [docs hub](../README.md) · Related: [data-flow.md](../architecture/data-flow.md), [workflows.md](../usage/workflows.md)

```
script/
├── generation.py       run constrained sampling → eval_gen_<label>.pt
├── gen_utils.py        SampleDataset + nanotube-template DB glue
├── sc_utils.py         SC_* constraint classes  → see structural-constraints.md
├── sc_natm.py          empirical atom-count distributions
├── eval_utils.py       load_model, recommand_step_lr, CompScaler, validity
├── eval_funcs.py       GNN-screening helpers (used by eval_screen.py)
├── eval_screen.py      filter generated structures via GNN classifiers → CIFs
├── save_cif.py         convert .pt output → .cif files (+ zip)
├── compute_metrics.py  validity / novelty / uniqueness / coverage metrics
├── mat_utils.py        structure conversion + movie rendering
└── traj_movie.py       render a GIF of the diffusion trajectory
```

## `generation.py` — the sampling driver
Invoked by `gen_mul.py` or directly. Flow:
1. `eval_utils.load_model(model_path, load_data=False)` loads the trained checkpoint.
2. Monkey-patch `model.sample_scigen = sample_scigen.__get__(model)`.
3. Build a `gen_utils.SampleDataset(...)` (the pinned conditioning batch) → PyG `DataLoader`.
4. Local `diffusion(loader, model, step_lr, save_traj)` iterates batches calling `model.sample_scigen(batch, step_lr=step_lr)`.
5. Concatenate outputs, convert lattice → `(lengths, angles)` via `lattices_to_params_shape`.
6. `torch.save` a dict to `<model_path>/eval_gen_<label>.pt`.

**Key CLI args:** `--model_path`, `--dataset`, `--sc` (nargs, default `['shl']`), `--natm_range`, `--known_species`, `--c_scale`, `--c_vert`, `--frac_z`, `--reduced_mask`, `--save_traj`, `--batch_size`, `--num_batches_to_samples`, `--step_lr`, `--label`, `--max_decorators`, `--bond_sigma_per_mu`, `--use_min_bond_len`. The full set of `shl` guidance knobs is tabulated in [configuration.md](configuration.md#generation-time-guidance-knobs) — `--template_source`, the `--pin_*` / `--geom_*` ramps, `--cyl_*` band + frame tracking, `--density_*`, and `--ang_*`.

**Output contract (`eval_gen_<label>.pt`):** a dict with `frac_coords`, `atom_types`, `lengths`, `angles`, `num_atoms`, `num_known`, plus (if `--save_traj`) `all_frac_coords` / `all_atom_types` / `all_lattices` / `all_lengths` / `all_angles`, and metadata (`c_vec_cons`, `seed`, `time`, and the guidance configs `pin_cfg` / `cyl_cfg` / `dens_cfg` / `ang_cfg` / `geom_pin_cfg`). Every downstream script reads this file.

## `gen_utils.py` — building the conditioning batch
Home of **`SampleDataset(Dataset)`**, the PyG dataset that turns constraint choices into pinned `Data` objects. Pipeline (per [data-flow.md](../architecture/data-flow.md) stages 2–3):
- `num_atom_distribution()` — loads empirical atom-count distributions (`sc_natm.natm_dist`), keyed by dataset or constraint.
- `process()` — for each sample: pick a constraint from `sc_list`; sample a bond length (KDE from `kde_bond.pkl`, or Gaussian from `metallic_radius` when `bond_sigma_per_mu` is set, or `fallback_bond_len` for `C`); instantiate the `SC_*` class; for `shl` fetch a DB template (`nanotube_template_from_db`, falling back to the private `_SC_NanotubeFallback` if none fits) and measure the radial band; set `num_atom`; call `frac_coords_all()` / `atm_types_all()`.
- `generate_dataset()` — wrap into PyG `Data` with `num_atoms`, `num_known`, `frac_coords_known`, `lattice_known`, `atom_types_known`, `mask_x/mask_l/mask_t`, plus the `shl` shell metadata: `is_alx`, `r_min`, `r_max`, `tube_axis`, `tube_centroid`, **`tube_centroid_frac`**, `tube_a_hat`, `tube_e1`, `tube_e2`, `dens_force`, `dens_grid_lo`, `dens_grid_dr`, `template_index`, `template_source`.
  > `tube_centroid` is Cartesian in the *template's* cell, so it is only valid at `t=0`; `tube_centroid_frac` is the same point in fractional coords, which is what lets the sampler rebuild the frame from the current (still-noisy) lattice each step. Graphs with `is_alx=0` carry a **zero** frame — the sampler must skip them, or `apply_radial_band` maps all their atoms to the origin. **Sets `atom_types = [6]*N` when `dataset == 'carbon_24'`** (the carbon-flattening special case — see [known-discrepancies.md](../known-discrepancies.md)).

Also here: the nanotube template DB glue — `class _NanotubeTemplateDB`, `NANOTUBE_DB` singleton, `nanotube_template_from_db`, plus `metallic_radius`, `fallback_bond_len = {'C': 1.42}`, and `set_seeds`. Detail in [nanotube-template-db.md](nanotube-template-db.md).

> **Heads-up:** [`gen_utils.py`](../../script/gen_utils.py) contains an accidentally-pasted LLM-prompt comment just above `parse_none_or_value` (~line 105). It is inert noise — see [known-discrepancies.md](../known-discrepancies.md) §6. The `parse_none_or_value` function itself is used to pass `None` through CLI args.

## `sc_natm.py` — atom-count distributions
`natm_dist_0` — empirical atom-count-per-cell probabilities for `perov_5`, `carbon_24`, `mp_20`, `uniform`. `natm_dist_sc` — same, keyed per structural constraint (still using the **legacy 2D names** `tri`/`hon`/…). `natm_dist` (merged, imported by `gen_utils`). When a constraint key isn't present, `SampleDataset` falls back to the dataset-level distribution.

## `eval_utils.py`
- `load_model(...)` — instantiate the model + load a checkpoint.
- `lattices_to_params_shape(...)` — lattice matrix → `(lengths, angles)`.
- `recommand_step_lr` — nested dict of recommended Langevin step sizes per task (`csp`/`csp_multi`/`gen`) × dataset.
- `CompScaler` — composition scaler built from `constants.CompScalerMeans/Stds`; SMACT validity helpers (`smact.screening.pauling_test`).

## Post-processing scripts
| Script | Input → Output | Notes |
|---|---|---|
| `save_cif.py` | `eval_gen_<label>.pt` → `.cif` files + zip | via `mat_utils.get_pstruct_list`; also dumps `eval_setting.json`. |
| `eval_screen.py` | `.pt` → screened `.cif` | SMACT validity → occupancy ratio < 1.7 → GNN classifier cascade. → [gnn-screening.md](gnn-screening.md) |
| `traj_movie.py` | `.pt` (with trajectory) → `.gif` | needs `--save_traj` at generation time; uses `mat_utils.movie_structs`. |
| `compute_metrics.py` | `.pt` → metrics | CDVAE/DiffCSP-style validity/novelty/uniqueness/coverage; `matminer` fingerprints, Wasserstein distances. |
| `mat_utils.py` | (library) | `get_pstruct_list`, `output_gen`, `movie_structs`, `ase2pmg`, `save_combined_cif`, etc. |

**Complementary notebook path:** before running any of the above, `ntgen_generation.ipynb` converts the raw tensors straight to `pymatgen.Structure` in memory (Section 6) and exports CIFs (Section 7) — a faster, visual first pass. The sibling `ntgen_validation.ipynb` then runs the actual post-generation analysis (geometry/chemistry/CHGNet-energy/relaxation tiers + composite ranking) against those CIFs, ahead of `compute_metrics.py`. See [usage/inspecting-outputs.md](../usage/inspecting-outputs.md).

## Batch drivers (top level)
- **`gen_mul.py`** — edit the params block (`dataset`, `batch_size`, `num_batches_to_samples`, `sc_list`, `atom_list`, `frac_z`, `sc_natm_range`, …) then `python gen_mul.py`; it loops `sc_list × num_run` and shells out to `script/generation.py`, optionally calling `save_cif.py`. Current defaults target the geometric shell: `dataset='mp_20'`, `sc_list=['shl']`, `sc_natm_range={'shl': [24, 64], 'van': [1, 20]}`.
- **`screen_mul.py`** — batch screening driver. ⚠ **Stale:** its `sc_list` still lists the old 2D lattice names (`tri`, `hon`, …), not the nanotube constraints. See [known-discrepancies.md](../known-discrepancies.md) §2.

## Next

- The template DB → [nanotube-template-db.md](nanotube-template-db.md)
- Run these end-to-end → [usage/workflows.md](../usage/workflows.md)

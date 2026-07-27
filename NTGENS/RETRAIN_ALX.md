# Retraining CSPDiffusion on Alexandria 1D (nanotube-native checkpoint)

The `ntgen_generation.ipynb` notebook generates with the public **mp_20** diffusion
checkpoint. That checkpoint was trained on **≤20-atom bulk crystals with no vacuum
and no 1D periodicity**, so it is out-of-distribution for nanotubes: even with the
`shl` shell fixing a real tube's geometry (`SC_DBShell`), the model cannot place the
generated atoms sensibly inside the wall, and (before the fix) freely inflated the
tube's c-axis. This retrain on the Alexandria 1D nanotube data is the root fix for
**generation quality** — the mp_20-is-OOD fault.

Everything below is scaffolded in this repo; only the data-prep and the training run
(GPU, hours–days) are left, and both must run on a machine with `pymatgen` +
`torch_geometric` (neither is available in the local analysis env).

## Already scaffolded

- `data/alx_1D/build_train_csv.py` — converts `alexandria_1d_nanotubes.pkl` →
  `data/alx_1D/{train,val,test}.csv` (columns `material_id, pretty_formula,
  energy_per_atom, cif`), reusing the ase-optional loader from `build_templates.py`.
- `conf/data/alx_1d.yaml` — hydra data config (`prop: energy_per_atom`,
  `max_atoms: 64`, targets `scigen.pl_data.*`), mirroring `conf/data/carbon_24.yaml`.
- `scigen -> ntgent` symlink — required for `import scigen` / `python scigen/run.py`.
  If missing: `ln -s ntgent scigen` from the repo root.

## Step 1 — build the training CSVs (needs pymatgen)

```bash
cd NTGENS
python data/alx_1D/build_train_csv.py
```

Produces `data/alx_1D/{train,val,test}.csv` (80/10/10 split of structures with
1–64 atoms). Notes:
- The generative CSPDiffusion trains on **geometry**, not the property; the
  `energy_per_atom` column is a best-effort/placeholder value (the Alexandria pickle
  usually carries no attached calculator once unpickled), which is fine — the
  datamodule only fits a scaler over it.
- `max_atoms: 64` in `conf/data/alx_1d.yaml` must match `MAX_NATM` in the prep script.

## Step 2 — prerequisites (GPU machine)

1. Python env with: `torch` (CUDA), `torch_geometric`, `torch_scatter`,
   `pytorch_lightning`, `hydra-core`, `omegaconf`, `pymatgen`, `p_tqdm`, `wandb`.
2. Environment variables (copy `.env.template` → `.env` or export):
   - `PROJECT_ROOT` = absolute path of this `NTGENS` directory
   - `HYDRA_JOBS`   = run output dir (e.g. `$PROJECT_ROOT/hydra`)
   - `WANDB_DIR`    = wandb scratch dir (e.g. `$PROJECT_ROOT/wandb`)

## Step 3 — train

```bash
cd NTGENS
python scigen/run.py data=alx_1d model=diffusion_w_type expname=alx_1d
```

- `model=diffusion_w_type` selects `CSPDiffusion` with atom-type diffusion
  (`conf/model/diffusion_w_type.yaml`; costs: coord 1.0, lattice 1.0, type 20.0).
- First run preprocesses every CIF (CrystalNN graphs) and caches
  `data/alx_1D/{train,val,test}_ori.pt`; later runs reuse the cache.
- Resume by rerunning the same command — the newest `epoch=*.ckpt` is auto-picked.

## Step 4 — point the notebook at the new checkpoint

`${HYDRA_JOBS}/singlerun/<yyyy-mm-dd>/alx_1d/` will contain `epoch=*.ckpt`,
`hparams.yaml`, `lattice_scaler.pt`, `prop_scaler.pt` — exactly the bundle
`load_model_for_inference` expects. In `ntgen_generation.ipynb` set `MODEL_PATH`
to that directory and rerun. Because the model now knows tube geometry + chemistry,
you can relax the aggressive template-domination (raise `MAX_DECORATORS`) and expect
sensible decoration rather than a mp_20 bulk-crystal blob.

## Sanity check without launching training

```bash
python scigen/run.py data=alx_1d model=diffusion_w_type expname=smoke --cfg job
```

prints the fully-resolved config; verify `data.root_path` ends in `data/alx_1D` and
`model._target_` is `scigen.pl_modules.diffusion_w_type.CSPDiffusion`.

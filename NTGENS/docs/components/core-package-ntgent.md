# Core Package: `ntgent/` (imported as `scigen`)

> **Audience:** programmer. **Purpose:** the DiffCSP-derived ML package — data plumbing, geometry, and the training entry point.
> ⬅ Back to [docs hub](../README.md) · Related: [diffusion-model.md](diffusion-model.md), [module-dependencies.md](../architecture/module-dependencies.md)

The directory is `ntgent/` but it is imported everywhere as `scigen` via the [symlink](../architecture/module-dependencies.md). It contains three sub-packages plus the training entry point.

```
ntgent/
├── run.py            training entry point (@hydra.main)
├── common/           utilities: geometry, env, constants
│   ├── data_utils.py    CIF↔graph, PBC geometry, scalers  (the big one, ~1344 lines)
│   ├── utils.py         PROJECT_ROOT, get_env, load_envs, log_hyperparameters
│   └── constants.py     CompScalerMeans / CompScalerStds (composition featurizer)
├── pl_data/          PyTorch Lightning data layer
│   ├── dataset.py       CrystDataset, TensorCrystDataset
│   └── datamodule.py    CrystDataModule
└── pl_modules/       the models — see components/diffusion-model.md
```

## `common/data_utils.py` — the geometry & preprocessing core

The largest and most-reused module. It converts CIF strings into the graph tensors the model consumes and provides periodic-boundary-condition (PBC) geometry primitives. Notable contents:

- **CIF → graph:** `preprocess(...)`, `preprocess_tensors(...)` — parse CIF via pymatgen, build crystal graphs (uses `pymatgen.analysis.local_env` / `StructureGraph`), parallelized with `p_tqdm.p_umap` / `pathos`.
- **Coordinate transforms:** `cart_to_frac_coords`, `frac_to_cart_coords`, `lattice_params_to_matrix_torch`, `lengths_angles_to_volume`.
- **PBC neighbor search:** `get_pbc_distances`, `radius_graph_pbc` / `radius_graph_pbc_wrapper`, `min_distance_sqr_pbc`; `OFFSET_LIST` (the 27 3D image offsets).
- **Scaling / metrics:** `StandardScaler`, `get_scaler_from_data_list`, `add_scaled_lattice_prop`, `mard`.
- **Constants:** `chemical_symbols`, `EPSILON`.

If you need to convert between representations or compute distances under PBC, the function almost certainly already exists here — check before writing new geometry code.

## `common/utils.py`

Environment & logging helpers:
- `PROJECT_ROOT` — derived from the `PROJECT_ROOT` env var (set in `.env`).
- `get_env(name, default=None)`, `load_envs(env_file=None)` — thin wrappers over `python-dotenv`.
- `log_hyperparameters(trainer, model, cfg)` — pushes the resolved config to the logger.
- `STATS_KEY = "stats"`.

## `common/constants.py`

Pure numeric data: `CompScalerMeans` / `CompScalerStds` — normalization constants for a composition-based featurizer. `script/eval_utils.py` builds `CompScaler = StandardScaler(means=CompScalerMeans, stds=CompScalerStds, …)` from these.

## `pl_data/dataset.py` — PyG datasets

- **`CrystDataset(Dataset)`** — reads a CSV (the `cif` column holds a full CIF string per row) into `self.df`, then `preprocess()` either loads a cached `.pt` or builds one via `data_utils.preprocess`. `__getitem__` returns a `torch_geometric.data.Data` with `frac_coords`, `atom_types`, `lengths`, `angles`, `edge_index`, `to_jimages`, `num_atoms`, `y` (scaled property), and optionally space-group / position-index fields.
- **`TensorCrystDataset(Dataset)`** — same shape of output but built from an in-memory `crystal_array_list` rather than a CSV (used for reconstruction/eval workflows).

> **The `.pt` cache:** the first time a CSV is used, preprocessing is expensive; the result is saved next to the CSV (`train_ori.pt`, etc.) and reused. These caches are gitignored (`data/*/*.pt`). Delete them to force a rebuild.

## `pl_data/datamodule.py` — `CrystDataModule`

A `pl.LightningDataModule` built from Hydra configs (`datasets`, `num_workers`, `batch_size`):
- `get_scaler(scaler_path)` computes or loads the `lattice_scaler` and property `scaler` (`StandardScaler`s).
- `setup(stage)` instantiates train/val/test `CrystDataset`s and attaches the scalers.
- `train_dataloader` / `val_dataloader` / `test_dataloader` return PyG `DataLoader`s with per-worker RNG seeding (`worker_init_fn`).

The fitted scalers are persisted at train time (`lattice_scaler.pt`, `prop_scaler.pt`) and ship with the pretrained checkpoint bundle in `models/mp_20/`.

## `run.py` — training entry point

`@hydra.main(config_path=PROJECT_ROOT/"conf", config_name="default")`, `main(cfg)` → `run(cfg)`:
1. Seed RNG if `cfg.train.deterministic`.
2. Instantiate `datamodule` and `model` via `hydra.utils.instantiate`.
3. Copy `datamodule.lattice_scaler` / `scaler` onto the model and persist them.
4. Build callbacks (`LearningRateMonitor`, `EarlyStopping`, `ModelCheckpoint`) conditionally from config.
5. Optionally build a `WandbLogger`.
6. Auto-resume from the newest `*.ckpt` in the Hydra run dir if present.
7. `pl.Trainer(...)` → `trainer.fit()` then `trainer.test()`.

Invoked as (note the model override — the config default is `diffusion`, see [configuration.md](configuration.md)):
```bash
python scigen/run.py data=mp_20 model=diffusion_w_type expname=<name>
```

## Next

- The models themselves → [diffusion-model.md](diffusion-model.md)
- How configs compose → [configuration.md](configuration.md)

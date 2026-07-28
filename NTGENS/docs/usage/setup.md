# Setup

> **Audience:** programmer. **Purpose:** get a working environment, in order.
> ⬅ Back to [docs hub](../README.md) · Related: [configuration.md](../components/configuration.md), [workflows.md](workflows.md)

Run everything from the `NTGENS/` root unless noted — several scripts use `cwd`-relative paths (e.g. `./data/kde_bond.pkl`).

## 1. The `scigen → ntgent` symlink (do this first)
The package dir is `ntgent/` but all imports say `scigen`. Verify / create the symlink:
```bash
ls -l scigen              # expect: scigen -> ntgent
ln -s ntgent scigen       # if missing
```
Without it, `import scigen` fails and nothing runs. Full rationale: [module-dependencies.md](../architecture/module-dependencies.md).

## 2. Environment file
```bash
cp .env.template .env
# then fill in:
#   PROJECT_ROOT=/absolute/path/to/NTGENS
#   HYDRA_JOBS=/absolute/path/for/hydra/outputs
#   WANDB_DIR=/absolute/path/for/wandb/logs
```

## 3. User config scripts
```bash
cp config_scigen_template.py config_scigen.py     # edit home_dir, hydra_dir, job_dir, out_name, seedn, ...
# only if you will retrain screening classifiers:
cp gnn_eval/config_eval_template.py gnn_eval/config_eval.py
```
Both are gitignored. See [configuration.md](../components/configuration.md).

## 4. Python dependencies
There is **no `requirements.txt`** — versions are documented only as prose in the two READMEs. Consolidated pinned list (from top-level [`README.md`](../../README.md)):

```
python==3.9.20         torch==2.0.1+cu118      torch-geometric==2.3.0
pytorch_lightning==1.3.8   pymatgen==2023.9.25   hydra-core==1.1.0
hydra-joblib-launcher==1.1.5   matminer==0.7.3   torchmetrics==0.7.3
pandas  smact  wandb  imageio  python-dotenv  p-tqdm  pytest  einops  pyxtal
```
The `gnn_eval/` sub-project adds (see [`gnn_eval/README.md`](../../gnn_eval/README.md)): `e3nn==0.5.1`, `plotly`, `mp_api`, `mendeleev`, `seaborn`.

> `pytest` is listed but there is **no test suite** — see [known-discrepancies.md](../known-discrepancies.md) §5.

## 5. Pretrained checkpoint (for generation without training)
The generation notebooks auto-download the mp_20 diffusion checkpoint (Figshare article 27778134) into `models/mp_20/`. The bundle is:
```
models/mp_20/
├── epoch=819-step=86919.ckpt   PyTorch Lightning weights (~148 MB)
├── hparams.yaml                resolved train-time config
├── lattice_scaler.pt           fitted StandardScaler (lattice)
└── prop_scaler.pt              fitted StandardScaler (property)
```
For **nanotube-native** generation (the recommended checkpoint for `shl`) you need a model trained on the Alexandria 1D data — none ships locally; see [`RETRAIN_ALX.md`](../../RETRAIN_ALX.md) and [workflows.md](workflows.md).

## 6. Build the nanotube template cache (only for `sc='shl'`)
```bash
python data/nano_1D/build_templates.py     # pkl → nanotube_templates.npz (run once)
```
If `nanotube_templates.npz` is absent at runtime, `sc='shl'` prints a warning and falls back to the private synthetic ring geometry. Detail: [nanotube-template-db.md](../components/nanotube-template-db.md).

## Environment caveats (important)
- **No GPU / no `torch_geometric` / `torch_scatter`** in the local envs on the original dev Mac — **full generation and training must run elsewhere** (a GPU box / Colab). The notebooks in `../comp_models/NTGEN_generation/` are Colab-oriented for exactly this reason.
- For lightweight, stub-based unit testing of individual functions without the full stack, see the recipe in [extending.md](extending.md).

## Verify the setup
```bash
ls -l scigen                                          # symlink present?
python scigen/run.py data=mp_20 model=diffusion_w_type expname=smoke --cfg job
```
The second command resolves and prints the composed Hydra config **without training** — a fast smoke test that the symlink, env, and configs all load. (`--cfg job` is documented in `RETRAIN_ALX.md`.)

## Next

- Run an end-to-end task → [workflows.md](workflows.md)
- Understand the configs you just copied → [configuration.md](../components/configuration.md)

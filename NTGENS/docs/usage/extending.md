# Extending & Gotchas

> **Audience:** programmer. **Purpose:** how to add a constraint, plus the traps that will bite you.
> ⬅ Back to [docs hub](../README.md) · Related: [structural-constraints.md](../components/structural-constraints.md)

## Add a new structural constraint

The `SC_*` classes in [`script/sc_utils.py`](../../script/sc_utils.py) follow a fixed contract; copy the scaffold and register it.

1. **Copy `SC_Template`** in `sc_utils.py` to `SC_MyTube(SC_Base)`. Fill in:
   - `a_scale`, `b_scale` — lattice scaling relative to `bond_len`.
   - `self.cell = self.get_cell(gamma=…)` — the lattice matrix.
   - `self.frac_known` — the fixed skeleton fractional coords `(num_known, 3)`.
   - `self.num_known = self.frac_known.shape[0]`.
   - Override `get_mask_l()` if you need a specific lattice mask (tube constraints return `mask_l_cvert`).
   - Override `atm_types_all()` only if you pin **per-atom** species (like `SC_DBTemplate`); otherwise the base method pins a single `type_known` and randomizes decorators.
   - Set `reduced_mask=False` behavior if your `mask_x` must stay `(N,3)`.
2. **Register it** in `sc_dict` at the bottom of `sc_utils.py`:
   ```python
   sc_dict = {..., 'myt': SC_MyTube}
   ```
3. **Add an atom-count distribution** in [`script/sc_natm.py`](../../script/sc_natm.py) — add a `'myt'` key to `natm_dist_sc` (or it falls back to the dataset-level distribution in `SampleDataset`).
4. **If it needs DB/parameter overrides**, add a branch in `SampleDataset.process()` ([gen_utils.py](../../script/gen_utils.py)) mirroring the `alx`/`ntb`/`cnt` cases that build `extra_kwargs`.
5. **Test it:** set `sc_list=['myt']` in `gen_mul.py` and run `python gen_mul.py` (this is the documented manual-test workflow — there is no test suite).

## Import & path conventions
- **Package imports use `scigen.*`** (the [symlink](../architecture/module-dependencies.md)), even though the dir is `ntgent/`.
- **`script/` files import each other flat** (`from sc_utils import *`), relying on being run with `cwd = NTGENS/`.
- **Run from `NTGENS/` root** — `gen_utils.py` opens `./data/kde_bond.pkl` by relative path at import; a wrong cwd throws `FileNotFoundError`.

## Gotchas (each has bitten someone)

### 1. Missing symlink → `ModuleNotFoundError: scigen`
Recreate: `ln -s ntgent scigen`. Check first when *anything* fails to import. → [module-dependencies.md](../architecture/module-dependencies.md).

### 2. carbon_24 flattens all species to carbon
`SampleDataset.generate_dataset()` sets `data.atom_types = [6]*num_atom` when `dataset == 'carbon_24'` ([gen_utils.py](../../script/gen_utils.py), the `is_carbon` branch). This is correct for CNTs but **destroys multi-element `alx` templates**. **Rule:** run `alx` with a general dataset (`mp_20`/`uniform`), run `cnt` with `carbon_24`. → [known-discrepancies.md](../known-discrepancies.md).

### 3. `MAX_NATM` (build) vs `natm_range` (runtime) mismatch
`build_templates.py` caches structures up to `MAX_NATM = 64`, but generation filters tighter by `natm_range` (e.g. 24 for carbon_24). `nanotube_template_from_db` further caps `N ≤ natm_max - 1` to leave room for ≥1 decorator. If no template fits, `alx` silently falls back to parametric `ntb`. So a too-small `natm_range` can mean you never actually use real templates. → [nanotube-template-db.md](../components/nanotube-template-db.md).

### 4. `sample_scigen` is monkey-patched, not a method
It's a module-level function attached at runtime: `model.sample_scigen = sample_scigen.__get__(model)` (in `generation.py`). If you call generation outside that script, you must patch it yourself. → [diffusion-model.md](../components/diffusion-model.md).

### 5. Config defaults aren't the used values
`conf/default.yaml` composes `model: diffusion` and `data: default`, but the used model is `diffusion_w_type` and there is no `data/default.yaml`. Always pass `data=… model=diffusion_w_type` on the CLI. → [configuration.md](../components/configuration.md).

## Stub-based manual testing (no full stack)
The original dev environment lacks `torch_geometric`/GPU, so individual functions were tested with stubs. Pattern:
- Stub `scigen.pl_modules.diffusion_w_type` (only `MAX_ATOMIC_NUM = 100` is needed by `sc_utils.py`) and `torch_geometric.data.Data`.
- Run from the `NTGENS/` cwd (so `gen_utils` finds `./data/kde_bond.pkl`).
- `build_templates.py` already self-stubs `ase`, so it can be tested without ASE installed.

This lets you exercise `SC_*` classes and `SampleDataset` logic in a plain `torch`-only interpreter. Full generation/training still requires a GPU box or Colab ([setup.md](setup.md)).

## Next

- The constraint contract in detail → [../components/structural-constraints.md](../components/structural-constraints.md)
- All known drift → [../known-discrepancies.md](../known-discrepancies.md)

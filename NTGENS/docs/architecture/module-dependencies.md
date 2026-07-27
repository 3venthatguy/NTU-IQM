# Module Dependencies & the `scigen` Symlink

> **Audience:** programmer. **Purpose:** the import graph, the two import styles, and the one symlink everything depends on.
> ⬅ Back to [docs hub](../README.md)

## ⚠ The `scigen → ntgent` symlink (read this first)

The Python package directory on disk is **`ntgent/`**, but **every import, every Hydra `_target_`, and `setup.py` all say `scigen`**:

```python
# script/sc_utils.py
from scigen.pl_modules.diffusion_w_type import MAX_ATOMIC_NUM
# conf/model/diffusion_w_type.yaml
_target_: scigen.pl_modules.diffusion_w_type.CSPDiffusion
```

A symlink bridges the name mismatch:

```
scigen  ->  ntgent
```

**This is a hard runtime dependency, not a convenience.** Without it, `import scigen` fails and *nothing* runs — not training, not generation, not the notebooks.

To (re)create it, from the `NTGENS/` root:
```bash
ln -s ntgent scigen
```

**Symptom when missing:** `ModuleNotFoundError: No module named 'scigen'`. Check with `ls -l scigen` (should show `scigen -> ntgent`). Verified at doc-write time that the symlink can go missing after directory reorganizations, so check it first when imports fail. Rationale is also documented in [`RETRAIN_ALX.md`](../../RETRAIN_ALX.md).

## Two import styles coexist

**1. Package imports (`scigen.*`)** — used by everything inside `ntgent/` and by `script/` files that reach into the model:
```python
from scigen.pl_modules.diffusion_w_type import sample_scigen, MAX_ATOMIC_NUM
from scigen.common.data_utils import lattice_params_to_matrix_torch
```
These require the symlink and that `NTGENS/` (or an installed `scigen`) is importable.

**2. Flat same-directory imports** — used *within* `script/`:
```python
# script/gen_utils.py
from sc_utils import *            # not scigen.sc_utils — a sibling file
from sc_natm import natm_dist
# script/generation.py
from eval_utils import load_model
from gen_utils import SampleDataset
```
These rely on `script/` being on `sys.path` — which happens because the scripts are **run with `cwd = NTGENS/`** and use `sys.path` manipulation. This is also why `gen_utils.py` opens `./data/kde_bond.pkl` with a **relative path** — run it from the wrong directory and it fails. See [usage/extending.md](../usage/extending.md) for this gotcha.

## Dependency direction (who imports whom)

```
data/alx_1D/build_templates.py        (standalone: numpy, optional ase)
        │  produces nanotube_templates.npz
        ▼
script/gen_utils.py  ── _NanotubeTemplateDB, SampleDataset
        │  from sc_utils import *
        ▼
script/sc_utils.py   ── SC_* classes, sc_dict
        │  from scigen.pl_modules.diffusion_w_type import MAX_ATOMIC_NUM
        ▼
script/generation.py ── driver
        │  from scigen.pl_modules.diffusion_w_type import sample_scigen
        │  from eval_utils import load_model
        ▼
ntgent/pl_modules/diffusion_w_type.py  (CSPDiffusion, sample_scigen)
        │  uses cspnet.py, diff_utils.py
        ▼
ntgent/common/data_utils.py            (geometry / graph primitives)

  downstream (consume eval_gen_<label>.pt):
    script/save_cif.py ─ mat_utils.py
    script/eval_screen.py ─ eval_funcs.py ─▶ gnn_eval/utils/*
    script/traj_movie.py ─ mat_utils.py
    script/compute_metrics.py
```

### Key cross-layer edges
- `script/sc_utils.py` → `scigen.pl_modules.diffusion_w_type` (only for the `MAX_ATOMIC_NUM=100` constant).
- `script/generation.py` → `scigen.pl_modules.diffusion_w_type.sample_scigen` + local `eval_utils`, `gen_utils`.
- `script/eval_screen.py` → `config_scigen` (user file) + `gnn_eval.utils.data` / `gnn_eval.utils.record` + local `eval_funcs`.
- `gnn_eval/train_*.py` → `gnn_eval/utils/*` (relative `from utils.data import …`) + `config_eval` (user file).
- `data/alx_1D/build_templates.py` is **standalone** — only `numpy` and optional `ase`; it feeds `gen_utils._NanotubeTemplateDB`.

## Training vs. runtime dependency footprint

| Task | Needs |
|---|---|
| `build_templates.py` | `numpy` (+ optional `ase`; self-stubs if absent) |
| Generation runtime | `torch`, `torch_geometric`, the model, `nanotube_templates.npz` (no `ase`) |
| Training | full stack + Hydra + PyTorch Lightning + a dataset |
| Screening | `gnn_eval` deps (`e3nn`, etc.) + trained classifier weights |

Environment caveats (torch_geometric / GPU availability) are in [usage/setup.md](../usage/setup.md).

## Next

- The package internals → [components/core-package-ntgent.md](../components/core-package-ntgent.md)
- The config system → [components/configuration.md](../components/configuration.md)

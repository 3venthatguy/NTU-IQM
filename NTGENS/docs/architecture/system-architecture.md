# System Architecture

> **Audience:** both. **Purpose:** the layered view of NTGENS and each layer's responsibility.
> ⬅ Back to [docs hub](../README.md)

## The layers

NTGENS is organized as a stack. Higher layers depend on lower ones; the constraint/generation layer is the "glue" that sits on top of the core ML package and turns it into a nanotube generator.

```
┌──────────────────────────────────────────────────────────────────────┐
│  ENTRY POINTS                                                          │
│  ../comp_models/NTGEN_generation/*.ipynb   gen_mul.py   screen_mul.py  │
│  (notebooks + editable batch drivers — no single main.py)             │
└───────────────┬──────────────────────────────────────────────────────┘
                │ shell out / import
┌───────────────▼──────────────────────────────────────────────────────┐
│  SCRIPT LAYER  (script/)                                              │
│   generation.py  → run constrained sampling → eval_gen_<label>.pt      │
│   gen_utils.py   → SampleDataset (builds the pinned batch)            │
│   sc_utils.py    → SC_* constraint classes (the known skeleton)       │
│   sc_natm.py     → atom-count distributions                          │
│   save_cif.py / eval_screen.py / traj_movie.py / compute_metrics.py   │
│   eval_utils.py / eval_funcs.py / mat_utils.py  (helpers)             │
└───────────────┬───────────────────────────────┬──────────────────────┘
                │ import scigen.*                │ reads templates
┌───────────────▼───────────────────┐  ┌─────────▼──────────────────────┐
│  CORE ML PACKAGE  (ntgent/, aka   │  │  TEMPLATE DATA  (data/nano_1D/) │
│  `scigen` via symlink)            │  │   build_templates.py (ETL)     │
│   pl_modules/  diffusion, cspnet, │  │   → nanotube_templates.npz     │
│                gnn, energy, model │  │   consumed by gen_utils DB     │
│   pl_data/     dataset, datamodule│  └────────────────────────────────┘
│   common/      data_utils (geom), │
│                utils, constants   │
│   run.py       training entry     │
└───────────────┬───────────────────┘
                │ configured by
┌───────────────▼──────────────────────────────────────────────────────┐
│  CONFIG LAYER                                                          │
│   conf/ (Hydra: data/model/optim/train/logging)   .env                │
│   config_scigen.py (user copy)   gnn_eval/config_eval.py (user copy)   │
└──────────────────────────────────────────────────────────────────────┘

   POST-PROCESSING / SCREENING  (runs on eval_gen_<label>.pt)
   save_cif.py → CIFs   traj_movie.py → GIF   compute_metrics.py → metrics
   eval_screen.py ──uses──▶  gnn_eval/  (3 trained e3nn classifiers)
```

## Layer responsibilities

### Config layer
Everything that parameterizes a run without touching code. Hydra configs (`conf/`) drive **training**; the editable Python config `config_scigen.py` and CLI flags drive **generation/screening**; `.env` supplies `PROJECT_ROOT`, `HYDRA_JOBS`, `WANDB_DIR`. → [components/configuration.md](../components/configuration.md).

### Core ML package (`ntgent/`, imported as `scigen`)
The DiffCSP-derived model and its data plumbing. Knows nothing about nanotubes — it diffuses lattice + fractional coords + atom types, and (via `sample_scigen`) supports the pinned/known-atom inpainting protocol. → [components/core-package-ntgent.md](../components/core-package-ntgent.md), [components/diffusion-model.md](../components/diffusion-model.md).

> **Critical:** the package directory is `ntgent/` but every import says `scigen`. A `scigen → ntgent` symlink bridges this. If it is missing, nothing runs. → [architecture/module-dependencies.md](module-dependencies.md).

### Script layer (`script/`)
The nanotube-specific business logic layered on top of the model: define the known skeleton (`sc_utils.py`), assemble it into a batch of pinned conditioning data (`gen_utils.py`), run the sampler (`generation.py`), and post-process. → [components/structural-constraints.md](../components/structural-constraints.md), [components/generation-scripts.md](../components/generation-scripts.md).

### Template data (`data/nano_1D/`)
The real-structure database feature. A one-time ETL (`build_templates.py`) compresses ~7000 ASE nanotube structures into an mmap-friendly `.npz`; at runtime the `shl` constraint samples real tube geometries from it. → [components/nanotube-template-db.md](../components/nanotube-template-db.md).

### Post-processing & screening
Operates purely on the `eval_gen_<label>.pt` bundle that generation produces: CIF export, trajectory movies, benchmark metrics, and a GNN-classifier screening cascade (`gnn_eval/`). → [components/gnn-screening.md](../components/gnn-screening.md).

## Design characteristic: no unified CLI

This is a **research codebase**, not an application. There is intentionally no `main.py`. Instead, three styles of entry point coexist:

- **Editable "config scripts"** you copy and edit, then run (`config_scigen.py`, `gen_mul.py`, `screen_mul.py`).
- **argparse CLIs** under `script/` invoked with flags (`python script/generation.py --sc shl …`).
- **Jupyter notebooks** (in `../comp_models/NTGEN_generation/`) as the friendliest driver for the full nanotube workflow.

This matters for agents: to "run generation," you either edit-and-run `gen_mul.py` or call `script/generation.py` directly — see [usage/workflows.md](../usage/workflows.md).

## Next

- Trace a structure through the stack → [architecture/data-flow.md](data-flow.md)
- Understand imports and the symlink → [architecture/module-dependencies.md](module-dependencies.md)

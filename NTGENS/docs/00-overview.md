# Overview

> **Audience:** both. **Purpose:** the big picture — what NTGENS is, its lineage, and where it lives.
> ⬅ Back to [docs hub](README.md)

## What NTGENS is

NTGENS generates **1D nanotube crystal structures** with a **constrained diffusion model**. The core idea (inherited from SCIGEN):

1. Build a **known skeleton** — a set of atoms whose positions and species are fixed (e.g. a ring of atoms around a tube axis, or a real structure from a database).
2. Run a **diffusion model** that, at every reverse-diffusion step, keeps the known atoms pinned and denoises the *unknown* atoms around them — i.e. **inpainting** for crystals.
3. The result is a full crystal structure that provably contains the target geometric motif (here: a nanotube), decorated with model-generated atoms.

The output is a set of structures saved as a `.pt` tensor bundle, convertible to CIF files and screenable with GNN classifiers before any expensive DFT. See [technical-foundations.md](technical-foundations.md) for the math and [architecture/data-flow.md](architecture/data-flow.md) for the pipeline.

## Lineage: DiffCSP → SCIGEN → NTGEN

| Layer | What it contributed | Where it lives now |
|---|---|---|
| **DiffCSP** | The base crystal-structure diffusion model: joint diffusion over lattice, fractional coordinates, and atom types; the CSPNet decoder. | The `ntgent/` package (imported as `scigen`) is a DiffCSP derivative. A pristine copy sits in the sibling `../comp_models/DiffCSP-main/`. |
| **SCIGEN** | *Structural Constraint Integration in the GENerative model*: the `sample_scigen` inpainting sampler and the `SC_*` constraint classes that pin known atoms into 2D lattice motifs (honeycomb, kagome, …). Published in *Nature Materials* 2025 (arXiv:2407.04557). | `ntgent/pl_modules/diffusion_w_type.py::sample_scigen` + `script/sc_utils.py`. |
| **NTGEN** (this repo) | Specialized SCIGEN to **nanotubes only**: replaced the 2D-lattice constraints with 1D tube constraints, and added a real-structure template database (Alexandria 1D). | `script/sc_utils.py` `sc_dict`, `data/alx_1D/`. |

## The four constraints (`sc_dict`)

NTGEN is **nanotube-only**. The dispatch table in [`script/sc_utils.py`](../script/sc_utils.py) (bottom of file) is:

```python
sc_dict = {'ntb': SC_Nanotube,   'cnt': SC_CarbonTube,
           'alx': SC_DBTemplate,  'van': SC_Vanilla}
```

| Key | Class | What it pins |
|---|---|---|
| `ntb` | `SC_Nanotube` | A **parametric** skeleton ring — `n_circ` atoms of one element around a tube axis. |
| `cnt` | `SC_CarbonTube` | A **full rolled-graphene** carbon-nanotube wall for chiral indices `(n,m)`. Species forced to `C`. |
| `alx` | `SC_DBTemplate` | A **real** multi-element nanotube structure loaded from the Alexandria 1D database. |
| `van` | `SC_Vanilla` | **No constraint** — plain unconditional diffusion (baseline). |

> The upstream 2D-lattice constraints (`tri`, `hon`, `kag`, …) were **deliberately removed**. The top-level `README.md` still documents them — see [known-discrepancies.md](known-discrepancies.md) §1. Details of each class in [components/structural-constraints.md](components/structural-constraints.md).

## Where NTGENS sits in the `NTU-IQM` repo

```
NTU-IQM/
├── NTGENS/          ← this codebase (the NTGEN model + scripts + data)
├── comp_models/     ← companion material:
│   ├── DiffCSP-main/         upstream, unmodified DiffCSP reference
│   └── NTGEN_generation/     the driver notebooks:
│       ├── ntgen_generation.ipynb    (sc='alx', Alexandria templates + mp_20)
│       ├── 05_ctgen_generation.ipynb (sc='cnt', carbon_24)
│       └── 04_scigen_generation.ipynb (original 2D-lattice SCIGEN demo)
└── README.md
```

The **notebooks in `../comp_models/NTGEN_generation/` are the primary interactive entry points** that drive this code, but they assume an older `NTGEN-edit` folder name — see [known-discrepancies.md](known-discrepancies.md) §3.

## What this repo is *not*

- **Not a packaged app.** There is no unified `main.py`. It's a research codebase driven by editable config scripts (`config_scigen.py`, `gen_mul.py`), argparse CLIs under `script/`, and notebooks. See [architecture/system-architecture.md](architecture/system-architecture.md).
- **Not test-covered.** `pytest` is listed as a dependency but there is no test suite. See [known-discrepancies.md](known-discrepancies.md) §5 and the manual-testing recipe in [usage/extending.md](usage/extending.md).

## Next

- New to the design? → [architecture/system-architecture.md](architecture/system-architecture.md)
- Want to run it? → [usage/setup.md](usage/setup.md)

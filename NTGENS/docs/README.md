# NTGENS Documentation

> **Audience:** both LLM agents and human programmers.
> **Purpose:** a navigable, self-contained map of every component in `NTGENS/` — what it is, how it works, and how to run/extend it.

`NTGENS/` is the **NTGEN** codebase: a nanotube-specialized fork of **SCIGEN**, which is itself built on **DiffCSP**. It uses a constrained diffusion model to generate 1D nanotube crystal structures by *pinning* a known atomic skeleton and letting the diffusion model "inpaint" the rest.

These docs describe **the code as it actually behaves today**. Where existing artifacts in the repo (the top-level `README.md`, `screen_mul.py`, the sibling notebooks) disagree with the current code, that drift is called out in [known-discrepancies.md](known-discrepancies.md) rather than silently propagated — read that doc before trusting any older documentation in this folder.

> **Snapshot note:** at the time of writing, `NTGENS/` is an *untracked* top-level directory, byte-identical to the last committed `models/NTGEN-edit/`. File/line references were verified against the working tree — re-check a symbol if the code has since moved. See [known-discrepancies.md](known-discrepancies.md) §4.

---

## Start here — reading paths

**Agent path** (understand the system fast):
1. [00-overview.md](00-overview.md) — what NTGENS is and where it sits.
2. [architecture/system-architecture.md](architecture/system-architecture.md) — the layers.
3. [architecture/data-flow.md](architecture/data-flow.md) — how a structure is actually produced.
4. Jump to the relevant [components/](components/) doc for the part you're touching.

**Programmer path** (get it running / extend it):
1. [usage/setup.md](usage/setup.md) — env, the critical `scigen → ntgent` symlink, configs.
2. [usage/workflows.md](usage/workflows.md) — train / generate / screen / visualize commands.
3. [usage/inspecting-outputs.md](usage/inspecting-outputs.md) — sanity-check generated structures before export.
4. [usage/extending.md](usage/extending.md) — add your own constraint; gotchas to avoid.
5. [technical-foundations.md](technical-foundations.md) — the "why" behind the ML/physics.

---

## Full map

```
docs/
├── README.md                     ← you are here (hub + navigation)
├── 00-overview.md                What NTGENS is, lineage (DiffCSP→SCIGEN→NTGEN), repo layout
├── architecture/
│   ├── system-architecture.md    The layered component map + diagram
│   ├── data-flow.md              End-to-end pipeline, stage by stage
│   └── module-dependencies.md    Import graph + the scigen→ntgent symlink
├── components/
│   ├── core-package-ntgent.md    ntgent/ package (common/, pl_data/, run.py)
│   ├── diffusion-model.md         pl_modules/ — CSPDiffusion, sample_scigen, cspnet
│   ├── structural-constraints.md  script/sc_utils.py — the SC_* class hierarchy
│   ├── generation-scripts.md      script/ CLI layer (generation, save_cif, screen, metrics)
│   ├── nanotube-template-db.md    data/nano_1D/ — build_templates + npz cache + DB loader
│   ├── gnn-screening.md           gnn_eval/ classifier sub-project
│   └── configuration.md           conf/ Hydra tree, .env, config_scigen.py
├── technical-foundations.md       Diffusion, wrapped-normal, inpainting, nanotube geometry
├── usage/
│   ├── setup.md                   Environment, symlink, config copies, checkpoint
│   ├── workflows.md               Runnable command walkthroughs
│   ├── inspecting-outputs.md      Tensors → pymatgen → lattice/space-group/XRD sanity checks
│   └── extending.md               Add a constraint; conventions & gotchas
├── glossary.md                    Domain + code terms in one place
└── known-discrepancies.md         Stale artifacts and how to interpret them
```

## One-line index

| Doc | Read it when you need to… |
|---|---|
| [00-overview.md](00-overview.md) | grasp the big picture and lineage |
| [architecture/system-architecture.md](architecture/system-architecture.md) | see how the layers fit together |
| [architecture/data-flow.md](architecture/data-flow.md) | trace one structure from template to CIF |
| [architecture/module-dependencies.md](architecture/module-dependencies.md) | understand imports and the symlink |
| [components/core-package-ntgent.md](components/core-package-ntgent.md) | work on data loading / geometry / training loop |
| [components/diffusion-model.md](components/diffusion-model.md) | understand or modify the diffusion model |
| [components/structural-constraints.md](components/structural-constraints.md) | understand the `SC_*` constraint classes |
| [components/generation-scripts.md](components/generation-scripts.md) | run or modify generation/eval scripts |
| [components/nanotube-template-db.md](components/nanotube-template-db.md) | work on the Alexandria template DB |
| [components/gnn-screening.md](components/gnn-screening.md) | screen outputs with GNN classifiers |
| [components/configuration.md](components/configuration.md) | change Hydra configs or env |
| [technical-foundations.md](technical-foundations.md) | understand the math/physics |
| [usage/setup.md](usage/setup.md) | set up the environment |
| [usage/workflows.md](usage/workflows.md) | run an end-to-end task |
| [usage/inspecting-outputs.md](usage/inspecting-outputs.md) | sanity-check generated structures (lattice/space-group/XRD) |
| [usage/extending.md](usage/extending.md) | add a constraint / avoid gotchas |
| [glossary.md](glossary.md) | look up a term |
| [known-discrepancies.md](known-discrepancies.md) | avoid being misled by stale docs |

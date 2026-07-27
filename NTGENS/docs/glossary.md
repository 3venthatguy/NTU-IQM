# Glossary

> **Audience:** both. **Purpose:** domain + code terms in one place.
> ⬅ Back to [docs hub](README.md)

| Term | Meaning |
|---|---|
| **DiffCSP** | Diffusion for Crystal Structure Prediction — the base model NTGEN derives from. Joint diffusion over lattice, fractional coords, atom types. Pristine copy in `../comp_models/DiffCSP-main/`. |
| **SCIGEN** | *Structural Constraint Integration in the GENerative model*. Adds the inpainting sampler + `SC_*` constraint classes on top of DiffCSP. arXiv:2407.04557. |
| **NTGEN** | This repo: SCIGEN specialized to **nanotubes only** (`shl`/`van`), plus a real-structure template DB. |
| **CSP** | Crystal Structure Prediction — predicting stable atomic arrangements. |
| **CSPDiffusion** | The diffusion model class actually used, in `ntgent/pl_modules/diffusion_w_type.py`. |
| **DDPM** | Denoising Diffusion Probabilistic Model — the noise-then-denoise generative framework. |
| **inpainting** | Fixing known parts and letting the model fill the rest. Here: pin skeleton atoms, generate the others. → [technical-foundations.md](technical-foundations.md) §2. |
| **`sample_scigen`** | The constrained reverse-diffusion sampler (module-level fn, monkey-patched onto the model). Recombines known/unknown via masks each step. |
| **known / unknown atoms** | Known = pinned skeleton (mask=1); unknown = model-generated decorators (mask=0). |
| **`mask_x` / `mask_t` / `mask_l`** | Per-sample masks marking fixed fractional coords `(N,3)`, atom types `(N,)`, lattice entries `(3,3)`. 1 = pinned. |
| **wrapped normal** | A normal distribution on a periodic domain; used for fractional-coordinate diffusion (`diff_utils.py`). |
| **Pathway 3** | The strategy of grounding generation in a **real** database structure (vs. synthesizing one). Implemented by `SC_DBShell` / the Alexandria DB — `shl` uses the real tube's geometry as a radial shell. |
| **Alexandria 1D DB** | Source of ~7002 real 1D-nanotube structures (`data/alx_1D/`), multi-element compounds. |
| **template cache (`.npz`)** | `nanotube_templates.npz` — CSR-packed, mmap-friendly, ASE-free cache built by `build_templates.py`. |
| **CSR arrays** | Compressed-Sparse-Row packing: flat concatenated arrays + a `splits` row-pointer to slice per structure. |
| **KDE bond length** | Kernel-density-estimated bond length per element (`data/kde_bond.pkl`), sampled at generation time. |
| **`metallic_radius`** | Dict of metallic radii for magnetic metals; alternative Gaussian bond-length source (`gen_utils.py`). |
| **SMACT** | Library for semiconducting-materials validity checks (charge neutrality, Pauling electronegativity). First screening filter. |
| **occupancy ratio** | Density check in screening; cells with ratio ≥ 1.7 are rejected. |
| **e_above_hull / e_hull** | Energy above the convex hull — a thermodynamic stability measure. `train_stab_ehull.py` classifies < 0.1 eV. |
| **e3nn** | Euclidean-equivariant neural network library used by the `gnn_eval` classifiers. |
| **Hydra** | Config framework composing `conf/` groups; drives training via `scigen/run.py`. |
| **PyG `Data`** | `torch_geometric.data.Data` — the graph object carrying a crystal (coords, types, cell, masks). |
| **lattice_scaler / prop_scaler** | Fitted `StandardScaler`s (persisted `.pt`) normalizing lattice params and the training property. |
| **`natm_range` / `natm_dist`** | Atom-count bounds (CLI) and empirical atom-count distributions (`sc_natm.py`) used to sample `num_atom`. |
| **`MAX_ATOMIC_NUM`** | `= 100`; upper bound on atom-type classes (`diffusion_w_type.py`). |
| **`MAX_NATM`** | `= 64`; the template-cache atom-count ceiling in `build_templates.py`. |
| **decorator atoms** | The `num_atom - num_known` unknown atoms the model generates around the skeleton. |
| **CSPNet** | The message-passing decoder network (`cspnet.py`) inside `CSPDiffusion`. |
| **`eval_gen_<label>.pt`** | The saved generation output bundle; input to every post-processing script. |

## Next

- The concepts in context → [technical-foundations.md](technical-foundations.md)
- Back to the [docs hub](README.md)

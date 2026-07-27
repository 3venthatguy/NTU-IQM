# Technical Foundations

> **Audience:** both (the "why"). **Purpose:** the ML + physics underneath, each concept linked to the file that implements it.
> ⬅ Back to [docs hub](README.md)

## 1. Diffusion generative models (DDPM), for crystals

A diffusion model learns to reverse a gradual noising process. **Forward:** start from a real structure and progressively add noise over `T` timesteps (here `timesteps: 1000`) until it's pure noise. **Reverse:** a network learns to denoise one step at a time, so sampling = start from noise and denoise down to a clean structure.

For a **crystal**, "the structure" is three coupled quantities, and NTGEN diffuses all three jointly (`CSPDiffusion.forward`, [diffusion-model.md](components/diffusion-model.md)):

| Quantity | Space | Noising |
|---|---|---|
| Lattice (`3×3` cell) | ℝ⁹ | Gaussian, cosine β-schedule |
| Fractional coordinates | periodic torus `[0,1)³` | **wrapped normal** (see §3) |
| Atom types | categorical | one-hot + Gaussian |

Three MSE losses (coord / lattice / type) are combined with weights `1 / 1 / 20` (`cost_*` in [`conf/model/diffusion_w_type.yaml`](../conf/model/diffusion_w_type.yaml)) — atom-type prediction is weighted heavily.

## 2. Constrained generation as inpainting

This is the SCIGEN contribution and the reason NTGEN can *guarantee* a nanotube motif. Classic image inpainting: fix known pixels, let the diffusion model fill the rest. Here the "known pixels" are the **skeleton atoms** of a nanotube.

At every reverse step, `sample_scigen` overwrites the known parts with their pinned values (renoised to the current noise level) and keeps only the model's prediction for the unknown parts:

```
x_t = mask_x ⊙ x_0_known + (1 - mask_x) ⊙ x_unknown     # fractional coords
l_t = mask_l ⊙ l_0_known + (1 - mask_l) ⊙ l_unknown     # lattice
t_t = mask_t ⊙ t_0_known + (1 - mask_t) ⊙ t_unknown     # atom types
```

Because the known atoms are re-imposed at *every* timestep, they never drift — the final structure provably contains the skeleton, while the model harmonizes the surrounding (unknown) atoms to be physically plausible. The masks come from the `SC_*` classes ([structural-constraints.md](components/structural-constraints.md)); the recombination is in `sample_scigen` ([diffusion-model.md](components/diffusion-model.md)).

## 3. Why wrapped-normal for fractional coordinates

Fractional coordinates live on a **periodic** domain: `0.99` and `0.01` are close (they wrap around the cell). A plain Gaussian doesn't respect this. NTGEN uses a **wrapped normal** distribution and score-matches against it via `p_wrapped_normal` / `d_log_p_wrapped_normal` in [`ntgent/pl_modules/diff_utils.py`](../ntgent/pl_modules/diff_utils.py). This is standard for DiffCSP-family models and is what makes coordinate diffusion behave correctly under periodic boundary conditions.

## 4. "Pathway 3": real structures as geometry templates

NTGEN offers two ways to shape generation:
- **Database template ("Pathway 3")** — draw an *actual* structure from the Alexandria 1D database and use its **geometry** (`SC_DBShell`, `shl`). This grounds generation in real, DFT-relaxed nanotube shapes instead of idealized ones, while leaving every atom for the model to generate. → [nanotube-template-db.md](components/nanotube-template-db.md).
- **None** (`SC_Vanilla`) — unconstrained baseline.

## 5. Nanotube geometry cheat-sheet

**Geometric shell (`SC_DBShell`).** The template's real cell fixes the tube frame — axis `a_hat` (the lattice direction the atoms fill most), transverse basis `e1, e2` (Gram-Schmidt, robust to non-orthogonal cells), and cross-section centroid. Each atom's transverse radius `r = |perp − centroid|` is confined to a band `[r_min, r_max]` measured (by percentile) from the template's own atom radii, so generation is "tube-shaped" without pinning any atom. → [structural-constraints.md](components/structural-constraints.md).

**Private synthetic-ring fallback (`_SC_NanotubeFallback`).** When no real template fits the atom-count range, `shl` falls back to `n_circ` atoms evenly spaced on a circle of radius `R = bond_len / (2·sin(π / n_circ))` (adjacent ring atoms one `bond_len` apart), tube axis `c` (periodic, vertical), `a,b` a vacuum box (`2R + vacuum`). Not a user-facing mode.

## 6. Post-generation validity screening

Generated structures are filtered before DFT ([gnn-screening.md](components/gnn-screening.md)):
- **SMACT validity** — charge-neutrality + electronegativity (Pauling) sanity via `smact`.
- **Occupancy ratio** — reject over-dense cells (< 1.7).
- **GNN classifiers** — e3nn equivariant networks predicting pristine-vs-diffused, e_hull stability, and magnetism.

## References

- SCIGEN: *Structural Constraint Integration in the GENerative model*, Nature Materials 2025, arXiv:2407.04557 (see top-level [`README.md`](../README.md)).
- DiffCSP: the base crystal-structure diffusion model (pristine copy in `../comp_models/DiffCSP-main/`).

## Next

- See it implemented → [components/diffusion-model.md](components/diffusion-model.md)
- Look up a term → [glossary.md](glossary.md)

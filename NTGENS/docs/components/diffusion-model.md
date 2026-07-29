# Diffusion Model: `ntgent/pl_modules/`

> **Audience:** programmer. **Purpose:** the model that does the generating — training forward pass and the constrained `sample_scigen` sampler.
> ⬅ Back to [docs hub](../README.md) · Related: [technical-foundations.md](../technical-foundations.md), [data-flow.md](../architecture/data-flow.md)

```
ntgent/pl_modules/
├── diffusion_w_type.py   CSPDiffusion + sample_scigen  ← the model actually used
├── cspnet.py             CSPNet decoder / CSPLayer message passing
├── diff_utils.py         noise schedules + wrapped-normal score functions
├── gnn.py                DimeNet++-style encoder (from OCP)
├── energy_model.py       energy-guided diffusion variant
└── model.py              CrystGNN_Supervise (property-prediction model)
```

## `diffusion_w_type.py` — the heart

`MAX_ATOMIC_NUM = 100` (module constant, imported by `sc_utils.py`).

### `class CSPDiffusion(BaseModule)`
Joint diffusion over three quantities of a crystal:
- **lattice** (the `3×3` cell matrix),
- **fractional coordinates** (periodic → uses wrapped-normal score matching),
- **atom types** (one-hot + Gaussian noise).

It instantiates a decoder (`cspnet`, via Hydra `_target_`), a `beta_scheduler` (cosine) and `sigma_scheduler` (wrapped), and a `SinusoidalTimeEmbeddings`.

**`forward(batch)`** — the training step: noise each quantity per its schedule, predict the noise/score, and compute three MSE losses combined with weights from [`conf/model/diffusion_w_type.yaml`](../../conf/model/diffusion_w_type.yaml):

| Loss | Weight (`cost_*`) |
|---|---|
| `loss_coord` | `cost_coord = 1.0` |
| `loss_lattice` | `cost_lattice = 1.0` |
| `loss_type` | `cost_type = 20.0` |

(`training_step` / `validation_step` / `test_step` / `compute_stats` are standard PL hooks.)

### `sample_scigen(self, batch, diff_ratio=1.0, step_lr=1e-5)` — the constrained sampler
Defined at **module level** (not a method) and monkey-patched onto a model instance at generation time:
```python
model.sample_scigen = sample_scigen.__get__(model)   # done in script/generation.py
```

This is the **SCIGEN inpainting sampler**. At every reverse-diffusion timestep it splits each quantity into a **known** part (pinned from the batch's constraint data) and an **unknown** part (regular predictor-corrector / annealed Langevin denoising), then recombines them via the masks:

```
x_t = mask_x * x_0_known + (1 - mask_x) * x_unknown      # fractional coords
l_t = mask_l * l_0_known + (1 - mask_l) * l_unknown      # lattice
t_t = mask_t * t_0_known + (1 - mask_t) * t_unknown      # atom types
```

The known atoms therefore **never drift** — the diffusion only fills in the unknown atoms around the fixed skeleton. It returns `(final_structure, full_trajectory)` — the trajectory is what `traj_movie.py` renders. The known quantities (`x_0_known`, `l_0_known`, `t_0_known`) and masks come from the PyG `Data` built by [`SampleDataset`](generation-scripts.md); `t_0_known` is `F.one_hot(batch.atom_types_known - 1, num_classes=MAX_ATOMIC_NUM)`.

> Why this equals inpainting, and the wrapped-normal detail for periodic coordinates, are in [technical-foundations.md](../technical-foundations.md).

## Supporting modules

### `cspnet.py` — the decoder
The message-passing backbone used as the diffusion decoder (`conf/model/decoder/cspnet.yaml`). Defines `SinusoidsEmbedding` (Fourier positional embedding) and `CSPLayer` (combines node hidden state, edge-distance embedding, and the `3×3` lattice matrix into an equivariant-ish update), assembled into `CSPNet`.

### `diff_utils.py` — schedules, periodic scores & geometric guidance
Noise schedules (`cosine_beta_schedule`, `linear_beta_schedule`, `quadratic_beta_schedule`, `sigmoid_beta_schedule`) and the wrapped-normal functions `p_wrapped_normal(x, sigma, N=10, T=1.0)` / `d_log_p_wrapped_normal(...)` used for diffusing **fractional coordinates** (which live on a periodic torus, not ℝ³).

It also holds the constrained-sampling machinery `sample_scigen` calls each step (pure torch — no `torch_scatter`, so it is unit-testable standalone):

| Function | Role |
|---|---|
| `pinning_strength(t, T, cfg)` (+ `_linear` / `_sigmoid`) | the ramp `ψ(t)`; returns 1.0 when disabled ⇒ original binary pinning |
| `lattice_tube_frame(lat, axis, centroid_frac)` | rebuilds `(a_hat, e1, e2, ctr, area, pn)` from the **current** lattice |
| `transverse_scale(area, pn, area_ref, pn_ref)` | `s_t` = current/template transverse scale, + a degenerate-cell gate |
| `apply_radial_band(...)` | confines `r` into `[r_lo, r_hi]` — **θ-invariant** |
| `apply_density_force(...)` | steps `r` up `d/dr log ρ(r)` — **θ-invariant** |
| `apply_angular_spread(...)` | rotates about the axis to reduce the circular resultant — **θ only**, preserves `r`/`z` exactly |

Why the frame must track `lat`, and why `ψ` is split into `psi` / `psi_geom`: [technical-foundations.md](../technical-foundations.md) §5b.

### `gnn.py` — alternative encoder
A DimeNet++-style encoder ("Adapted from the Open Catalyst Project"): `InteractionPPBlock`, `BesselBasisLayer`, `SphericalBasisLayer`, etc. Used by the supervised property model.

### `energy_model.py` — energy-guided variant
A parallel diffusion variant with energy guidance (imports `scipy.optimize.linear_sum_assignment` → Hungarian-matching loss). Same `BaseModule` boilerplate. Not the default path.

### `model.py` — `CrystGNN_Supervise`
A supervised GNN **regression** model (encoder-only) for property prediction (e.g. formation energy, lattice params), with `compute_stats` producing MAE/MARD metrics including a special-cased lattice-parameter decomposition.

## Which model runs?

The **used** model is `CSPDiffusion` (`model=diffusion_w_type`). Note the config *default* is `model: diffusion` (see [`conf/default.yaml`](../../conf/default.yaml)); every real command overrides it to `diffusion_w_type`. See [configuration.md](configuration.md) and [known-discrepancies.md](../known-discrepancies.md).

## Next

- Where the pinned skeleton comes from → [structural-constraints.md](structural-constraints.md)
- The math → [technical-foundations.md](../technical-foundations.md)

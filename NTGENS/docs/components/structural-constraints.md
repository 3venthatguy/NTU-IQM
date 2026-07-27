# Structural Constraints: `script/sc_utils.py`

> **Audience:** programmer. **Purpose:** the `SC_*` class hierarchy that builds the pinned known skeleton for each constraint type.
> ⬅ Back to [docs hub](../README.md) · Related: [diffusion-model.md](diffusion-model.md), [extending.md](../usage/extending.md)

Every constraint is a class in [`script/sc_utils.py`](../../script/sc_utils.py). Its job: produce a **known skeleton** — fixed fractional coordinates, fixed atom types, a cell — plus the three masks that tell the sampler which parts to pin. The dispatch table at the bottom of the file:

```python
sc_dict = {'shl': SC_DBShell, 'van': SC_Vanilla}
```

## Class hierarchy

```
SC_Base                     (abstract: cell building, mask selection, fill methods)
├── SC_Vanilla   'van'      no constraint (baseline)
├── SC_Template  (—)        copy-me scaffold for new constraints (not in sc_dict)
├── SC_DBShell   'shl'      real tube GEOMETRY from the Alexandria 1D DB (no atoms pinned)
└── _SC_NanotubeFallback (—) private synthetic ring — shl's fallback, not in sc_dict
```

> The earlier atom-pinning constraints — `SC_Nanotube` (`ntb`), `SC_CarbonTube`
> (`cnt`), and `SC_DBTemplate` (`alx`) — were **removed**. They pinned most or all
> atoms as a known skeleton, leaving little or nothing for the model to generate.
> `SC_DBShell` keeps only the tube's *shape*. The ring geometry of the old
> `SC_Nanotube` survives, renamed `_SC_NanotubeFallback`, purely as `shl`'s
> synthetic safety net when no real template fits the requested atom-count range.

## `SC_Base` — the contract

Constructor: `(bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device)`. Key methods:

- **`get_cell(alpha, beta, gamma)`** — builds the lattice matrix from `a_scale/b_scale/c_scale × bond_len` (via `lattice_params_to_matrix_xy_torch`).
- **`get_mask_l()`** — picks which lattice-matrix entries are pinned, based on `c_vec_cons` (`{'scale', 'vert'}`) and `reduced_mask`. Returns one of the `mask_l_*` constants.
- **`frac_coords_all()`** — writes `self.frac_coords` (known atoms first, rest zeros) and `self.mask_x` (1 for known coords). Wraps coords into `[0,1)`.
- **`atm_types_all()`** — writes `self.atom_types` (known species first, then **randomly-typed** decorator atoms) and `self.mask_t` (1 for known types).

Subclasses set `a_scale`/`b_scale`, call `get_cell()`, and populate `frac_known` + `num_known`. The `num_atom - num_known` trailing atoms are the "unknown" ones the model generates.

### The `mask_l_*` lattice masks
Constants at the top of the file; 1 = that lattice-matrix entry is fixed/known:

| Constant | Meaning |
|---|---|
| `mask_l_cvert` | a,b rows fixed; c-vector kept **vertical** with free length. |
| `mask_l_allfixed` | entire cell fixed `(1,3,3)`. **Used by `SC_DBShell`** — the template's real cell is ground truth. |
| `mask_l_default` | a,b fixed, c free |
| `mask_l_full` | entire cell fixed |
| `mask_l_zeros` | nothing fixed (used by `SC_Vanilla`) |
| `mask_l_reduced` / `..._full` | reduced `(1,3)` variants (legacy) |

## The constraints

### `SC_DBShell` (`shl`) — real tube geometry, no atoms pinned
Draws a **real** nanotube from the Alexandria 1D DB but pins **only its geometry**, not its atoms. Constructor takes `frac_known`, `atom_numbers_known`, `cell` (supplied by [`gen_utils.nanotube_template_from_db`](generation-scripts.md)) but **discards the atoms**: `num_known = 0`, so `mask_x` / `mask_t` are all-zero and the diffusion model generates every atom's position **and** species freely. The template's real cell is fixed (`mask_l_allfixed`) so the tube axis, axial period, and vacuum box define a meaningful `(r, θ, z)` frame. `gen_utils` measures a radial band `[r_min, r_max]` (and an optional per-template log-density guidance force) from the same template and carries it on the batch; the sampler softly, `ψ(t)`-ramped, confines every atom onto the wall shell during denoising — enforcing "tube-shaped" without dictating what fills it. Target atom count = the drawn tube's real `nsites`. Requires `reduced_mask=False`.

> **Fallback:** if no template fits the requested atom-count range, `SampleDataset` swaps `shl` for the private `_SC_NanotubeFallback` (synthetic ring geometry). It is **not** a registered mode. See [data-flow.md](../architecture/data-flow.md) stage 2.

### `SC_Vanilla` (`van`) — no constraint
`use_constraints = False`: a single dummy known atom, `mask_l = zeros`, and both `mask_x` / `mask_t` zeroed in the fill methods. This yields plain unconditional diffusion — the baseline.

### `SC_Template` — the scaffold (not registered)
A documented copy-me class (two known atoms, `#TODO` placeholders for lattice scaling, cell angle, and fractional coords). Copy it to add a new constraint — see [usage/extending.md](../usage/extending.md).

## How a constraint is used at generation time

`SampleDataset.process()` ([gen_utils.py](generation-scripts.md)) instantiates the right class, samples `num_atom` from an empirical distribution truncated to leave room for decorators, then calls `frac_coords_all()` and `atm_types_all()`. The resulting `frac_coords`, `atom_types`, `cell`, and masks flow into the PyG `Data` batch consumed by `sample_scigen`.

## Next

- How the batch is assembled → [generation-scripts.md](generation-scripts.md)
- Add your own constraint → [usage/extending.md](../usage/extending.md)

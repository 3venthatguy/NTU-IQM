# Structural Constraints: `script/sc_utils.py`

> **Audience:** programmer. **Purpose:** the `SC_*` class hierarchy that builds the pinned known skeleton for each constraint type.
> ⬅ Back to [docs hub](../README.md) · Related: [diffusion-model.md](diffusion-model.md), [extending.md](../usage/extending.md)

Every constraint is a class in [`script/sc_utils.py`](../../script/sc_utils.py). Its job: produce a **known skeleton** — fixed fractional coordinates, fixed atom types, a cell — plus the three masks that tell the sampler which parts to pin. The dispatch table at the bottom of the file:

```python
sc_dict = {'ntb': SC_Nanotube,   'cnt': SC_CarbonTube,
           'alx': SC_DBTemplate,  'van': SC_Vanilla}
```

## Class hierarchy

```
SC_Base                     (abstract: cell building, mask selection, fill methods)
├── SC_Vanilla   'van'      no constraint (baseline)
├── SC_Template  (—)        copy-me scaffold for new constraints (not in sc_dict)
├── SC_Nanotube  'ntb'      parametric skeleton ring
├── SC_CarbonTube 'cnt'     full rolled-graphene CNT wall
└── SC_DBTemplate 'alx'     real structure from the Alexandria 1D DB
```

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
| `mask_l_cvert` | a,b rows fixed; c-vector kept **vertical** with free length. **Used by all three tube constraints.** |
| `mask_l_default` | a,b fixed, c free |
| `mask_l_full` | entire cell fixed |
| `mask_l_zeros` | nothing fixed (used by `SC_Vanilla`) |
| `mask_l_reduced` / `..._full` | reduced `(1,3)` variants (legacy) |

## The constraints

### `SC_Nanotube` (`ntb`) — parametric ring
Pins `n_circ` atoms of one element on a ring of radius `R = bond_len / (2·sin(π/n_circ))` around the c/z axis, inside a large vacuum box so transverse periodic images don't interact. c is the periodic tube axis (vertical, free length via `mask_l_cvert`). Optional `chirality=(n,m)` adds a helical z-offset per ring atom.

Defaults (drawn when args are `None`):
```python
NANOTUBE_DEFAULTS = {'n_circ_range': (4, 10), 'vacuum': 15.0, 'axial_per_ring': 1.0}
```
Requires `reduced_mask=False` so `mask_x` stays `(N, 3)`.

### `SC_CarbonTube` (`cnt`) — rolled graphene wall
Builds the **full** CNT wall for chiral indices `(n,m)` via exact chiral-vector math: cut a graphene sheet along `Ch = n·a1 + m·a2` and the translational vector `T`, roll it so `Ch` becomes the circumference. All `num_wall = 4(n²+nm+m²)/d_R` wall atoms of one axial period are pinned as **carbon** (`type_known` is forced to `'C'` regardless of what was passed). Positions are exact integer rationals so periodic duplicates collapse cleanly (see `_wall_frac_coords` and its assertion).

Defaults:
```python
CARBONTUBE_DEFAULTS = {
    'chirality_options': [(3,3),(4,4),(5,5),(4,0),(5,0)],  # kept small so N ≤ 24
    'a_cc': 1.42, 'vacuum': 15.0}
```
The chirality options are chosen so wall-atom counts (12/16/20/16/20) stay under the `carbon_24` atom ceiling. Requires `reduced_mask=False`.

### `SC_DBTemplate` (`alx`) — real database structure
Pins an **actual** nanotube from the Alexandria 1D DB. Constructor takes `frac_known`, `atom_numbers_known`, `cell` (supplied by [`gen_utils.nanotube_template_from_db`](generation-scripts.md)). Because templates are **multi-element**, it **overrides `atm_types_all()`** to pin each atom's real species (atomic number = index into `chemical_symbols`) instead of a single `type_known`. Cell comes straight from the template; `mask_l_cvert`. Requires `reduced_mask=False`.

> **Fallback:** if no template fits the requested atom-count range, `SampleDataset` swaps `alx → ntb` (parametric). See [data-flow.md](../architecture/data-flow.md) stage 2.

### `SC_Vanilla` (`van`) — no constraint
`use_constraints = False`: a single dummy known atom, `mask_l = zeros`, and both `mask_x` / `mask_t` zeroed in the fill methods. This yields plain unconditional diffusion — the baseline.

### `SC_Template` — the scaffold (not registered)
A documented copy-me class (two known atoms, `#TODO` placeholders for lattice scaling, cell angle, and fractional coords). Copy it to add a new constraint — see [usage/extending.md](../usage/extending.md).

## How a constraint is used at generation time

`SampleDataset.process()` ([gen_utils.py](generation-scripts.md)) instantiates the right class, samples `num_atom` from an empirical distribution truncated to leave room for decorators, then calls `frac_coords_all()` and `atm_types_all()`. The resulting `frac_coords`, `atom_types`, `cell`, and masks flow into the PyG `Data` batch consumed by `sample_scigen`.

## Next

- How the batch is assembled → [generation-scripts.md](generation-scripts.md)
- Add your own constraint → [usage/extending.md](../usage/extending.md)

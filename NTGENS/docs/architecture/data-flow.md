# Data Flow

> **Audience:** both. **Purpose:** trace one structure end-to-end, naming the concrete artifact handed between each stage.
> ⬅ Back to [docs hub](../README.md)

## The pipeline at a glance

```
stage0_survivors.pkl (synthetic) + alexandria_direct_1d.json (real)   (raw sources)
        │  build_templates.py  (reduce → quality-filter → merge; run once)
        ▼
nanotube_templates.npz                              (CSR arrays, mmap-friendly)
        │  _NanotubeTemplateDB.sample()   [only for sc='shl']
        ▼
SC_* object  ── cell, radial band [r_min, r_max], frame + centroid_frac  (num_known=0 for shl)
        │  SampleDataset.process() + .generate_dataset()
        ▼
PyG `Data`  ── frac_coords_known, lattice_known, atom_types_known,
               mask_x, mask_l, mask_t, num_atoms, num_known
        │  model.sample_scigen(batch)   [constrained reverse diffusion]
        ▼
eval_gen_<label>.pt   ── frac_coords, atom_types, lengths, angles,
                          num_atoms, num_known, (+ trajectory if saved)
        │
        ├── save_cif.py        → .cif files (+ zip)
        ├── eval_screen.py     → screened .cif (via gnn_eval classifiers)
        ├── traj_movie.py      → .gif of the diffusion trajectory
        └── compute_metrics.py → validity / novelty / coverage metrics
```

## Stage by stage

### 1. Raw data → template cache (one-time, `shl` only)
Two sources — `stage0_survivors_structures.pkl` (synthetic) and `alexandria_direct_1d.json` (real, DFT) — are processed by [`build_templates.py`](../../data/nano_1D/build_templates.py): each tube is reduced to its simplest periodic cell (`reduce_templates`), quality-filtered (`filter_templates` — 5 gates: contacts, hollow core, radius ≤ 8 Å, atom count ≤ `MAX_NATM`=128, peaked ρ), deduped across sources, and packed into `nanotube_templates.npz`: CSR-style flat arrays (`numbers`, `frac_coords`, `cells`, `splits`, `natoms`, `source`). No ASE needed at runtime. → [components/nanotube-template-db.md](../components/nanotube-template-db.md).

### 2. Template → known skeleton (`SC_*` object)
[`SampleDataset.process()`](../../script/gen_utils.py) picks a constraint type from `sc_list` for each sample and instantiates the matching class from `sc_dict`:
- `shl`: calls `nanotube_template_from_db(natm_min, natm_max)` → a real `{frac_known, atom_numbers_known, cell}`, but `SC_DBShell` keeps only the `cell` and **discards the atoms** (`num_known=0`); `gen_utils` measures the radial band `[r_min, r_max]` from the template's atoms, plus the cylindrical frame and the cross-section centroid in **both** Cartesian (`centroid`, valid only at `t=0`) and **fractional** (`centroid_frac`, lattice-independent) form. If nothing fits the atom-count range, it **falls back to the private synthetic ring** `_SC_NanotubeFallback`.
- `van`: a single dummy atom, no constraint.

Each `SC_*` object exposes `cell` and — after `frac_coords_all()` / `atm_types_all()` — the full `frac_coords`, `atom_types`, and the masks `mask_x`, `mask_t`, `mask_l` (all-zero for `shl`, since it pins no atoms). → [components/structural-constraints.md](../components/structural-constraints.md).

**The number of atoms** (`num_atom`) for `shl` is the drawn tube's real `nsites`; otherwise it is sampled from an empirical per-dataset distribution (`sc_natm.natm_dist`), truncated so `num_known < num_atom ≤ natm_max`.

### 3. Skeleton → conditioning batch (PyG `Data`)
[`SampleDataset.generate_dataset()`](../../script/gen_utils.py) wraps each sample into a `torch_geometric.data.Data` carrying the *known* quantities plus the three masks. The masks are the heart of the constraint:

| Mask | Shape | Meaning: 1 = pinned/known, 0 = free/generated |
|---|---|---|
| `mask_x` | `(N, 3)` | which fractional coordinates are fixed |
| `mask_t` | `(N,)` | which atom types are fixed |
| `mask_l` | `(3, 3)` | which lattice-matrix entries are fixed |

> **Carbon special case:** when `dataset == 'carbon_24'`, `generate_dataset()` sets `data.atom_types = [6]*num_atom`, flattening all species to carbon. Fine if you specifically want all-carbon tubes, but it overrides the free species generation that `shl` otherwise does — use a general dataset (`mp_20`/`uniform`) for multi-element tubes. See [known-discrepancies.md](../known-discrepancies.md) and [usage/extending.md](../usage/extending.md).

### 4. Batch → structures (constrained reverse diffusion)
[`generation.py`](../../script/generation.py) loads the trained model, monkey-patches `model.sample_scigen`, and iterates the `SampleDataset` loader. For each batch, `sample_scigen` runs reverse diffusion where — at **every** timestep — the known quantities are re-imposed via the masks:

```
x_t = mask_x * x_0_known + (1 - mask_x) * x_unknown      # coords
l_t = mask_l * l_0_known + (1 - mask_l) * l_unknown      # lattice
t_t = mask_t * t_0_known + (1 - mask_t) * t_unknown      # types
```

This is the **inpainting** protocol: the known skeleton is never allowed to drift; only the unknown atoms are denoised. The masks are scaled by a ramp `ψ(t)` (`pin_cfg`), so with a schedule set the skeleton fades in rather than being frozen from `t=T`.

After each of the two update stages (corrector and predictor), `radial_envelope` applies the **geometric guidance** to `x_t` — never to `x_T`, which must stay a generic prior sample. It converts to Cartesian, rebuilds the tube frame from the **current** lattice `l_t` (`lattice_tube_frame`) and scales the band by `s_t` (`transverse_scale`), then applies the radial band, the density force, and optionally the angular-dispersion term, before converting back to fractional coords. Graphs with `is_alx=0` are gated out entirely — a zero frame would otherwise map all their atoms to the origin. The force gain uses `psi_geom`, which follows `pin_cfg` unless `geom_pin_cfg` is set. → [components/diffusion-model.md](../components/diffusion-model.md), [technical-foundations.md](../technical-foundations.md) §5b.

### 5. Structures → saved bundle
`generation.py` concatenates outputs, converts lattice matrices to `(lengths, angles)`, and `torch.save`s a dict to `<model_path>/eval_gen_<label>.pt`. Key fields: `frac_coords`, `atom_types`, `lengths`, `angles`, `num_atoms`, `num_known`, the guidance configs (`pin_cfg`, `cyl_cfg`, `dens_cfg`, `ang_cfg`, `geom_pin_cfg`), and (if `--save_traj`) the full per-timestep trajectory tensors.

### 6. Bundle → deliverables
The `.pt` bundle is the hand-off point for all downstream tools:
- `save_cif.py` → CIF files + a zip.
- `eval_screen.py` → filters via SMACT validity, occupancy ratio, then GNN classifiers → surviving CIFs. → [components/gnn-screening.md](../components/gnn-screening.md).
- `traj_movie.py` → GIF of the trajectory (needs `--save_traj` at generation time).
- `compute_metrics.py` → benchmark metrics (validity, novelty, uniqueness, coverage).

## Next

- The masks and sampler in detail → [components/diffusion-model.md](../components/diffusion-model.md)
- Run the pipeline yourself → [usage/workflows.md](../usage/workflows.md)

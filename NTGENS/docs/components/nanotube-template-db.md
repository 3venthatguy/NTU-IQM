# Nanotube Template Database: `data/alx_1D/`

> **Audience:** programmer. **Purpose:** the real-structure template pipeline behind the `shl` constraint.
> ⬅ Back to [docs hub](../README.md) · Related: [structural-constraints.md](structural-constraints.md), [generation-scripts.md](generation-scripts.md)

The `shl` constraint grounds generation in **real** nanotube geometry instead of synthesizing it. Those structures come from the Alexandria 1D database, preprocessed once into a compact cache. (`shl` uses the template's *cell/shape* to define a radial shell band; it does **not** pin the template's atoms — see [structural-constraints.md](structural-constraints.md).)

```
data/alx_1D/
├── alexandria_1d.json.bz2          raw source records (bzip2 JSON)
├── alexandria_1d_nanotubes.pkl     ~7002 ASE Atoms (multi-element, ~48 MB) — build input
├── build_templates.py              ETL: pkl → npz  (run once)
├── nanotube_templates.npz          compact CSR cache (~650 KB) — runtime input
└── __pycache__/
```

## The source: `alexandria_1d_nanotubes.pkl`
A pickled `list[ase.atoms.Atoms]` of ~7002 1D-nanotube structures. They are **multi-element compounds** (binary/ternary, e.g. Ta-Mn-Te, F-Cu), often hundreds of atoms, with an empty `info` dict (no chirality / n_circ metadata). `shl` uses their real cells to define the tube frame + radial band (it generates fresh atoms, so it is not tied to any single dataset's chemistry). See [known-discrepancies.md](../known-discrepancies.md) and [extending.md](../usage/extending.md).

## The ETL: `build_templates.py` (run once)
```bash
python data/alx_1D/build_templates.py
```
Reads the pkl, keeps every structure with `0 < natoms ≤ MAX_NATM` (=64) and a non-degenerate cell (`|det(cell)| ≥ 1e-6`), computes wrapped fractional coords (`pos @ inv(cell)`, mod 1), and writes `nanotube_templates.npz`. Prints a summary line (templates written, skipped, natoms range, size).

**The ASE-stub trick:** if `ase` is installed it unpickles normally; if not, it dynamically registers minimal stub modules (`ase`, `ase.atoms`, `ase.cell`, …) into `sys.modules` — tiny `_Stub` classes implementing only `__setstate__` — so the pickle loads **without a real ASE install**. `_numbers_positions_cell(atoms)` then extracts `(numbers, positions, cell)` from either a real `ase.Atoms` or a stub. This is why the runtime needs no ASE.

> `MAX_NATM = 64` is the **cache ceiling**, deliberately looser than the runtime `natm_range` filters applied downstream. The runtime filters tighter — see the note in [extending.md](../usage/extending.md).

## The cache: `nanotube_templates.npz`
CSR-style (compressed-sparse-row) packing so ragged per-structure arrays concatenate into flat arrays sliceable by structure:

| Array | Shape | Dtype | Meaning |
|---|---|---|---|
| `numbers` | `(ΣN,)` | int16 | atomic numbers, all structures concatenated |
| `frac_coords` | `(ΣN, 3)` | float32 | fractional coords, concatenated |
| `cells` | `(M, 3, 3)` | float32 | one cell per structure |
| `splits` | `(M+1,)` | int64 | row pointers: structure `i` = `[splits[i]:splits[i+1]]` |
| `natoms` | `(M,)` | int32 | atom count per structure |
| `max_natm` | scalar | int32 | the build ceiling (64) |

To read structure `i`: `numbers[splits[i]:splits[i+1]]`, `frac_coords[splits[i]:splits[i+1]]`, `cells[i]`.

## Runtime consumption: `_NanotubeTemplateDB`
In [`script/gen_utils.py`](../../script/gen_utils.py):
- `class _NanotubeTemplateDB(path='./data/alx_1D/nanotube_templates.npz')` — `np.load(..., mmap_mode='r')` so arrays stay on disk; buckets structure indices by atom count (`_by_natm`) for O(1) range queries. If the file is missing it prints a warning and `sc='shl'` falls back to the private synthetic ring (`_SC_NanotubeFallback`).
- `NANOTUBE_DB = _NanotubeTemplateDB()` — module-level singleton, loaded at import.
- `_NanotubeTemplateDB.sample(natm_min, natm_max)` — returns one random matching template `{frac_known, atom_numbers_known, cell}`, or `None`.
- `nanotube_template_from_db(natm_min, natm_max)` — wraps `.sample`, capping N at `natm_max - 1`; returns `{}` (→ fall back to `_SC_NanotubeFallback`) if nothing fits.

The returned kwargs are passed straight into `SC_DBShell(**extra_kwargs)`, which keeps the cell and **discards the atoms** — see [structural-constraints.md](structural-constraints.md).

## Next

- The `SC_DBShell` class → [structural-constraints.md](structural-constraints.md)
- Extending the pipeline → [usage/extending.md](../usage/extending.md)

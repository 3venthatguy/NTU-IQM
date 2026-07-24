# Nanotube Template Database: `data/alx_1D/`

> **Audience:** programmer. **Purpose:** the real-structure template pipeline behind the `alx` constraint.
> ⬅ Back to [docs hub](../README.md) · Related: [structural-constraints.md](structural-constraints.md), [generation-scripts.md](generation-scripts.md)

The `alx` constraint pins **real** nanotube structures instead of synthesizing geometry. Those structures come from the Alexandria 1D database, preprocessed once into a compact cache.

```
data/alx_1D/
├── alexandria_1d.json.bz2          raw source records (bzip2 JSON)
├── alexandria_1d_nanotubes.pkl     ~7002 ASE Atoms (multi-element, ~48 MB) — build input
├── build_templates.py              ETL: pkl → npz  (run once)
├── nanotube_templates.npz          compact CSR cache (~650 KB) — runtime input
└── __pycache__/
```

## The source: `alexandria_1d_nanotubes.pkl`
A pickled `list[ase.atoms.Atoms]` of ~7002 1D-nanotube structures. They are **multi-element compounds** (binary/ternary, e.g. Ta-Mn-Te, F-Cu), often hundreds of atoms, with an empty `info` dict (no chirality / n_circ metadata). This is why `alx` templates are used as *real pinned skeletons*, not as parametric parameters — and why they must run with a **general** dataset (mp_20/uniform), never carbon_24. See [known-discrepancies.md](../known-discrepancies.md) and [extending.md](../usage/extending.md).

## The ETL: `build_templates.py` (run once)
```bash
python data/alx_1D/build_templates.py
```
Reads the pkl, keeps every structure with `0 < natoms ≤ MAX_NATM` (=64) and a non-degenerate cell (`|det(cell)| ≥ 1e-6`), computes wrapped fractional coords (`pos @ inv(cell)`, mod 1), and writes `nanotube_templates.npz`. Prints a summary line (templates written, skipped, natoms range, size).

**The ASE-stub trick:** if `ase` is installed it unpickles normally; if not, it dynamically registers minimal stub modules (`ase`, `ase.atoms`, `ase.cell`, …) into `sys.modules` — tiny `_Stub` classes implementing only `__setstate__` — so the pickle loads **without a real ASE install**. `_numbers_positions_cell(atoms)` then extracts `(numbers, positions, cell)` from either a real `ase.Atoms` or a stub. This is why the runtime needs no ASE.

> `MAX_NATM = 64` is the **cache ceiling**, deliberately looser than the runtime `natm_range` filters applied downstream (e.g. carbon_24's ceiling of 24). The runtime filters tighter — see the mismatch note in [extending.md](../usage/extending.md).

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
- `class _NanotubeTemplateDB(path='./data/alx_1D/nanotube_templates.npz')` — `np.load(..., mmap_mode='r')` so arrays stay on disk; buckets structure indices by atom count (`_by_natm`) for O(1) range queries. If the file is missing it prints a warning and `sc='alx'` falls back to `SC_Nanotube`.
- `NANOTUBE_DB = _NanotubeTemplateDB()` — module-level singleton, loaded at import.
- `_NanotubeTemplateDB.sample(natm_min, natm_max)` — returns one random matching template `{frac_known, atom_numbers_known, cell}`, or `None`.
- `nanotube_template_from_db(natm_min, natm_max)` — wraps `.sample`, capping N at `natm_max - 1` to leave room for **≥1 decorating atom**; returns `{}` (→ fall back to parametric `ntb`) if nothing fits.

The returned kwargs are passed straight into `SC_DBTemplate(**extra_kwargs)` — see [structural-constraints.md](structural-constraints.md).

## Stub hooks (future work)
`nanotube_params_from_db()` (for `ntb`) and `carbontube_params_from_db()` (for `cnt`) currently return `{}` — placeholders for future DB-informed parametric sampling. `CARBONTUBE_DB = None`. The `ntb`/`cnt` constraints therefore use their built-in `*_DEFAULTS` today.

## Next

- The `SC_DBTemplate` class → [structural-constraints.md](structural-constraints.md)
- The carbon-24 caveat in full → [usage/extending.md](../usage/extending.md)

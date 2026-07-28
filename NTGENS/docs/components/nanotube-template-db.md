# Nanotube Template Database: `data/nano_1D/`

> **Audience:** programmer. **Purpose:** the real-structure template pipeline behind the `shl` constraint.
> ⬅ Back to [docs hub](../README.md) · Related: [structural-constraints.md](structural-constraints.md), [generation-scripts.md](generation-scripts.md)

The `shl` constraint grounds generation in **real** nanotube geometry instead of synthesizing it. Those structures come from two 1D-nanotube datasets, preprocessed once into a compact, quality-filtered cache. (`shl` uses the template's *cell/shape* to define a radial shell band; it does **not** pin the template's atoms — see [structural-constraints.md](structural-constraints.md).)

```
data/nano_1D/
├── build_templates.py             ETL: sources → reduce → quality-filter → npz
├── reduce_templates.py            axial fold/truncate to the simplest periodic cell
├── filter_templates.py            quality gates + metrics + poster figure (single source of truth)
├── load_sources.py                ase-optional readers for .pkl / .json sources
├── gdrive_fetch.py                pull raw sources from Google Drive
├── gdrive_manifest.json           Drive file ids for the sources
├── nanotube_templates.npz         compact CSR cache (~4.6 MB) — runtime input
├── filter_poster.png              filter funnel + 3 example tubes in 3D (poster figure)
└── nano_templates_forensics.ipynb forensics of the filtered cache
```

## The sources (merged, provenance-tagged)
Two datasets feed the cache, read by `load_sources.py` (no pymatgen/ase needed at build):
- **synthetic** — `stage0_survivors_structures.pkl` (large `list[ase.atoms.Atoms]`, raw pre-relaxation candidates; huge, up to ~15,750 atoms/cell).
- **real** — `alexandria_direct_1d.json` (pymatgen `ComputedStructureEntry` list, DFT-relaxed, ≤40 atoms, has energies).

They are pulled from Google Drive by `gdrive_fetch.py` (`gdrive_manifest.json` holds the file ids) or passed directly via `--src PATH:label`. Each template is tagged with a `source` code (0=synthetic, 1=real) so the runtime can pin real-only / synthetic-only / mixed.

## The ETL: `build_templates.py` (run once)
```bash
python data/nano_1D/build_templates.py \
  --src stage0_survivors_structures.pkl:synthetic alexandria_direct_1d.json:real \
  --reduce tiered --target-natm 128 --poster filter_poster.png
```
For each source structure it: **(1) reduces** oversized tubes to their simplest periodic cell (`reduce_templates.reduce_structure`: axial repeat-unit fold → truncate → drop, `≤ target_natm=128`); **(2) quality-filters** on the reduced cell (`filter_templates.passes_filter`, default on; `--no-filter` disables); **(3) dedups** across sources; then writes `nanotube_templates.npz` and prints a per-gate funnel. `--poster` renders `filter_poster.png`.

**The ASE-stub trick:** if `ase` is installed the synthetic pickle unpickles normally; if not, `build_templates._load_ase_pickle` registers minimal stub modules (`ase`, `ase.atoms`, `ase.cell`, …) into `sys.modules` — tiny `_Stub` classes implementing only `__setstate__` — so the pickle loads **without a real ASE install**. `_numbers_positions_cell(atoms)` then extracts `(numbers, positions, cell)` from either a real `ase.Atoms` or a stub. This is why the runtime needs no ASE.

## The quality filter: `filter_templates.py`
The single source of truth for the gates (thresholds mirror `comp_models/Analysis/nanotube_rtheta_forensics.ipynb`, which derived them). A template must clear all five to enter the cache:

| Gate | Symbol | Rejects |
|---|---|---|
| contacts | `d_min ≥ 0.7 Å` | unphysical atom overlaps (raw synthetic tubes) |
| hollow core | `r_min ≥ 1.0 Å` (5th-pct radius) | filled / collapsed cross-sections (non-tubes) |
| radius | `r_max ≤ 8 Å` (95th-pct radius) | giant flat/off-annular loops (14–28 Å) beyond the real-DFT envelope (real max 7.3 Å); their tube axis also mis-detects, so radius is the robust discriminator |
| atom count | `4 ≤ N ≤ 128` | too small for 2D geometry / above the cache cap |
| peaked ρ | `ρ_peak/ρ̄ ≥ 2.0` | flat radial density (no gradient for the density-guidance force → filled slabs) |

`filter_templates.py` also exposes `template_metrics`, the geometry helpers, `draw_tube_3d`, `poster_figure`, and `load_npz_records` (reused by the forensic notebook). **Current cache: 2,210 templates (1,268 synthetic + 942 real) from 45,158 inputs (~5.5% pass).** All 942 real templates clear the radius gate (real max 7.26 Å); it culls the large-radius synthetic tail.

## The cache: `nanotube_templates.npz`
CSR-style (compressed-sparse-row) packing so ragged per-structure arrays concatenate into flat arrays sliceable by structure:

| Array | Shape | Dtype | Meaning |
|---|---|---|---|
| `numbers` | `(ΣN,)` | int16 | atomic numbers, all structures concatenated |
| `frac_coords` | `(ΣN, 3)` | float32 | fractional coords, concatenated |
| `cells` | `(M, 3, 3)` | float32 | one cell per structure |
| `splits` | `(M+1,)` | int64 | row pointers: structure `i` = `[splits[i]:splits[i+1]]` |
| `natoms` | `(M,)` | int32 | atom count per structure |
| `source` | `(M,)` | int8 | provenance: 0=synthetic, 1=real |
| `max_natm` | scalar | int32 | the build ceiling (128) |

To read structure `i`: `numbers[splits[i]:splits[i+1]]`, `frac_coords[splits[i]:splits[i+1]]`, `cells[i]`.

## Runtime consumption: `_NanotubeTemplateDB`
In [`script/gen_utils.py`](../../script/gen_utils.py):
- `class _NanotubeTemplateDB(path='./data/nano_1D/nanotube_templates.npz')` — `np.load(..., mmap_mode='r')` so arrays stay on disk; buckets structure indices by atom count (`_by_natm`) for O(1) range queries. If the file is missing it prints a warning and `sc='shl'` falls back to the private synthetic ring (`_SC_NanotubeFallback`).
- `NANOTUBE_DB = _NanotubeTemplateDB()` — module-level singleton, loaded at import.
- `_NanotubeTemplateDB.sample(natm_min, natm_max, source_filter=None)` — returns one random matching template `{frac_known, atom_numbers_known, cell}`, or `None`. `source_filter='real'/'synthetic'` restricts by provenance (ignored on untagged caches).
- `nanotube_template_from_db(natm_min, natm_max, source_filter=None)` — wraps `.sample`, capping N at `natm_max - 1`; returns `{}` (→ fall back to `_SC_NanotubeFallback`) if nothing fits.

The returned kwargs are passed straight into `SC_DBShell(**extra_kwargs)`, which keeps the cell and **discards the atoms** — see [structural-constraints.md](structural-constraints.md).

## Next

- The `SC_DBShell` class → [structural-constraints.md](structural-constraints.md)
- Extending the pipeline → [usage/extending.md](../usage/extending.md)

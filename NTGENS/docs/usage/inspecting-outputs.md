# Inspecting Generated Outputs

> **Audience:** both. **Purpose:** how to turn raw generation tensors into inspectable structures, and three quick sanity checks to run on them.
> ⬅ Back to [docs hub](../README.md) · Related: [generation-scripts.md](../components/generation-scripts.md), [technical-foundations.md](../technical-foundations.md)

`script/generation.py` and the CLI route save raw tensors to `eval_gen_<label>.pt` (the [output contract](../components/generation-scripts.md)). The generation **notebooks** take a shortcut path: they keep the tensors in memory and convert straight to `pymatgen.Structure` objects for inspection before ever writing a CIF. This page documents that in-memory pattern — the worked example is **Section 6** of [`ntgen_generation.ipynb`](../../../comp_models/NTGEN_generation/ntgen_generation.ipynb).

> **Post-generation analysis moved:** `ntgen_generation.ipynb` used to run bulk-3D-crystal-style checks here (lattice parameter histograms, pymatgen space-group analysis, simulated XRD) in its own Section 7. Those were retired — as the caveats below always noted, they're a poor fit for a tube-in-vacuum cell (space group collapses to **P1**, XRD sub-15° peaks are vacuum-box artifacts). The real, nanotube-relevant analysis now lives in the sibling [`ntgen_validation.ipynb`](../../../comp_models/NTGEN_generation/ntgen_validation.ipynb): radial-band compliance / wall thickness (Tier 1 geometry), SMACT validity / occupancy ratio (Tier 2 chemistry), CHGNet energy (Tier 3), and relaxation drift (Tier 4), rolled into a composite ranking. Run it against the CIFs `ntgen_generation.ipynb` exports.

## 1. Raw tensors → `pymatgen.Structure`

After generation you have five parallel tensors (see [generation-scripts.md](../components/generation-scripts.md) and [data-flow.md](../architecture/data-flow.md)):

| Tensor | Shape | Meaning |
|---|---|---|
| `frac_coords` | `(ΣN, 3)` | fractional coordinates, all structures concatenated |
| `atom_types` | `(ΣN,)` | atomic numbers (Z), concatenated |
| `lattices` | `(n_struct, 3, 3)` | one cell matrix per structure |
| `num_atoms` | `(n_struct,)` | atom count per structure (for slicing the concatenated arrays) |
| `num_known` | `(n_struct,)` | pinned-skeleton atom count per structure |

Convert a `(3,3)` lattice matrix to `pymatgen`'s `(lengths, angles)` parameterization, then build a `Structure` per slice:

```python
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

def lattices_to_params(lat):
    """(3,3) lattice matrix -> (lengths[3], angles_deg[3])."""
    lengths = np.linalg.norm(lat, axis=1)
    angles = np.zeros(3)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        cos = np.dot(lat[j], lat[k]) / (lengths[j] * lengths[k])
        angles[i] = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return lengths, angles

start = 0
structures = []
for i in range(num_atoms.shape[0]):
    n_i = int(num_atoms[i])
    coords_i = frac_coords[start:start + n_i].numpy()
    types_i = atom_types[start:start + n_i].numpy()
    start += n_i
    lengths_i, angles_i = lattices_to_params(lattices[i].numpy())
    species = [chemical_symbols[int(t)] for t in types_i]
    structure = Structure(Lattice.from_parameters(*lengths_i, *angles_i),
                           species, coords_i, coords_are_cartesian=False)
    structures.append(structure)
```

> **Naming caveat:** this per-structure numpy `lattices_to_params` is a **different helper** from [`script/eval_utils.py::lattices_to_params_shape`](../../script/eval_utils.py) (the batched torch version used by the CLI `generation.py` path). Both do the same conversion, just at different points in the pipeline — don't confuse them if you're cross-referencing code. `chemical_symbols` comes from `sc_utils.py` ([structural-constraints.md](../components/structural-constraints.md)); index it by atomic number directly (`chemical_symbols[0] == 'X'`).

Wrap the conversion in a `try/except` — occasional degenerate cells (near-zero volume, extreme angles) can fail `Structure` construction; keep a `None` placeholder so downstream cells can filter (`[s for s in structures if s is not None]`) rather than crash the notebook.

## 2. Three inspection techniques (retired — kept for reference)

These three were demonstrated back-to-back in `ntgen_generation.ipynb`'s old Section 7 (mirroring the checks in the original 2D-lattice capstone notebook, `04_scigen_generation.ipynb`), but that section has been removed — see the analysis-moved note above. Kept here since the pymatgen recipes are still correct if you ever want to re-add one of these checks to a notebook.

### Lattice parameter distributions
Batch-convert all `lattices` to `(lengths, angles)` (a torch version of the same math, vectorized over the structure dimension) and histogram them. For small batches (`n < 20`, the common case on a single Colab run), fall back to a jittered strip plot so every point stays visible instead of collapsing into sparse histogram bars.

### Space-group analysis
`pymatgen.symmetry.analyzer.SpacegroupAnalyzer(structure, symprec=0.1)` gives the space-group symbol/number, crystal system, and point group. Aggregate across structures into a space-group-number histogram and a crystal-system pie chart.

### Simulated XRD
`pymatgen.analysis.diffraction.xrd.XRDCalculator(wavelength='CuKa').get_pattern(structure)` returns 2θ/intensity/hkl data; render as stem plots, annotating peaks above an intensity threshold with their `hkl` indices.

## 3. Nanotube-specific caveats

These three checks were designed for **bulk 3D crystals**. A nanotube generation cell is a tube embedded in a large vacuum box (`a, b ≈ 2×radius + vacuum`, only `c` is the physically periodic tube axis — see [technical-foundations.md](../technical-foundations.md) §5 and the `SC_DBShell` cell construction in [structural-constraints.md](../components/structural-constraints.md)). Read the results accordingly:

| Check | What's real | What's an artifact |
|---|---|---|
| Lattice parameters | `c` — the tube's true periodic repeat length | `a`, `b` — just the vacuum-box size, not a bulk lattice constant (expect `a ≈ b`) |
| Space group | relative comparison across candidates (e.g. "did this `shl` tube come out more symmetric than that one?") | the absolute space-group symbol — pymatgen's 3D analyzer has no notion of 1D rod/line-group symmetry, and the vacuum padding tends to collapse results toward **P1** |
| Simulated XRD | the pattern as a relative fingerprint between candidates | peaks below roughly 2θ≈10–15°, and the pattern as a literal prediction of an experimental bulk powder measurement — both are artifacts of the vacuum-padded cell |

Treat all three as **relative sanity checks across your own generated candidates**, not as literal descriptors of a bulk crystal.

## 4. Where this fits in the pipeline

The tensor→`Structure` conversion (Section 1) happens **before** CIF export, as an early quality gate — catch degenerate or obviously-wrong candidates before writing files. The actual analysis now happens afterward, against the exported CIFs, in `ntgen_validation.ipynb`. It's complementary to, not a replacement for, the CLI post-processing tools:

| Tool | When to use it |
|---|---|
| `ntgen_generation.ipynb` Section 6 | Quick, in-memory conversion + visual check right after generation |
| `script/save_cif.py` | Persist structures to disk as CIF files |
| `ntgen_validation.ipynb` | Tiered geometry/chemistry/CHGNet-energy/relaxation screening + composite ranking, against exported CIFs |
| `script/eval_screen.py` | Rigorous, automated filtering (SMACT validity, GNN classifiers) before DFT |
| `script/compute_metrics.py` | Benchmark-style aggregate metrics (validity/novelty/uniqueness/coverage) across a large batch |

## Next

- The current worked example → `comp_models/NTGEN_generation/ntgen_generation.ipynb`, Section 6
- The full post-generation analysis pipeline → `comp_models/NTGEN_generation/ntgen_validation.ipynb`
- The `.pt` output contract → [components/generation-scripts.md](../components/generation-scripts.md)
- Why the geometry looks this way → [technical-foundations.md](../technical-foundations.md)

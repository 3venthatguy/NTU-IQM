"""Convert the Alexandria 1D-nanotube ASE pickle into train/val/test CSVs for
retraining the diffusion model on nanotubes (see ../../RETRAIN_ALX.md).

Input : alexandria_1d_nanotubes.pkl  (list of ase.atoms.Atoms, ~48 MB)
Output: train.csv / val.csv / test.csv  (columns: material_id, pretty_formula,
        energy_per_atom, cif)  -- the schema scigen.pl_data.dataset.CrystDataset reads.

Unlike build_templates.py (which packs a compact npz of *pinned* skeletons for the
`alx` constraint), this emits full CIF strings so the CSPDiffusion model can be
*trained* on real nanotube geometries instead of the out-of-distribution mp_20
bulk-crystal checkpoint. That retrained checkpoint is what makes `alx` decoration
sensible (root fix for the mp_20-is-OOD fault).

Requires: numpy, pymatgen  (pymatgen only needed here, not at generation time).
ASE is optional -- the pickle is unpickled via the same lightweight stubs as
build_templates.py when ase is absent.

Run once (on a machine with pymatgen):
    python data/alx_1D/build_train_csv.py
"""
import sys, random
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE / 'alexandria_1d_nanotubes.pkl'
MAX_NATM = 64          # keep in sync with conf/data/alx_1d.yaml `max_atoms`
MIN_NATM = 1
VAL_FRAC, TEST_FRAC = 0.1, 0.1
SEED = 42

# Reuse the proven, ase-optional loader from build_templates.py.
sys.path.insert(0, str(HERE))
from build_templates import _load_ase_pickle, _numbers_positions_cell  # noqa: E402

try:
    from pymatgen.core import Structure, Lattice
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit(
        'pymatgen is required to build CIF strings for training CSVs. '
        'Install it (pip install pymatgen) and re-run. '
        f'(import error: {e})')


def _energy_per_atom(atoms, n):
    """Best-effort energy/atom from the ASE object; 0.0 if unavailable.

    The Alexandria pickle often carries no attached calculator once loaded via
    stubs, so this is typically a placeholder. The generative CSPDiffusion model
    trains on geometry, not this property, but the datamodule still fits a scaler
    over it -- a constant column is fine.
    """
    for getter in ('get_potential_energy',):
        fn = getattr(atoms, getter, None)
        if callable(fn):
            try:
                return float(fn()) / max(n, 1)
            except Exception:
                pass
    return 0.0


def build():
    db = _load_ase_pickle(SRC)
    rows = []
    skipped = 0
    for i, atoms in enumerate(db):
        nums, pos, cell = _numbers_positions_cell(atoms)
        n = len(nums)
        if n < MIN_NATM or n > MAX_NATM:
            skipped += 1
            continue
        if not np.isfinite(np.linalg.det(cell)) or abs(np.linalg.det(cell)) < 1e-6:
            skipped += 1
            continue
        try:
            struct = Structure(Lattice(np.asarray(cell, float)),
                               [int(z) for z in nums],
                               np.asarray(pos, float),
                               coords_are_cartesian=True)
            cif = struct.to(fmt='cif')
        except Exception:
            skipped += 1
            continue
        rows.append({
            'material_id': f'alx1d-{i:06d}',
            'pretty_formula': struct.composition.reduced_formula,
            'energy_per_atom': _energy_per_atom(atoms, n),
            'cif': cif,
        })

    if not rows:
        raise SystemExit('No usable structures found -- check the input pickle.')

    random.Random(SEED).shuffle(rows)
    n_total = len(rows)
    n_test = int(n_total * TEST_FRAC)
    n_val = int(n_total * VAL_FRAC)
    test, val, train = rows[:n_test], rows[n_test:n_test + n_val], rows[n_test + n_val:]

    for name, split in (('train', train), ('val', val), ('test', test)):
        out = HERE / f'{name}.csv'
        pd.DataFrame(split, columns=['material_id', 'pretty_formula',
                                     'energy_per_atom', 'cif']).to_csv(out, index=False)
        print(f'wrote {out.name}: {len(split)} structures')
    print(f'total kept {n_total} (skipped {skipped})')


if __name__ == '__main__':
    build()

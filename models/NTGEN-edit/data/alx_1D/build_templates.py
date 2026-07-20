"""Convert the Alexandria 1D-nanotube ASE pickle into a compact template cache.

Input : alexandria_1d_nanotubes.pkl  (list of ase.atoms.Atoms, ~48 MB, needs ase)
Output: nanotube_templates.npz       (CSR-style numpy arrays, ~MB, no ase needed)

The cache stores, for every structure with atom count <= MAX_NATM, the pinned
"known" skeleton the diffusion model needs (Pathway 3): atomic numbers, fractional
coordinates, and the 3x3 cell. Runtime (gen_utils.nanotube_params_from_db) then
mmaps this file and filters by the caller's natm_range with zero per-object cost.

Run once:  python data/alx_1D/build_templates.py
It unpickles via ASE if installed, else via lightweight stubs (ase not required).
"""
import sys, types, pickle
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / 'alexandria_1d_nanotubes.pkl'
DST = HERE / 'nanotube_templates.npz'
MAX_NATM = 64          # cache ceiling; runtime filters tighter by natm_range


def _load_ase_pickle(path):
    """Unpickle the ASE list. Uses real ase if present, else minimal stubs that
    capture only numbers/positions/cell (all we need)."""
    try:
        import ase  # noqa: F401
        return pickle.load(open(path, 'rb'))
    except ModuleNotFoundError:
        pass
    ase = types.ModuleType('ase'); ase.__path__ = []; sys.modules['ase'] = ase
    for sub in ('ase.atoms', 'ase.cell', 'ase.constraints',
                'ase.calculators', 'ase.calculators.singlepoint'):
        m = types.ModuleType(sub); m.__path__ = []; sys.modules[sub] = m

    class _Stub:
        def __init__(self, *a, **k): pass
        def __setstate__(self, s): self._state = s

    for mod, cls in (('ase.atoms', 'Atoms'), ('ase.cell', 'Cell'),
                     ('ase.constraints', 'FixAtoms'),
                     ('ase.calculators.singlepoint', 'SinglePointCalculator')):
        setattr(sys.modules[mod], cls, type(cls, (_Stub,), {}))
    return pickle.load(open(path, 'rb'))


def _numbers_positions_cell(atoms):
    """Return (Z[int], pos[N,3], cell[3,3]) whether atoms is a real ase.Atoms
    or a stub carrying _state."""
    st = getattr(atoms, '_state', None)
    if st is not None:                                   # stub
        nums = np.asarray(st['arrays']['numbers'])
        pos = np.asarray(st['arrays']['positions'], float)
        cobj = st['_cellobj']
        cs = getattr(cobj, '_state', {})
        cell = np.asarray(cs.get('array', cs.get('_array')), float)
    else:                                                # real ase.Atoms
        nums = atoms.get_atomic_numbers()
        pos = atoms.get_positions()
        cell = np.asarray(atoms.get_cell())
    return nums, pos, cell


def build():
    db = _load_ase_pickle(SRC)
    numbers, fracs, cells, natoms = [], [], [], []
    skipped = 0
    for atoms in db:
        nums, pos, cell = _numbers_positions_cell(atoms)
        n = len(nums)
        if n == 0 or n > MAX_NATM:
            skipped += 1
            continue
        det = np.linalg.det(cell)
        if not np.isfinite(det) or abs(det) < 1e-6:      # degenerate cell
            skipped += 1
            continue
        frac = pos @ np.linalg.inv(cell)
        frac -= np.floor(frac)                            # wrap into [0, 1)
        numbers.append(nums.astype(np.int16))
        fracs.append(frac.astype(np.float32))
        cells.append(cell.astype(np.float32))
        natoms.append(n)

    natoms = np.asarray(natoms, np.int32)
    splits = np.concatenate([[0], np.cumsum(natoms)]).astype(np.int64)
    np.savez_compressed(
        DST,
        numbers=np.concatenate(numbers).astype(np.int16),     # (sum N,)
        frac_coords=np.concatenate(fracs).astype(np.float32),  # (sum N, 3)
        cells=np.stack(cells).astype(np.float32),              # (M, 3, 3)
        splits=splits,                                         # (M+1,)
        natoms=natoms,                                         # (M,)
        max_natm=np.int32(MAX_NATM),
    )
    kb = DST.stat().st_size / 1024
    print(f'wrote {DST.name}: {len(natoms)} templates '
          f'(skipped {skipped}), natoms {natoms.min()}-{natoms.max()}, {kb:.0f} KB')


if __name__ == '__main__':
    build()

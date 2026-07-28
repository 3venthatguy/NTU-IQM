"""One reader that yields pinning-template records from either input format.

Two source formats feed the `alx` pinning database (see build_templates.py):
  - `.pkl`  : list of ase.atoms.Atoms  (synthetic stage0 set, and the legacy
              alexandria_1d_nanotubes.pkl). Unpickled via the ase-optional stubs.
  - `.json` : pymatgen ComputedStructureEntry list under ["entries"]
              (real alexandria_direct_1d set). Parsed as plain dicts -- NO pymatgen
              dependency on this fast path.

Each record is normalized to the same tuple the builder packs:
    Record(numbers int16 (N,), frac_coords float32 (N,3) in [0,1),
           cell float32 (3,3), energy_per_atom float, source str)

Both formats are tagged with a `source` label ('synthetic' | 'real') so the builder
can persist provenance and the runtime can pin real-only / synthetic-only / mixed.
"""
from collections import namedtuple
from pathlib import Path
import json

import numpy as np

# Reuse the proven, ase-optional pickle loader + accessor from build_templates.
from build_templates import _load_ase_pickle, _numbers_positions_cell

HERE = Path(__file__).resolve().parent

Record = namedtuple('Record', 'numbers frac_coords cell energy_per_atom source')

# Z <- element symbol, without importing pymatgen/ase. Index into this list is the
# atomic number (X=0, H=1, ...), matching script/sc_utils.py::chemical_symbols.
_SYMBOLS = [
    'X', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al',
    'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
    'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y',
    'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',
    'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb',
    'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
    'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
    'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf',
    'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts',
    'Og']
_Z = {s: i for i, s in enumerate(_SYMBOLS)}


def _iter_pkl(path, source):
    """Yield Records from an ase.Atoms pickle. Energy is best-effort (usually
    unavailable once loaded via stubs) -> NaN."""
    db = _load_ase_pickle(path)
    for atoms in db:
        nums, pos, cell = _numbers_positions_cell(atoms)
        cell = np.asarray(cell, float)
        det = np.linalg.det(cell)
        if not np.isfinite(det) or abs(det) < 1e-6:      # degenerate cell -> skip
            continue
        frac = np.asarray(pos, float) @ np.linalg.inv(cell)
        frac -= np.floor(frac)                            # wrap into [0, 1)
        epa = np.nan
        fn = getattr(atoms, 'get_potential_energy', None)
        if callable(fn) and len(nums):
            try:
                epa = float(fn()) / len(nums)
            except Exception:
                epa = np.nan
        yield Record(np.asarray(nums, np.int16), frac.astype(np.float32),
                     cell.astype(np.float32), epa, source)


def _iter_json(path, source):
    """Yield Records from a pymatgen ComputedStructureEntry JSON (plain-dict parse)."""
    with open(path) as fh:
        entries = json.load(fh)['entries']
    for e in entries:
        s = e['structure']
        cell = np.asarray(s['lattice']['matrix'], float)
        det = np.linalg.det(cell)
        if not np.isfinite(det) or abs(det) < 1e-6:
            continue
        sites = s['sites']
        try:
            nums = np.asarray([_Z[st['species'][0]['element']] for st in sites],
                              np.int16)
        except KeyError:                                  # unknown/odd species -> skip
            continue
        frac = np.asarray([st['abc'] for st in sites], float)
        frac -= np.floor(frac)
        n = len(sites)
        epa = float(e['energy']) / n if n and e.get('energy') is not None else np.nan
        yield Record(nums, frac.astype(np.float32), cell.astype(np.float32), epa,
                     source)


def iter_source(path, source):
    """Dispatch on file extension. `source` is the provenance label to stamp."""
    path = Path(path)
    if path.suffix == '.json':
        yield from _iter_json(path, source)
    else:                                                 # .pkl (and unknown -> pkl)
        yield from _iter_pkl(path, source)


def iter_sources(specs):
    """Chain Records from many (path, source) specs, printing a per-source count."""
    for path, source in specs:
        n = 0
        for rec in iter_source(path, source):
            n += 1
            yield rec
        print(f'  read {n} structures from {Path(path).name} [{source}]')

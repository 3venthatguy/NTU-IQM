"""Merge the 1D-nanotube source datasets into a compact, provenance-tagged cache.

Inputs (see load_sources.py for the readers; both formats are supported):
  - synthetic : stage0_survivors_structures.pkl  (large ase.atoms.Atoms pickle)
  - real      : alexandria_direct_1d.json         (pymatgen ComputedStructureEntry)
Output: nanotube_templates.npz  (CSR-style numpy arrays, ~MB, no ase needed at run)

The cache stores, for every structure with atom count <= MAX_NATM, the pinned
"known" skeleton the diffusion model needs (Pathway 3): atomic numbers, fractional
coordinates, the 3x3 cell, and a per-structure `source` code (0=synthetic, 1=real).
Runtime (gen_utils._NanotubeTemplateDB) mmaps this file and filters by the caller's
natm_range -- and optionally by source -- with zero per-object cost.

The sources live in Google Drive; gdrive_fetch.py pulls them into this folder.
Structures are deduped across sources, so the Drive set can grow without changing
the runtime path.

Run once:
    python data/nano_1D/build_templates.py            # build from the default sources
    python data/nano_1D/build_templates.py --fetch    # pull from Drive first, then build
    python data/nano_1D/build_templates.py --src a.pkl:synthetic b.json:real
    python data/nano_1D/build_templates.py --max-natm 128   # pin the cap explicitly
It unpickles via ASE if installed, else via lightweight stubs (ase not required).
"""
import argparse
import sys, types, pickle
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DST = HERE / 'nanotube_templates.npz'

# Provenance codes persisted in the npz `source` array.
SOURCE_CODES = {'synthetic': 0, 'real': 1}

# Default source datasets, tagged by role. gdrive_fetch.py places these filenames
# in this folder; pass --src to override (e.g. build straight from ~/Downloads).
DEFAULT_SPECS = [
    (HERE / 'stage0_survivors_structures.pkl', 'synthetic'),
    (HERE / 'alexandria_direct_1d.json', 'real'),
]

# Optional pre-filter: drop structures with more than this many atoms before cap
# selection. None -> keep every structure (the synthetic set runs to ~15,750 atoms;
# retaining the full large database is the chosen behavior). Set via --collect-max
# to bound memory / npz size for smaller runs.
COLLECT_MAX = None


def _infer_label(path):
    """Source label for a --src entry with no explicit :label (json -> real)."""
    return 'real' if Path(path).suffix == '.json' else 'synthetic'


def _parse_specs(src_args):
    """Turn CLI --src tokens (`path` or `path:label`) into (Path, label) pairs."""
    specs = []
    for tok in src_args:
        if ':' in tok and tok.rsplit(':', 1)[1] in SOURCE_CODES:
            path, label = tok.rsplit(':', 1)
        else:
            path, label = tok, _infer_label(tok)
        specs.append((Path(path), label))
    return specs


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


def _struct_key(nums, frac):
    """Cheap dedup key so overlapping pkls don't double-count a structure:
    sorted atomic numbers + coarsely rounded, sorted fractional coords."""
    order = np.lexsort((nums,))
    fr = np.round(frac[order] % 1.0, 3)
    return (tuple(np.sort(nums).tolist()), fr.tobytes())


def _report_hist(natoms):
    """Print the collected natoms distribution + survivor counts at candidate caps,
    so the chosen MAX_NATM is data-informed."""
    print(f'\ncollected {len(natoms)} structures')
    if not len(natoms):
        return
    pct = {p: int(np.percentile(natoms, p)) for p in (50, 75, 90, 95, 99)}
    print('  natoms  min %d  median %d  mean %.1f  max %d'
          % (natoms.min(), int(np.median(natoms)), natoms.mean(), natoms.max()))
    print('  percentiles ' + '  '.join(f'p{p}={v}' for p, v in pct.items()))
    top = int(natoms.max())
    for cap in (64, 128, 256, 512, 1024, top):
        keep = int((natoms <= cap).sum())
        print(f'    <= {cap:6d}: {keep:6d}  ({100 * keep / len(natoms):.1f}%)')


def build(specs=None, max_natm=None, collect_max=COLLECT_MAX,
          reduce='tiered', target_natm=128, fold_tol=0.25, cluster_dedup=False,
          filter=True, poster_path=None, poster_seed=None):
    """Merge sources into the pinning cache.

    collect_max   : drop structures above this size in pass 1 (None -> keep all).
    max_natm      : final cap written to the npz (None -> max observed atom count).
    reduce        : 'tiered' shrinks oversized tubes to <= target_natm (fold ->
                    truncate -> drop); 'none' disables reduction.
    target_natm   : per-template atom ceiling for the reduction.
    fold_tol      : Angstrom tolerance for the axial repeat-unit fold.
    cluster_dedup : collapse near-duplicate tubes (same composition + radial profile),
                    keeping the smallest representative.
    filter        : apply the quality gates (filter_templates.passes_filter) so only
                    genuine, guidance-friendly nanotubes reach the cache.
    poster_path   : if set, render the research-poster figure (funnel table + 3 example
                    tubes in 3D) to this PNG after building."""
    specs = specs or DEFAULT_SPECS
    missing = [p for p, _ in specs if not Path(p).exists()]
    if missing:
        raise SystemExit(
            'Missing source(s): ' + ', '.join(Path(m).name for m in missing) +
            '\nFetch them from Drive first:  python data/nano_1D/gdrive_fetch.py')

    from load_sources import iter_sources           # lazy: avoids circular import
    if reduce != 'none':
        from reduce_templates import reduce_structure, fingerprint
    if filter or poster_path:
        from filter_templates import template_metrics, passes_filter, funnel

    # Pass 1: read every source, reduce oversized tubes, quality-filter, dedup, collect.
    numbers, fracs, cells, natoms, source = [], [], [], [], []
    all_metrics = []                                  # metrics for every reduced tube
    seen = set()
    dupes = too_big = filtered = 0
    reduce_drop = 0
    tag_counts = {'keep': 0, 'fold': 0, 'truncate': 0}
    kept_by_src, read_by_src = {}, {}
    for rec in iter_sources(specs):
        read_by_src[rec.source] = read_by_src.get(rec.source, 0) + 1
        nums, frac, cell = rec.numbers, rec.frac_coords, rec.cell
        if reduce != 'none':                          # tiered shrink to <= target
            r = reduce_structure(nums, frac, cell, target=target_natm, tol=fold_tol)
            if r is None:                             # can't reach target -> drop
                reduce_drop += 1
                continue
            nums, frac, cell, tag = r
            tag_counts[tag] += 1
        n = len(nums)
        if n == 0 or (collect_max is not None and n > collect_max):
            too_big += int(collect_max is not None and n > collect_max)
            continue
        if filter or poster_path:                     # gate metrics on the reduced cell
            m = template_metrics(nums, frac, cell)
            all_metrics.append(m)
            if filter and not passes_filter(m):       # reject non-nanotube geometry
                filtered += 1
                continue
        key = _struct_key(nums, frac)
        if key in seen:                               # same structure in >1 source
            dupes += 1
            continue
        seen.add(key)
        numbers.append(np.asarray(nums, np.int16))
        fracs.append(np.asarray(frac, np.float32))
        cells.append(np.asarray(cell, np.float32))
        natoms.append(n)
        source.append(SOURCE_CODES.get(rec.source, 0))
        kept_by_src[rec.source] = kept_by_src.get(rec.source, 0) + 1

    if not natoms:
        raise SystemExit('No usable structures found -- check the inputs.')

    if reduce != 'none':
        print(f'  reduction tags  : {tag_counts}  (dropped {reduce_drop})')

    if filter and all_metrics:                        # cumulative quality-gate funnel
        total, rows, survivors = funnel(all_metrics)
        print(f'  quality filter  : {survivors}/{total} pass all gates '
              f'({100 * survivors / total:.1f}%)')
        for sym, desc, keep in rows:
            print(f'      {sym:16s} {desc:26s} {keep:6d}  ({100 * keep / total:.1f}%)')

    if cluster_dedup:                                 # collapse near-duplicate tubes
        keep_idx, fp_best = [], {}
        for i in range(len(natoms)):
            fp = fingerprint(numbers[i], fracs[i], cells[i])
            j = fp_best.get(fp)
            if j is None:
                fp_best[fp] = len(keep_idx)
                keep_idx.append(i)
            elif natoms[i] < natoms[keep_idx[j]]:     # keep the smallest representative
                keep_idx[j] = i
        clustered = len(natoms) - len(keep_idx)
        numbers = [numbers[i] for i in keep_idx]
        fracs = [fracs[i] for i in keep_idx]
        cells = [cells[i] for i in keep_idx]
        natoms = [natoms[i] for i in keep_idx]
        source = [source[i] for i in keep_idx]
        print(f'  cluster-dedup   : removed {clustered} near-duplicates')

    natoms = np.asarray(natoms, np.int32)
    source = np.asarray(source, np.int8)
    _report_hist(natoms)
    if max_natm is None:                  # keep everything collected
        max_natm = int(natoms.max())
    print(f'\nusing MAX_NATM = {max_natm}  (size-skipped: {too_big}, '
          f'quality-filtered: {filtered}, cross-source dupes: {dupes})')

    # Pass 2: apply the chosen cap and pack CSR arrays (structure order preserved).
    keep = np.nonzero(natoms <= max_natm)[0]
    numbers_k = [numbers[i] for i in keep]
    fracs_k = [fracs[i] for i in keep]
    cells_k = np.stack([cells[i] for i in keep]).astype(np.float32)
    natoms_k = natoms[keep].astype(np.int32)
    source_k = source[keep].astype(np.int8)
    splits = np.concatenate([[0], np.cumsum(natoms_k)]).astype(np.int64)

    np.savez_compressed(
        DST,
        numbers=np.concatenate(numbers_k).astype(np.int16),    # (sum N,)
        frac_coords=np.concatenate(fracs_k).astype(np.float32),  # (sum N, 3)
        cells=cells_k,                                         # (M, 3, 3)
        splits=splits,                                         # (M+1,)
        natoms=natoms_k,                                       # (M,)
        source=source_k,                                       # (M,)  0=syn 1=real
        max_natm=np.int32(max_natm),
    )

    inv = {v: k for k, v in SOURCE_CODES.items()}
    by_src_final = {inv.get(int(s), s): int((source_k == s).sum())
                    for s in np.unique(source_k)}
    kb = DST.stat().st_size / 1024
    print(f'\nwrote {DST.name}: {len(natoms_k)} templates '
          f'(natoms {int(natoms_k.min())}-{int(natoms_k.max())}), {kb:.0f} KB')
    print(f'  read per source : {read_by_src}')
    print(f'  collected       : {kept_by_src}')
    print(f'  in cache (<=cap) : {by_src_final}')

    if poster_path:                                   # research-poster figure
        from filter_templates import poster_figure
        all_num = np.concatenate(numbers_k); all_frac = np.concatenate(fracs_k)
        survivor_records = []                          # all survivors, source-tagged
        for i in range(len(natoms_k)):                 # poster_figure picks 2 real + 2 syn
            lo, hi = int(splits[i]), int(splits[i + 1])
            fr = np.asarray(all_frac[lo:hi], float)
            survivor_records.append(dict(numbers=all_num[lo:hi], frac=fr,
                                         cell=cells_k[i], cart=fr @ cells_k[i],
                                         source=int(source_k[i])))
        poster_figure(all_metrics, survivor_records, poster_path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fetch', action='store_true',
                    help='pull sources from Google Drive (gdrive_fetch.py) first')
    ap.add_argument('--src', nargs='+', metavar='PATH[:label]',
                    help='explicit sources; label in {synthetic,real} (json->real)')
    ap.add_argument('--max-natm', type=int, default=None,
                    help='final atom-count cap; default = keep all collected')
    ap.add_argument('--collect-max', type=int, default=COLLECT_MAX,
                    help='drop structures above this size in pass 1 (default: keep all)')
    ap.add_argument('--reduce', choices=['none', 'tiered'], default='tiered',
                    help='shrink oversized tubes to <= --target-natm (default: tiered)')
    ap.add_argument('--target-natm', type=int, default=128,
                    help='per-template atom ceiling for reduction (default: 128)')
    ap.add_argument('--fold-tol', type=float, default=0.25,
                    help='Angstrom tolerance for the axial repeat-unit fold')
    ap.add_argument('--cluster-dedup', action='store_true',
                    help='collapse near-duplicate tubes, keep smallest representative')
    ap.add_argument('--no-filter', dest='filter', action='store_false',
                    help='skip the quality gates (keep every reduced template)')
    ap.add_argument('--poster', nargs='?', const=str(HERE / 'filter_poster.png'),
                    default=None, metavar='PATH',
                    help='render the filter poster PNG (default: nano_1D/filter_poster.png)')
    ap.set_defaults(filter=True)
    args = ap.parse_args()

    if args.fetch:
        from gdrive_fetch import fetch_all
        fetch_all()

    specs = _parse_specs(args.src) if args.src else None
    build(specs, max_natm=args.max_natm, collect_max=args.collect_max,
          reduce=args.reduce, target_natm=args.target_natm, fold_tol=args.fold_tol,
          filter=args.filter, poster_path=args.poster,
          cluster_dedup=args.cluster_dedup)

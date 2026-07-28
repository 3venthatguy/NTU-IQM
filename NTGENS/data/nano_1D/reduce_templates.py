"""Shrink oversized 1D-nanotube templates to small periodic cells (numpy/scipy).

The synthetic stage0 tubes run to ~455 A / 15,750 atoms along the tube axis, far too
large to pin as a single periodic unit cell during denoising. This module reduces a
structure to a compact cell via a tiered strategy (see reduce_structure):

  1. axial repeat-unit fold  -- if the tube is (approximately) periodic along its axis,
     keep one period. Correct & lossless for clean tubes.
  2. axial truncation        -- else cut a short axial window at a low atomic-density
     plane (fewest bonds crossing the periodic seam).
  3. drop                    -- if even the minimal full-cross-section axial unit still
     exceeds the target (very large-circumference tubes), the structure cannot be a
     faithful small periodic cell -> return None.

Pure numpy + scipy.spatial.cKDTree so the build runs anywhere (no pymatgen/spglib/ase).
All coordinates are fractional (N,3); `cell` is the (3,3) lattice matrix (row vectors),
cartesian = frac @ cell. The tube axis is taken to be the longest lattice vector.
"""
from functools import reduce as _reduce
from math import gcd

import numpy as np
from scipy.spatial import cKDTree


def _axis(cell):
    """Index of the longest lattice vector (the 1D tube axis)."""
    return int(np.argmax(np.linalg.norm(np.asarray(cell, float), axis=1)))


def _divisors_desc(n, min_unit=2):
    """Divisors k>=2 of n with n/k >= min_unit, largest first (smallest unit first)."""
    ks = [k for k in range(2, n // min_unit + 1) if n % k == 0]
    return sorted(ks, reverse=True)


def axial_repeat_unit(nums, frac, cell, axis, tol=0.25, min_period=2.0,
                      min_unit_atoms=3):
    """Return one axial period (nums, frac, cell) if the structure is periodic along
    `axis` within `tol` (Angstrom), else None. Picks the largest valid fold k (=> the
    smallest primitive unit).

    Guards against spurious folds in dense/noisy generated structures: a candidate
    period c/k must be >= `min_period` (no nanotube repeats faster than a bond length),
    and the resulting unit must keep >= `min_unit_atoms` (else it is a point-cloud
    collapse, not a real repeat)."""
    nums = np.asarray(nums)
    N = len(nums)
    cart = np.asarray(frac, float) @ np.asarray(cell, float)
    axis_vec = np.asarray(cell, float)[axis]
    c_len = np.linalg.norm(axis_vec)
    # Augment with a +axis image so atoms shifted past the top match wrapped partners.
    tree = cKDTree(np.vstack([cart, cart + axis_vec]))
    aug_sp = np.concatenate([nums, nums])

    best_k = 1
    for k in _divisors_desc(N):
        if c_len / k < min_period or N // k < min_unit_atoms:   # physical guards
            continue
        shifted = cart + axis_vec / k
        dist, idx = tree.query(shifted, k=1)
        if np.all(dist < tol) and np.all(aug_sp[idx] == nums):
            best_k = k
            break                                   # largest k first -> primitive
    if best_k == 1:
        return None

    fa = np.asarray(frac, float)[:, axis] % 1.0
    sel = np.floor(fa * best_k).astype(int) == 0    # keep the first 1/k slab
    frac2 = np.asarray(frac, float)[sel].copy()
    frac2[:, axis] = (fa[sel] * best_k) % 1.0        # rescale slab to [0, 1)
    cell2 = np.asarray(cell, float).copy()
    cell2[axis] = axis_vec / best_k
    return nums[sel], frac2, cell2


def truncate_axial(nums, frac, cell, axis, target, rcut=2.0, steps=400):
    """Cut a short axial window (<= target atoms) that (a) matches the parent
    stoichiometry, (b) cuts at low-density planes. Returns (nums, frac, cell) of the
    segment, or None if no window fits target while preserving the full cross-section."""
    nums = np.asarray(nums)
    cell = np.asarray(cell, float)
    N = len(nums)
    c_len = np.linalg.norm(cell[axis])
    z = (np.asarray(frac, float)[:, axis] % 1.0) * c_len
    d = N / c_len                                    # linear density (atoms / A)
    L = target / d                                   # window length ~ target atoms
    if L <= 0 or L >= c_len:
        return None

    order = np.argsort(z)
    zs = z[order]
    sp = nums[order]
    zext = np.concatenate([zs, zs + c_len])          # +axis images for wrap-around
    # Global composition + cumulative per-species counts over zext for O(1) windows.
    species = np.unique(nums)
    p_global = np.array([(nums == s_).sum() for s_ in species], float) / N
    spx = np.concatenate([sp, sp])
    cum = np.zeros((len(species), len(zext) + 1))
    for si, s_ in enumerate(species):
        cum[si, 1:] = np.cumsum(spx == s_)

    best = None
    for s in np.linspace(0, c_len, steps, endpoint=False):
        e = s + L
        lo = int(np.searchsorted(zext, s))
        hi = int(np.searchsorted(zext, e))
        cnt = hi - lo
        if cnt == 0 or cnt > target:
            continue
        comp = cum[:, hi] - cum[:, lo]
        dev = float(np.abs(comp / cnt - p_global).sum())      # stoichiometry deviation
        bs = int(np.searchsorted(zext, s + rcut / 2) - np.searchsorted(zext, s - rcut / 2))
        be = int(np.searchsorted(zext, e + rcut / 2) - np.searchsorted(zext, e - rcut / 2))
        # Prefer representative stoichiometry, then a clean cut, then a fuller window.
        key = (round(dev, 2), bs + be, -cnt)
        if best is None or key < best[0]:
            best = (key, s)
    if best is None:
        return None

    s = best[1]
    znorm = (z - s) % c_len
    sel = znorm < L
    frac2 = np.asarray(frac, float)[sel].copy()
    frac2[:, axis] = znorm[sel] / L                  # new axial frac in [0, 1)
    cell2 = cell.copy()
    cell2[axis] = cell[axis] * (L / c_len)           # shrink c to the window; keep a,b
    return nums[sel], frac2, cell2


def reduce_structure(nums, frac, cell, target=128, tol=0.25, max_fold_n=2000):
    """Tiered reduction. Returns (nums, frac, cell, tag) with tag in
    {'keep','fold','truncate'}, or None to drop the structure."""
    nums = np.asarray(nums)
    frac = np.asarray(frac, float) % 1.0
    cell = np.asarray(cell, float)
    axis = _axis(cell)
    if len(nums) <= target:                          # already small (e.g. real data)
        return nums, frac, cell, 'keep'

    if len(nums) <= max_fold_n:                      # Tier 1: repeat-unit fold
        folded = axial_repeat_unit(nums, frac, cell, axis, tol)
        if folded is not None:
            nums, frac, cell = folded
            if len(nums) <= target:
                return nums, frac, cell, 'fold'

    trunc = truncate_axial(nums, frac, cell, axis, target)   # Tier 2: truncate
    if trunc is not None:
        return (*trunc, 'truncate')
    return None                                       # Tier 3: drop


def fingerprint(nums, frac, cell):
    """Near-duplicate key: reduced composition + atom count + rounded radial
    cross-section profile (invariant to axial position / in-plane rotation)."""
    nums = np.asarray(nums)
    u, ct = np.unique(nums, return_counts=True)
    g = _reduce(gcd, ct.tolist())
    comp = tuple(sorted(zip(u.tolist(), (ct // g).tolist())))
    axis = _axis(cell)
    plane = [i for i in range(3) if i != axis]
    xy = (np.asarray(frac, float) @ np.asarray(cell, float))[:, plane]
    r = np.sort(np.round(np.linalg.norm(xy - xy.mean(0), axis=1), 1))
    return (comp, len(nums), tuple(r.tolist()))

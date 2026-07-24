"""Visualize a few nanotube structures from nanotube_templates.npz.

Loads the CSR-style template cache (see build_templates.py) and plots
selected structures as 3D scatter plots, colored by element, with periodic
copies along the tube axis (z) drawn faintly so the tube shape reads clearly.

Usage:
    python visualize_templates.py                # first 4 templates
    python visualize_templates.py --indices 0 5 9 --n-cells 3
    python visualize_templates.py --natm-range 20 40 --count 6
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
NPZ = HERE / 'nanotube_templates.npz'

# Minimal Z -> (symbol, color) map for common elements in this dataset.
ELEMENT_COLORS = {
    1: ('H', '#FFFFFF'), 5: ('B', '#FFB5B5'), 6: ('C', '#909090'),
    7: ('N', '#3050F8'), 8: ('O', '#FF0D0D'), 9: ('F', '#90E050'),
    13: ('Al', '#BFA6A6'), 14: ('Si', '#F0C8A0'), 15: ('P', '#FF8000'),
    16: ('S', '#FFFF30'), 17: ('Cl', '#1FF01F'), 30: ('Zn', '#7D80B0'),
    31: ('Ga', '#C28F8F'), 33: ('As', '#BD80E3'), 34: ('Se', '#FFA100'),
    49: ('In', '#A67573'), 51: ('Sb', '#9E63B5'), 52: ('Te', '#D47A00'),
    16 + 1: ('Cl', '#1FF01F'),
}


def element_info(z):
    sym, color = ELEMENT_COLORS.get(int(z), (f'Z{z}', '#4C9BE0'))
    return sym, color


def load_structure(data, idx):
    """Return (numbers, frac_coords, cell) for template `idx`."""
    lo, hi = data['splits'][idx], data['splits'][idx + 1]
    numbers = data['numbers'][lo:hi]
    frac = data['frac_coords'][lo:hi]
    cell = data['cells'][idx]
    return numbers, frac, cell


def plot_structure(ax, numbers, frac, cell, n_cells=2):
    """Scatter atoms in Cartesian coords, repeating along the tube axis (z)."""
    seen = set()
    for rep in range(n_cells):
        shift = frac + np.array([0.0, 0.0, rep])
        cart = shift @ cell
        alpha = 1.0 if rep == 0 else 0.35
        for z, (x, y, zc) in zip(numbers, cart):
            sym, color = element_info(z)
            label = sym if (rep == 0 and sym not in seen) else None
            if label:
                seen.add(sym)
            ax.scatter(x, y, zc, color=color, s=60, edgecolors='k',
                       linewidths=0.3, alpha=alpha, label=label)

    # draw the unit cell outline (rep 0 only) for orientation
    a, b, c = cell
    corners = np.array([
        [0, 0, 0], a, a + b, b, [0, 0, 0],
        c, a + c, a + b + c, b + c, c,
    ])
    ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], color='gray',
            linewidth=0.6, alpha=0.5)
    for p, q in [(a, a + c), (a + b, a + b + c), (b, b + c)]:
        ax.plot(*zip(p, q), color='gray', linewidth=0.6, alpha=0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--indices', type=int, nargs='+', default=None,
                         help='explicit template indices to plot')
    parser.add_argument('--count', type=int, default=4,
                         help='number of templates to plot if --indices not given')
    parser.add_argument('--natm-range', type=int, nargs=2, default=None,
                         metavar=('MIN', 'MAX'),
                         help='restrict candidates to this atom-count range')
    parser.add_argument('--n-cells', type=int, default=2,
                         help='periodic repeats drawn along the tube axis')
    parser.add_argument('--out', type=str, default=str(HERE / 'templates_preview.png'),
                         help='output image path')
    args = parser.parse_args()

    data = np.load(NPZ)
    natoms = data['natoms']

    if args.indices is not None:
        indices = args.indices
    else:
        candidates = np.arange(len(natoms))
        if args.natm_range:
            lo, hi = args.natm_range
            candidates = candidates[(natoms >= lo) & (natoms <= hi)]
        if len(candidates) == 0:
            raise SystemExit(f'no templates found in natm range {args.natm_range}')
        indices = candidates[:args.count]

    n = len(indices)
    ncols = min(n, 3)
    nrows = -(-n // ncols)
    fig = plt.figure(figsize=(5 * ncols, 5 * nrows))

    for i, idx in enumerate(indices):
        numbers, frac, cell = load_structure(data, idx)
        ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')
        plot_structure(ax, numbers, frac, cell, n_cells=args.n_cells)
        ax.set_title(f'template #{idx}  (N={len(numbers)} atoms)')
        ax.set_xlabel('x (Å)')
        ax.set_ylabel('y (Å)')
        ax.set_zlabel('z (Å, tube axis)')
        ax.legend(loc='upper right', fontsize=7)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f'saved {n} template(s) to {args.out}')


if __name__ == '__main__':
    main()

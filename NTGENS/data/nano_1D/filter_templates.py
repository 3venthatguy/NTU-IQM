"""Quality filter for 1D-nanotube pinning templates (numpy + scipy only).

The pinning cache (nanotube_templates.npz, built by build_templates.py) should hold
only genuine, guidance-friendly nanotubes: physical contacts, a real hollow core, a
tube radius within the real-data envelope (not a giant flat/off-annular loop), a sane
atom count, and a sharply peaked radial density (so the shl mask's density force
has a gradient to pull atoms onto the wall). This module is the single source of truth
for those gates -- build_templates.py imports `template_metrics` / `passes_filter` to
filter at build time, and the forensic notebook imports the same helpers so the DB and
the analysis always agree.

The geometry helpers are the numpy twins of comp_models/Analysis/
nanotube_rtheta_forensics.ipynb (which derived the thresholds). No ase/torch/pymatgen.

CLI (renders the research-poster figure: filter funnel table + 4 example tubes in 3D):
    python filter_templates.py --src A.pkl:synthetic B.json:real --poster filter_poster.png
    python filter_templates.py --from-npz nanotube_templates.npz --poster filter_poster.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent

# --- Filter thresholds (mirror the forensics notebook cell B) ----------------
MIN_CONTACT = 0.7      # Å   minimum nearest-neighbour distance (reject overlaps)
MIN_RMIN    = 1.00     # Å   inner hollow radius (r_min, 5th pct); reject filled cores
NATM_MIN    = 4        # atoms  lower bound (need a real 2D cross-section)
NATM_MAX    = 128      # atoms  upper bound (DB reduction cap)
MAX_RMAX    = 8.0      # Å   tube radius cap (real DFT envelope: real max 7.3 Å); reject
                       #     giant flat/off-annular loops (14-28 Å) whose axis mis-detects
PEAK_RATIO  = 2.0      # ρ_peak/ρ_mean (density-guidance viability; flat ρ -> no force)

# Ordered gates for the funnel/table: (key, math symbol, short description, predicate).
GATES = [
    ("contacts", f"d_min ≥ {MIN_CONTACT:.1f} Å", "min interatomic distance",
     lambda m: m["min_nn"] >= MIN_CONTACT),
    ("hollow", f"r_min ≥ {MIN_RMIN:.1f} Å", "inner hollow radius (5th pct)",
     lambda m: m["r_min"] >= MIN_RMIN),
    ("radius", f"r_max ≤ {MAX_RMAX:.0f} Å", "tube radius (real-data envelope)",
     lambda m: m["r_max"] <= MAX_RMAX),
    ("atoms", f"{NATM_MIN} ≤ N ≤ {NATM_MAX}", "atom-count window",
     lambda m: NATM_MIN <= m["nsites"] <= NATM_MAX),
    ("peaked ρ", "ρ_peak/ρ̄ ≥ %.1f" % PEAK_RATIO, "radial-density peakedness",
     lambda m: m["peak_ratio"] >= PEAK_RATIO),
]

# Z -> element symbol (index = atomic number; X=0), for plot legends. No ase/pymatgen.
_SYMBOLS = ['X','H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S',
    'Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As',
    'Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn',
    'Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho',
    'Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po',
    'At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm']

_OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9",
          "#F0E442", "#000000"]


def symbols_of(numbers):
    return np.array([_SYMBOLS[z] if 0 <= z < len(_SYMBOLS) else str(z) for z in numbers])


# ============================================================================
# Geometry helpers (numpy twins of the forensics notebook)
# ============================================================================
def detect_tube_axis(frac):
    "Lattice direction the atoms fill most (smallest circular vacuum gap)."
    frac = np.asarray(frac, float)
    occ = []
    for k in range(3):
        f = np.sort(frac[:, k] % 1.0)
        if len(f) < 2:
            occ.append(0.0); continue
        gaps = np.diff(np.concatenate([f, [f[0] + 1.0]]))
        occ.append(1.0 - gaps.max())
    return int(np.argmax(occ))


def cylindrical_coords(cart, cell, axis):
    "Return r, theta, z_axial, u, v in the frame perpendicular to the tube axis."
    cart = np.asarray(cart, float); cell = np.asarray(cell, float)
    a_hat = cell[axis] / np.linalg.norm(cell[axis])
    z = cart @ a_hat
    perp = cart - np.outer(z, a_hat)
    other = [k for k in range(3) if k != axis]
    e1 = cell[other[0]] - (cell[other[0]] @ a_hat) * a_hat
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(a_hat, e1)
    ctr = perp.mean(0)
    u = (perp - ctr) @ e1
    v = (perp - ctr) @ e2
    return np.hypot(u, v), np.arctan2(v, u), z, u, v


def radial_density(r, L, nbins=24):
    "Jacobian-corrected radial number density rho(r)."
    r = np.asarray(r, float)
    if r.size < 2 or np.ptp(r) < 1e-6:
        return np.array([float(r.mean()) if r.size else 0.0]), \
               np.array([float(r.size)]), np.array([r.size])
    nb = int(min(nbins, max(3, r.size)))
    counts, edges = np.histogram(r, bins=nb)
    ctr = 0.5 * (edges[:-1] + edges[1:]); dr = np.diff(edges)
    area = 2 * np.pi * ctr * dr
    rho = np.where(area > 0, counts / (area * max(L, 1e-9)), 0.0)
    return ctr, rho, counts


def nn_distances(cart, cell, axis, k=1):
    "Nearest-neighbour distance per atom, axial periodicity via +/-1 image tiling."
    cart = np.asarray(cart, float); cell = np.asarray(cell, float)
    imgs = np.vstack([cart + s * cell[axis] for s in (-1, 0, 1)])
    d, _ = cKDTree(imgs).query(cart, k=k + 1)
    return d[:, 1:]


# ============================================================================
# Per-template metrics + gate evaluation
# ============================================================================
def template_metrics(nums, frac, cell):
    "Gate metrics for one template: min_nn, r_min, r_max, axial_len, aspect, nsites, peak_ratio."
    frac = np.asarray(frac, float); cell = np.asarray(cell, float)
    n = len(frac)
    cart = frac @ cell
    ax = detect_tube_axis(frac)
    L = float(np.linalg.norm(cell[ax]))
    r, th, z, u, v = cylindrical_coords(cart, cell, ax)
    r05, r95 = (np.percentile(r, [5, 95]) if r.size else (0.0, 0.0))
    _, rho, _ = radial_density(r, L)
    pos = rho[rho > 0]
    peak_ratio = float(rho.max() / pos.mean()) if pos.size else 0.0
    nn = nn_distances(cart, cell, ax).ravel() if n > 1 else np.array([])
    return dict(nsites=int(n), min_nn=float(nn.min()) if nn.size else np.nan,
                r_min=float(r05), r_max=float(r95), axial_len=L,
                aspect=float(r95 / L) if L > 1e-9 else np.inf, peak_ratio=peak_ratio)


def passes_filter(m):
    "True iff a metrics dict clears every gate."
    return all(fn(m) for _, _, _, fn in GATES)


def funnel(metrics_list):
    """Cumulative-funnel survivor counts over a list of metrics dicts.
    Returns (total, [(symbol, description, cum_survivors), ...], final_survivors)."""
    total = len(metrics_list)
    keep = np.ones(total, bool)
    rows = []
    for _, sym, desc, fn in GATES:
        this = np.array([bool(fn(m)) for m in metrics_list]) if total else np.array([], bool)
        keep &= this
        rows.append((sym, desc, int(keep.sum())))
    return total, rows, int(keep.sum())


# ============================================================================
# Poster figure: funnel table + 3 example tubes in 3D
# ============================================================================
def draw_tube_3d(ax, rec, tag=None):
    """3D perspective of one tube: atoms coloured by element, r_min (blue) / r_max
    (red) cylinders about the tube axis, and x/y/z bounding-box extents in Å.
    `tag` in {'real','synthetic'} colours + prefixes the panel title with its provenance."""
    import itertools
    cart = np.asarray(rec["cart"], float); cell = np.asarray(rec["cell"], float)
    els = symbols_of(rec["numbers"])
    ax_i = detect_tube_axis(rec["frac"])
    a_hat = cell[ax_i] / np.linalg.norm(cell[ax_i])
    z = cart @ a_hat
    perp = cart - np.outer(z, a_hat)
    other = [k for k in range(3) if k != ax_i]
    e1 = cell[other[0]] - (cell[other[0]] @ a_hat) * a_hat
    e1 = e1 / np.linalg.norm(e1); e2 = np.cross(a_hat, e1)
    ctr = perp.mean(0)
    r = np.hypot((perp - ctr) @ e1, (perp - ctr) @ e2)
    r_min, r_max = (np.percentile(r, [5, 95]) if r.size else (0.0, 0.0))
    z0, z1 = (z.min(), z.max()) if z.size else (0.0, 1.0)

    uniq = sorted(set(els))
    cmap = {el: _OKABE[i % len(_OKABE)] for i, el in enumerate(uniq)}
    for el in uniq:
        mm = els == el
        ax.scatter(cart[mm, 0], cart[mm, 1], cart[mm, 2], s=42, color=cmap[el],
                   edgecolors="k", linewidths=0.3, label=el, depthshade=True)
    th = np.linspace(0, 2 * np.pi, 48); TZ = np.array([z0, z1])
    for R, col in [(r_min, "#277DA1"), (r_max, "#F94144")]:
        TT, ZZ = np.meshgrid(th, TZ)
        S = (ctr[None, None, :] + ZZ[..., None] * a_hat[None, None, :]
             + R * (np.cos(TT)[..., None] * e1[None, None, :]
                    + np.sin(TT)[..., None] * e2[None, None, :]))
        ax.plot_surface(S[..., 0], S[..., 1], S[..., 2], color=col, alpha=0.12,
                        linewidth=0, shade=False)
        ring = ctr + z1 * a_hat + R * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)
        ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=col, lw=1.6)
    lo, hi = cart.min(0), cart.max(0); d = hi - lo
    corners = np.array(list(itertools.product(*zip(lo, hi))))
    for i, j in itertools.combinations(range(len(corners)), 2):
        if int((corners[i] != corners[j]).sum()) == 1:
            seg = np.vstack([corners[i], corners[j]])
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="0.6", lw=0.5, alpha=0.5)
    ax.text((lo[0] + hi[0]) / 2, lo[1], lo[2], f"x = {d[0]:.1f} Å", fontsize=8)
    ax.text(hi[0], (lo[1] + hi[1]) / 2, lo[2], f"y = {d[1]:.1f} Å", fontsize=8)
    ax.text(hi[0], lo[1], (lo[2] + hi[2]) / 2, f"z = {d[2]:.1f} Å", fontsize=8)
    try:
        ax.set_box_aspect(tuple(np.maximum(d, 1e-3)))
    except Exception:
        pass
    ax.view_init(elev=18, azim=-60)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    formula = "".join(sorted(set(els)))
    tag_col = {"real": "#009E73", "synthetic": "#E69F00"}.get(tag, "k")
    prefix = f"[{tag.upper()}]  " if tag else ""
    ax.set_title(f"{prefix}{formula}  (N={len(cart)})\n"
                 f"r$_{{min}}$={r_min:.2f}  r$_{{max}}$={r_max:.2f} Å",
                 fontsize=9, color=tag_col, fontweight="bold")
    ax.legend(fontsize=6, loc="upper left", framealpha=0.6, handletextpad=0.1)


def poster_figure(metrics_list, survivor_records, out_path, seed=0):
    """Render the research-poster figure: cumulative-funnel table (top) + four random
    filtered tubes in 3D (bottom) — 2 real + 2 synthetic, each panel labelled by
    provenance. Saves a PNG to out_path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d)

    total, rows, survivors = funnel(metrics_list)
    pct = (lambda k: f"{100 * k / total:.1f}%" if total else "—")

    fig = plt.figure(figsize=(16, 8.8))
    gs = fig.add_gridspec(2, 4, height_ratios=[0.95, 1.55], hspace=0.30, wspace=0.06)

    # --- Filter funnel table (spans the top row) ---
    axt = fig.add_subplot(gs[0, :]); axt.axis("off")
    cell_text, row_colors = [], []
    for sym, desc, keep in rows:
        cell_text.append([sym, desc, pct(keep)])
        row_colors.append(["#eef3f6"] * 3)
    cell_text.append(["all gates", "surviving templates", pct(survivors)])
    row_colors.append(["#d7ece3"] * 3)
    tbl = axt.table(cellText=cell_text,
                    colLabels=["filter", "description", "cumulative pass %"],
                    cellColours=row_colors, cellLoc="left", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.6)
    for (row, col), c in tbl.get_celld().items():
        c.set_edgecolor("white")
        if row == 0:
            c.set_facecolor("#277DA1"); c.set_text_props(color="white", weight="bold")
        if row == len(cell_text):  # final survivors row
            c.set_text_props(weight="bold")
    axt.set_title(f"1D-nanotube template quality filter   (n = {total:,} input templates)",
                  fontsize=13, fontweight="bold", pad=12)

    # --- Four random filtered tubes: 2 real + 2 synthetic, labelled by provenance ---
    rng = np.random.default_rng(seed)

    def _pick(code, k):
        pool = [r for r in survivor_records if int(r.get("source", -1)) == code]
        if not pool:
            return []
        idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
        return [pool[i] for i in idx]

    chosen = [(r, "real") for r in _pick(1, 2)] + [(r, "synthetic") for r in _pick(0, 2)]
    if len(chosen) < 4:                               # untagged/short cache -> top up
        used = {id(r) for r, _ in chosen}
        extra = [r for r in survivor_records if id(r) not in used]
        if extra:
            ei = rng.choice(len(extra), size=min(4 - len(chosen), len(extra)), replace=False)
            chosen += [(extra[i], None) for i in ei]

    for c, (rec, tag) in enumerate(chosen[:4]):
        ax = fig.add_subplot(gs[1, c], projection="3d")
        draw_tube_3d(ax, rec, tag=tag)
    fig.text(0.5, 0.015,
             "example tubes  ·  [REAL] = DFT Alexandria (green)   ·   "
             "[SYNTHETIC] = stage0 generated (orange)",
             ha="center", fontsize=11, fontweight="bold")

    n_real = sum(t == "real" for _, t in chosen)
    n_syn = sum(t == "synthetic" for _, t in chosen)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote poster: {out_path}  ({survivors:,}/{total:,} survivors; "
          f"panels: {n_real} real + {n_syn} synthetic)")


# ============================================================================
# Record loaders (shared with the forensic notebook)
# ============================================================================
def load_npz_records(path):
    "Read the CSR template npz -> list of dicts {numbers, frac, cell, cart, nsites, source}."
    z = np.load(path)
    numbers, frac, cells = z["numbers"], z["frac_coords"], z["cells"]
    splits, natoms = z["splits"], z["natoms"]
    source = z["source"] if "source" in z.files else np.full(len(natoms), -1, np.int8)
    recs = []
    for i in range(len(natoms)):
        lo, hi = int(splits[i]), int(splits[i + 1])
        nums = np.asarray(numbers[lo:hi]); fr = np.asarray(frac[lo:hi], float)
        cell = np.asarray(cells[i], float)
        recs.append(dict(numbers=nums, frac=fr, cell=cell, cart=fr @ cell,
                         nsites=int(natoms[i]), source=int(source[i])))
    return recs


def _iter_reduced(specs, target=128, tol=0.25):
    "Yield reduced records {numbers, frac, cell, cart, nsites, source} from source specs."
    sys.path.insert(0, str(HERE))
    from load_sources import iter_sources
    from reduce_templates import reduce_structure
    for rec in iter_sources(specs):
        r = reduce_structure(rec.numbers, rec.frac_coords, rec.cell, target=target, tol=tol)
        if r is None:
            continue
        nums, frac, cell, _ = r
        frac = np.asarray(frac, float) % 1.0
        yield dict(numbers=np.asarray(nums), frac=frac, cell=np.asarray(cell, float),
                   cart=frac @ np.asarray(cell, float), nsites=len(nums),
                   source=1 if rec.source == 'real' else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Render the filter poster figure.")
    ap.add_argument("--src", nargs="+", metavar="PATH[:label]",
                    help="source datasets (reduce+metrics computed for the funnel table)")
    ap.add_argument("--from-npz", metavar="PATH",
                    help="use an already-built template npz (funnel is over survivors only)")
    ap.add_argument("--poster", default=str(HERE / "filter_poster.png"),
                    help="output PNG path (default: nano_1D/filter_poster.png)")
    ap.add_argument("--target-natm", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.src:
        # Import the spec parser so ':label' works exactly like build_templates.
        sys.path.insert(0, str(HERE))
        from build_templates import _parse_specs
        records, metrics, survivors = [], [], []
        for rec in _iter_reduced(_parse_specs(args.src), target=args.target_natm):
            m = template_metrics(rec["numbers"], rec["frac"], rec["cell"])
            metrics.append(m)
            if passes_filter(m):
                survivors.append(rec)
        poster_figure(metrics, survivors, args.poster, seed=args.seed)
    elif args.from_npz:
        survivors = load_npz_records(args.from_npz)
        metrics = [template_metrics(r["numbers"], r["frac"], r["cell"]) for r in survivors]
        poster_figure(metrics, survivors, args.poster, seed=args.seed)
    else:
        ap.error("give --src <sources> or --from-npz <npz>")

import torch
import numpy as np
import math
import random
from scigen.pl_modules.diffusion_w_type import MAX_ATOMIC_NUM 
Pi = math.pi

chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']


def lattice_params_to_matrix_xy_torch(lengths, angles):
    """Batched torch version to compute lattice matrix from params.

    lengths: torch.Tensor of shape (N, 3), unit A
    angles: torch.Tensor of shape (N, 3), unit degree (alpha, beta, gamma)
    
    Returns:
    A torch.Tensor of shape (N, 3, 3) representing the lattice matrix.
    """
    # Convert angles from degrees to radians
    angles_r = torch.deg2rad(angles)
    # Extract the angles for clarity
    alpha = angles_r[:, 0]
    beta = angles_r[:, 1]
    gamma = angles_r[:, 2]
    # Calculate cosines and sines of the angles
    cos_alpha = torch.cos(alpha)
    cos_beta = torch.cos(beta)
    cos_gamma = torch.cos(gamma)
    sin_gamma = torch.sin(gamma)
    # Lattice vector a along x-axis
    vector_a = torch.stack([lengths[:, 0],  # a_x = a
                            torch.zeros_like(lengths[:, 0]),  # a_y = 0
                            torch.zeros_like(lengths[:, 0])], dim=1)  # a_z = 0
    # Lattice vector b in the xy-plane
    vector_b = torch.stack([lengths[:, 1] * cos_gamma,  # b_x = b * cos(gamma)
                            lengths[:, 1] * sin_gamma,  # b_y = b * sin(gamma)
                            torch.zeros_like(lengths[:, 1])], dim=1)  # b_z = 0
    # Lattice vector c in the general 3D direction
    vector_c_x = lengths[:, 2] * cos_beta
    vector_c_y = lengths[:, 2] * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    vector_c_z = lengths[:, 2] * torch.sqrt(1 - cos_beta**2 - ((cos_alpha - cos_beta * cos_gamma) / sin_gamma)**2)
    vector_c = torch.stack([vector_c_x, vector_c_y, vector_c_z], dim=1)
    # Stack the vectors into a (N, 3, 3) matrix
    return torch.stack([vector_a, vector_b, vector_c], dim=1)

mask_l_reduced = torch.tensor([[1, 1, 0]])   #TODO: remove this line after making sure the mask_l is a (3,3) tensor
mask_l_reduced_full = torch.tensor([[1, 1, 1]])
mask_l_default = torch.tensor([[[1, 1, 1], 
                               [1, 1, 1], 
                               [0, 0, 0]]])   
mask_l_cvert = torch.tensor([[[1, 1, 1], 
                            [1, 1, 1], 
                            [1, 1, 0]]])
mask_l_full = torch.ones(3, 3, dtype=torch.int)
mask_l_zonly = torch.tensor([[[0, 0, 0], 
                            [0, 0, 0], 
                            [1, 1, 1]]])
mask_l_zeros = torch.zeros(3, 3, dtype=torch.int)
# Pin the ENTIRE cell (a, b AND the c tube axis). Kept in the (1, 3, 3) shape
# convention of the other 3D masks so batching/broadcasting in sample_scigen works
# (mask_l_full above is (3, 3) and would trip the reduced-mask code path).
mask_l_allfixed = torch.ones(1, 3, 3, dtype=torch.int)

def cart2frac(cart_coords, lattice_matrix): 
    """
    Converts Cartesian coordinates to fractional coordinates.
    
    Parameters:
    - cart_coords: torch.tensor with shape (N, 2) or (N, 3)
    - lattice_vectors: 2x2 or 3x3 matrix of lattice vectors (torch.tensor) as columns

    Returns:
    - Fractional coordinates as ndarray
    """
    # Calculate the inverse of the lattice matrix
    lattice_inv = torch.inverse(lattice_matrix)
    # Calculate fractional coordinates
    fractional_coords = torch.einsum('ij,ki->kj', lattice_inv, cart_coords)
    return fractional_coords

# --- Cylindrical geometry for the 1D-nanotube radial envelope -----------------
# These mirror the conventions in comp_models/Analysis/nanotube_rtheta_forensics.ipynb
# (tube-axis detection by occupied fraction; Gram-Schmidt transverse frame) so the
# per-template r_max used to constrain decorators matches that analysis. The numpy
# helpers run once per template at dataset-build time; the torch helper runs on
# device inside the sampler.

def detect_tube_axis(frac_coords):
    """Lattice direction the atoms fill most (smallest circular vacuum gap in the
    wrapped fractional coords). Robust to cell orientation; matches the forensics
    notebook. `frac_coords`: (N, 3) numpy array. Returns axis index in {0,1,2}."""
    frac = np.asarray(frac_coords, dtype=float)
    occ = []
    for k in range(3):
        f = np.sort(frac[:, k] % 1.0)
        if len(f) < 2:
            occ.append(0.0); continue
        gaps = np.diff(np.concatenate([f, [f[0] + 1.0]]))   # wrap-around gaps
        occ.append(1.0 - gaps.max())
    return int(np.argmax(occ))

def cylindrical_frame(cart, cell, axis):
    """Transverse cylindrical frame about the tube axis (numpy).

    cart: (N, 3) Cartesian coords (A). cell: (3, 3) row-vector lattice. axis: the
    tube-axis lattice index. Returns (r, theta, z, centroid, a_hat, e1, e2) where
    r/theta/z are (N,), and centroid/a_hat/e1/e2 are (3,) vectors defining the
    frame: a_hat = tube axis, e1/e2 = orthonormal transverse basis (Gram-Schmidt,
    so non-orthogonal cells are handled), centroid = cross-section centroid."""
    cart = np.asarray(cart, dtype=float)
    cell = np.asarray(cell, dtype=float)
    a_hat = cell[axis] / np.linalg.norm(cell[axis])
    z = cart @ a_hat                                   # axial coordinate
    perp = cart - np.outer(z, a_hat)                   # transverse component
    other = [k for k in range(3) if k != axis]
    e1 = cell[other[0]] - (cell[other[0]] @ a_hat) * a_hat
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(a_hat, e1)                            # completes the frame
    ctr = perp.mean(0)                                 # cross-section centroid
    u = (perp - ctr) @ e1
    v = (perp - ctr) @ e2
    r = np.hypot(u, v)
    theta = np.arctan2(v, u)
    return r, theta, z, ctr, a_hat, e1, e2

def estimate_template_bounds(frac_known, cell, r_percentile=95.0,
                             r_lo_percentile=5.0):
    """Per-template cylindrical bounds from a pinned skeleton (numpy, build time).

    frac_known: (N, 3) fractional coords. cell: (3, 3) row-vector lattice. Returns
    a dict {axis, r_max, r_min, centroid, a_hat, e1, e2} where r_max is the
    skeleton's r_percentile-th radius (a robust "tube radius" from the forensics
    analysis), r_min is its r_lo_percentile-th radius (the wall's inner edge), and
    the frame vectors are Cartesian (A). The sc='shl' shell uses [r_min, r_max] as
    the band that confines every generated atom."""
    frac = np.asarray(frac_known, dtype=float) % 1.0
    cell = np.asarray(cell, dtype=float)
    cart = frac @ cell
    axis = detect_tube_axis(frac)
    r, theta, z, ctr, a_hat, e1, e2 = cylindrical_frame(cart, cell, axis)
    r_max = float(np.percentile(r, r_percentile)) if r.size else 0.0
    r_min = float(np.percentile(r, r_lo_percentile)) if r.size else 0.0
    return {'axis': int(axis), 'r_max': r_max, 'r_min': r_min, 'centroid': ctr,
            'a_hat': a_hat, 'e1': e1, 'e2': e2}


def estimate_radial_density_force(frac_known, cell, grid_size=96,
                                  bandwidth_scale=1.0, force_clip=None,
                                  r_grid_max=None):
    """Per-template radial log-density guidance force (numpy, build time; v2).

    Builds the empirical marginal distribution rho(r) of the template's atom radii
    (a Gaussian KDE) and tabulates the guidance force f(r) = d/dr log rho(r) on a
    uniform grid. At sample time the sampler interpolates f at each generated atom's
    radius and steps the radius up the gradient, so atoms concentrate onto the real
    wall position(s) (the modes of rho) instead of filling the whole band. A
    bimodal rho (double-wall) yields two attractors automatically.

    We KDE the RAW radii (not the Jacobian-corrected density): the goal is to match
    the template's radial *distribution* of atoms, whose modes sit at the wall(s).
    The KDE bandwidth is the empirical wall thickness (Silverman's rule scaled by
    bandwidth_scale) -> data-driven, per template; <1 sharpens (thinner wall).

    frac_known: (N, 3) fractional coords. cell: (3, 3) row-vector lattice. Returns
    {grid_lo, grid_dr, force (grid_size,), bandwidth}. A degenerate/tiny template
    (or one with no radial spread) returns an all-zero force -> the guidance is a
    no-op and the v1 band alone governs that graph."""
    frac = np.asarray(frac_known, dtype=float) % 1.0
    cell = np.asarray(cell, dtype=float)
    cart = frac @ cell
    axis = detect_tube_axis(frac)
    r, theta, z, ctr, a_hat, e1, e2 = cylindrical_frame(cart, cell, axis)

    grid_lo = 0.0
    if r_grid_max is None:
        r_grid_max = float(r.max() * 1.2) if r.size else 1.0
    r_grid_max = max(r_grid_max, 1e-3)
    grid_dr = (r_grid_max - grid_lo) / max(grid_size - 1, 1)
    zero = {'grid_lo': grid_lo, 'grid_dr': grid_dr,
            'force': np.zeros(grid_size, dtype=float), 'bandwidth': 0.0}

    std = float(np.std(r)) if r.size else 0.0
    if r.size < 3 or std < 1e-6:
        return zero                                    # nothing to shape -> no-op
    # Silverman bandwidth = empirical wall thickness (scaled).
    h = 1.06 * std * (r.size ** (-1.0 / 5.0)) * float(bandwidth_scale)
    h = max(h, 1e-3)

    grid = grid_lo + grid_dr * np.arange(grid_size)    # (G,)
    # Gaussian KDE: rho(g) = mean_i N(g; r_i, h). (G, N) broadcast, small N.
    d = (grid[:, None] - r[None, :]) / h
    rho = np.exp(-0.5 * d * d).sum(1) / (r.size * h * math.sqrt(2 * Pi))
    log_rho = np.log(rho + 1e-12)
    force = np.gradient(log_rho, grid_dr)              # d/dr log rho, (G,)
    if force_clip is None:
        force_clip = 5.0 / h                           # ~a few wall-widths per unit
    force = np.clip(force, -force_clip, force_clip)

    # Tail robustness: far from every template radius rho underflows to the eps
    # floor, so log rho is flat and the KDE force vanishes -> an atom stranded in
    # the tail would get no guidance. Blend in a gentle constant pull toward the
    # density-weighted mean radius, weighted by how "in-support" each grid point is
    # (w = rho/rho.max()): in-support points keep the precise KDE force, tail points
    # drift back toward the wall. In the real pipeline the v1 band already
    # pre-confines atoms near the wall, so this only matters for wide bands.
    w = rho / (rho.max() + 1e-30)                      # (G,) in [0, 1]
    r_bar = float((grid * rho).sum() / (rho.sum() + 1e-30))
    tail_force = np.sign(r_bar - grid) * (0.3 * force_clip)
    force = w * force + (1.0 - w) * tail_force
    return {'grid_lo': grid_lo, 'grid_dr': grid_dr,
            'force': force.astype(float), 'bandwidth': h}

# The on-device torch radial confinement used inside the sampler lives in
# scigen.pl_modules.diff_utils.apply_radial_band (kept there to avoid a circular
# import, since sc_utils imports from that module).


def reflect_across_line(coords, line):
    """
    Reflects multiple points across a line defined by `line = [a, b]` corresponding to `y = ax + b`.
    
    Parameters:
    - coords: torch.tensor, tensor of shape (n, 2) where n is the number of points, each represented by (x, y).
    - line: torch.tensor, tensor of shape (2,) representing the line coefficients [a, b] for the line y = ax + b.
    
    Returns:
    - A tensor of shape (n, 2) representing the reflected points.
    """
    a, b = line
    x1, y1 = coords[:, 0], coords[:, 1]

    # Calculate the projection of (x1, y1) onto the line y = ax + b
    x_proj = (x1 + a * (y1 - b)) / (1 + a**2)
    y_proj = a * x_proj + b
    
    # Calculate the reflection points
    reflected_x = x1 + 2 * (x_proj - x1)
    reflected_y = y1 + 2 * (y_proj - y1)
    
    return torch.stack([reflected_x, reflected_y], dim=1)


def vector_to_line_equation(vector, points):    
    vx, vy = vector[0], vector[1]
    if vx == 0:
        raise ValueError("The vector defines a vertical line, not representable as y = ax + b")
    
    x0, y0 = points[:, 0], points[:, 1]
    a = vy / vx
    b = y0 - a * x0
    
    return torch.stack([a.expand_as(b), b], dim=1)

class SC_Base():
    """
    Base class for structural constraints
    """
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device):
        self.bond_len = bond_len
        self.num_atom = int(num_atom) if num_atom is not None else None
        self.type_known = type_known
        self.frac_z = frac_z
        self.c_vec_cons = c_vec_cons
        self.reduced_mask = reduced_mask
        self.device = device
        self.a_scale, self.b_scale, self.c_scale = None, None, None     # Initialize lattice scaling wrt self.bond_len
        self.mask_l = self.get_mask_l()
        self.frac_known = None  # Initialize known fractional coordinates
        self.num_known = None   # Initialize number of known atoms
        self.use_constraints = True

    def get_cell(self, alpha=90, beta=90, gamma=90):
        angles = [alpha, beta, gamma]
        self.c_scale = self.c_vec_cons['scale'] if self.c_vec_cons['scale'] is not None else np.mean([self.a_scale, self.b_scale])
        self.a_len, self.b_len, self.c_len = self.a_scale * self.bond_len, self.b_scale * self.bond_len, self.c_scale * self.bond_len
        self.cell_lengths = torch.tensor([self.a_len, self.b_len, self.c_len], dtype=torch.float, device=self.device)    # lttice lengths in Angstrom
        self.cell_angles_d = torch.tensor(angles, dtype=torch.float, device=self.device)   # lattice angles in degrees    #TODO: need to set
        return lattice_params_to_matrix_xy_torch(self.cell_lengths.unsqueeze(0), self.cell_angles_d.unsqueeze(0)).squeeze(0)
    

    def get_mask_l(self):
        c_scale, c_vert = self.c_vec_cons['scale'], self.c_vec_cons['vert']
        if c_vert:
            self.c_vec_cons['scale'] = None
            return mask_l_cvert
        
        if self.reduced_mask:
            if c_scale is None:
                self.c_vec_cons['vert'] = False
                return mask_l_reduced
            self.c_vec_cons['vert'] = True
            return mask_l_reduced_full
        
        else: 
            if c_scale is None:
                if c_vert:
                    return mask_l_cvert
                else:
                    return mask_l_default
            else:
                self.c_vec_cons['vert'] = True
                return mask_l_full

    def frac_coords_all(self):
        # self.num_atom, self.frac_known
        fcoords_zero = torch.zeros(self.num_atom, 3)
        # n_kn = self.frac_known.shape[0]
        fcoords, mask = fcoords_zero.clone(), fcoords_zero.clone()
        fcoords[:self.num_known, :] = self.frac_known
        fcoords = fcoords%1
        mask[:self.num_known, :] = torch.ones_like(self.frac_known) 
        if self.reduced_mask:
            mask = mask[:, 0].flatten()   #TODO: mask dimension must be (N, 3), which was transformed  from (N,) in the original code
        # return fcoords, mask
        self.frac_coords, self.mask_x = fcoords, mask
        if not self.use_constraints:
            self.mask_x = torch.zeros_like(self.mask_x)

    def atm_types_all(self):
        # self.num_atom, self.num_known, self.type_known
        types_idx_known = [chemical_symbols.index(self.type_known)] * self.num_known
        types_unk = random.choices(chemical_symbols[1:MAX_ATOMIC_NUM+1], k=int(self.num_atom-self.num_known))
        types_idx_unk = [chemical_symbols.index(elem) for elem in types_unk]    # list of unknown atom types (randomly chosen)
        types = torch.tensor(types_idx_known + types_idx_unk)
        mask = torch.zeros_like(types)
        mask[:self.num_known] = 1
        self.atom_types, self.mask_t = types, mask
        if not self.use_constraints:
            self.mask_t = torch.zeros_like(self.mask_t) 


class SC_Vanilla(SC_Base):
    """
    Vanilla case with no constraints
    """
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device)
        self.use_constraints = False
        # Lattice 
        self.a_scale, self.b_scale = 1, 1 
        self.cell = self.get_cell(gamma=90)
        # coords
        self.frac_known = torch.tensor([[0.0, 0.0, self.frac_z]]) 
        self.num_known = self.frac_known.shape[0]
        self.mask_l = torch.zeros_like(self.mask_l)
        

class SC_Template(SC_Base):
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device)
        # Lattice 
        self.a_scale, self.b_scale = 1, 1   #TODO: set the lattice scaling wrt self.bond_len.
        self.cell = self.get_cell(gamma=90) #TODO: set the lattice matrix by giving the lattice angle \gamma.
        # coords
        self.frac_known = torch.tensor([[0.0, 0.0, self.frac_z],
                                        [0.5, 0.0, self.frac_z]])   #TODO: set the fractional coordinates of the constrained atoms.
        self.num_known = self.frac_known.shape[0]
        

# --- Private synthetic-ring fallback for sc='shl' -----------------------------

# Default geometry ranges for the synthetic ring geometry. Used only by the
# private _SC_NanotubeFallback below, which sc='shl' falls back to when the
# Alexandria template DB has no structure in the requested atom-count range.
NANOTUBE_DEFAULTS = {
    'n_circ_range': (4, 10),      # number of skeleton atoms around one ring
    'vacuum': 15.0,               # Angstrom padding around the tube in the x,y box
    'axial_per_ring': 1.0,        # axial repeat length as a multiple of bond_len
}


class _SC_NanotubeFallback(SC_Base):
    """
    PRIVATE synthetic-ring fallback for sc='shl' (NOT a user-facing mode).

    This is not registered in sc_dict and has no CLI exposure. It exists only as
    the safety net the sc='shl' path uses when the real 1D-nanotube template DB
    has no structure matching the requested natm_range (see gen_utils.process()).

    Pins one element's skeleton atoms on a ring (the cylinder cross-section) of
    radius R around the c/z axis. The a,b lattice vectors form a large vacuum box
    so periodic images in the transverse plane do not interact; c is the periodic
    tube axis, kept vertical with free length via mask_l_cvert. The diffusion
    model then decorates the remaining (unknown) atoms.

    Geometry parameters (n_circ / chirality / axial_repeat / vacuum) default to
    NANOTUBE_DEFAULTS when left None.

    Requires reduced_mask=False so mask_x stays (N, 3) — the ring atoms are pinned
    per-coordinate.
    """
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons,
                 reduced_mask, device, n_circ=None, chirality=None,
                 axial_repeat=None, vacuum=None):
        # Draw geometry (hybrid: DB-informed values or NANOTUBE_DEFAULTS).
        self.n_circ = int(n_circ) if n_circ is not None else \
            random.randint(*NANOTUBE_DEFAULTS['n_circ_range'])
        self.chirality = chirality          # optional (n, m) -> helical z offset
        self.vacuum = vacuum if vacuum is not None else NANOTUBE_DEFAULTS['vacuum']
        self.axial_repeat = axial_repeat if axial_repeat is not None else \
            NANOTUBE_DEFAULTS['axial_per_ring'] * bond_len
        # Ring radius so adjacent skeleton atoms sit one bond_len apart.
        self.radius = bond_len / (2 * math.sin(Pi / self.n_circ))

        super().__init__(bond_len, num_atom, type_known, frac_z, c_vec_cons,
                         reduced_mask, device)

        self.cell = self.get_cell()
        self.frac_known = self._ring_frac_coords()
        self.num_known = self.frac_known.shape[0]

    def get_mask_l(self):
        # Fix the vacuum box (a, b vectors); keep c vertical with free length.
        return mask_l_cvert

    def get_cell(self, gamma=90):
        box = 2 * self.radius + self.vacuum        # vacuum box side length (A)
        self.a_len, self.b_len, self.c_len = box, box, self.axial_repeat
        self.cell_lengths = torch.tensor(
            [self.a_len, self.b_len, self.c_len], dtype=torch.float, device=self.device)
        self.cell_angles_d = torch.tensor(
            [90, 90, gamma], dtype=torch.float, device=self.device)
        return lattice_params_to_matrix_xy_torch(
            self.cell_lengths.unsqueeze(0), self.cell_angles_d.unsqueeze(0)).squeeze(0)

    def _ring_frac_coords(self):
        # Build the ring on CPU so it matches the CPU tensors in frac_coords_all().
        box_center = torch.tensor([[self.a_len / 2, self.b_len / 2]])
        angles = torch.arange(self.n_circ, dtype=torch.float) * (2 * Pi / self.n_circ)
        cart_xy = torch.stack([self.radius * torch.cos(angles),
                               self.radius * torch.sin(angles)], dim=1) + box_center
        cell_xy = self.cell[:2, :2].detach().cpu()
        frac_xy = cart2frac(cart_xy, cell_xy)
        if self.chirality is not None:
            n, m = self.chirality
            z_offsets = (torch.arange(self.n_circ, dtype=torch.float) * (m / max(n, 1))) % 1.0
            frac_z = (self.frac_z + z_offsets).unsqueeze(1)
        else:
            frac_z = self.frac_z * torch.ones((self.n_circ, 1))
        return torch.cat([frac_xy, frac_z], dim=-1)


# --- Geometric-shell constraint (real tube SHAPE, no atoms pinned) ------------

class SC_DBShell(SC_Base):
    """Pin only the *geometry* of a real Alexandria 1D tube, not its atoms (sc='shl').

    Rather than pinning template atoms (which would leave almost no room for
    generation), this keeps the template's cell (so the tube axis, axial period and
    vacuum box define a meaningful (r, theta, z) frame) but pins NO atoms: num_known
    is 0, so mask_x/mask_t are all-zero and the diffusion model generates every
    atom's position AND species freely. A soft, psi(t)-ramped radial band
    [r_min, r_max] (computed by gen_utils from the template and carried on the batch)
    confines all atoms onto the wall shell during denoising -- enforcing "tube
    shaped" without dictating what fills it.

    The template's frac/atomic-numbers are accepted only so gen_utils can measure
    the band from the same object; they are intentionally discarded here. Requires
    reduced_mask=False. The target atom count (= template nsites) is set by
    gen_utils via self.num_atom before frac_coords_all()/atm_types_all() run.
    """
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons,
                 reduced_mask, device, frac_known=None, atom_numbers_known=None,
                 cell=None):
        super().__init__(bond_len, num_atom, type_known, frac_z, c_vec_cons,
                         reduced_mask, device)
        # Fix the real cell; pin no atoms (geometry-only constraint).
        self.cell = torch.as_tensor(cell, dtype=torch.float, device=device)
        self.frac_known = torch.zeros(0, 3, dtype=torch.float)
        self.num_known = 0
        a, b = self.cell[0].norm().item(), self.cell[1].norm().item()
        self.a_len, self.b_len, self.c_len = a, b, self.cell[2].norm().item()

    def get_mask_l(self):
        # Pin the WHOLE real cell (a, b AND the c tube axis) so the (r, theta, z)
        # frame and vacuum box stay the template's ground-truth geometry.
        return mask_l_allfixed


# NTGEN is nanotube-only: real tube geometry with free de-novo generation ('shl')
# and the unconstrained vanilla model ('van'). 'shl' pins only the tube SHAPE (a
# soft radial band, no atoms), so the model generates every atom. Extend with new
# tube types here. (_SC_NanotubeFallback is deliberately NOT registered — it is a
# private synthetic-ring safety net used only by the 'shl' path.)
sc_dict = {'shl': SC_DBShell, 'van': SC_Vanilla}

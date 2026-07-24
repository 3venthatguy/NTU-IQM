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
        

# --- 1D nanotube constraint (Pathway 3: mask-based generation) ----------------

# Default geometry ranges for the hybrid parameter sampler. These act as a
# stand-in distribution until a real 1D-nanotube database is wired in through
# nanotube_param_sampler() in gen_utils.py (see plan File 3). SC_Nanotube draws
# from these whenever the corresponding argument is left as None.
NANOTUBE_DEFAULTS = {
    'n_circ_range': (4, 10),      # number of skeleton atoms around one ring
    'vacuum': 15.0,               # Angstrom padding around the tube in the x,y box
    'axial_per_ring': 1.0,        # axial repeat length as a multiple of bond_len
}


class SC_Nanotube(SC_Base):
    """
    1D nanotube constraint (skeleton-sublattice, hybrid geometry).

    Pins one element's skeleton atoms on a ring (the cylinder cross-section) of
    radius R around the c/z axis. The a,b lattice vectors form a large vacuum box
    so periodic images in the transverse plane do not interact; c is the periodic
    tube axis, kept vertical with free length via mask_l_cvert. The diffusion
    model then decorates the remaining (unknown) atoms, exactly as it fills a 2D
    motif in the other SC_* classes.

    Geometry parameters (n_circ / chirality / axial_repeat / vacuum) are the
    "hybrid" hook: pass values drawn from a database-informed distribution, or
    leave them None to sample from NANOTUBE_DEFAULTS.

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


# --- Carbon nanotube constraint (rolled graphene wall) ------------------------

# Default geometry for SC_CarbonTube. Chiralities are kept small so the wall atom
# count N = 4(n^2+nm+m^2)/d_R stays below the carbon_24 atom-count ceiling (24):
# (3,3)->12, (4,4)->16, (5,5)->20, (4,0)->16, (5,0)->20 atoms per axial period.
CARBONTUBE_DEFAULTS = {
    'chirality_options': [(3, 3), (4, 4), (5, 5), (4, 0), (5, 0)],
    'a_cc': 1.42,                 # C-C bond length in graphene (Angstrom)
    'vacuum': 15.0,               # Angstrom padding around the tube in the x,y box
}


class SC_CarbonTube(SC_Base):
    """
    Carbon nanotube constraint (rolled graphene wall, Pathway 3).

    Builds the full CNT wall for chiral indices (n, m) by the standard chiral-
    vector construction: a graphene sheet is cut along Ch = n*a1 + m*a2 and the
    translational vector T, then rolled so Ch becomes the tube circumference.
    All N = 4(n^2+nm+m^2)/d_R wall atoms of one axial period are pinned as known
    carbon atoms; the diffusion model may decorate any remaining atoms.

    Same cell/mask plumbing as SC_Nanotube: a,b span a vacuum box (fixed by the
    mask), c is the periodic tube axis (vertical, free length via mask_l_cvert).
    Requires reduced_mask=False. type_known is forced to 'C'.
    """
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons,
                 reduced_mask, device, chirality=None, a_cc=None, vacuum=None):
        self.chirality = tuple(chirality) if chirality is not None else \
            random.choice(CARBONTUBE_DEFAULTS['chirality_options'])
        self.a_cc = a_cc if a_cc is not None else CARBONTUBE_DEFAULTS['a_cc']
        self.vacuum = vacuum if vacuum is not None else CARBONTUBE_DEFAULTS['vacuum']

        n, m = self.chirality
        a = self.a_cc * math.sqrt(3)                    # graphene lattice constant
        d_R = math.gcd(2 * n + m, 2 * m + n)
        self.circumference = a * math.sqrt(n * n + n * m + m * m)    # |Ch|
        self.radius = self.circumference / (2 * Pi)
        self.axial_repeat = math.sqrt(3) * self.circumference / d_R  # |T|
        self.num_wall = 4 * (n * n + n * m + m * m) // d_R

        # The wall is carbon by construction, whatever known_species was passed.
        super().__init__(bond_len, num_atom, 'C', frac_z, c_vec_cons,
                         reduced_mask, device)

        self.cell = self.get_cell()
        self.frac_known = self._wall_frac_coords()
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

    def _wall_frac_coords(self):
        """Enumerate one (Ch, T) cell of graphene and wrap it onto the cylinder.

        The positions around (s) and along (t) the tube are exact integer
        rationals — s = s_num / [2(n^2+nm+m^2)], t = t_num / [2(t1^2+t1*t2+t2^2)]
        for atom p*a1 + q*a2 (+ basis) — so periodic duplicates collapse exactly,
        with no floating-point tolerance games."""
        n, m = self.chirality
        d_R = math.gcd(2 * n + m, 2 * m + n)
        t1, t2 = (2 * m + n) // d_R, -(2 * n + m) // d_R
        s_den = 2 * (n * n + n * m + m * m)
        t_den = 2 * (t1 * t1 + t1 * t2 + t2 * t2)

        # Sweep enough graphene cells (p, q) to cover the (Ch, T) supercell; the
        # B sublattice sits at (a1+a2)/3, contributing the b*(...) terms below.
        span = abs(n) + abs(m) + abs(t1) + abs(t2) + 2
        seen, s_list, t_list = set(), [], []
        for p in range(-span, span + 1):
            for q in range(-span, span + 1):
                for b in (0, 1):
                    s_num = (p * (2 * n + m) + q * (n + 2 * m) + b * (n + m)) % s_den
                    t_num = (p * (2 * t1 + t2) + q * (t1 + 2 * t2) + b * (t1 + t2)) % t_den
                    if (s_num, t_num) in seen:
                        continue
                    seen.add((s_num, t_num))
                    s_list.append(s_num / s_den)     # position around the tube [0,1)
                    t_list.append(t_num / t_den)     # position along the axis [0,1)
        assert len(s_list) == self.num_wall, \
            f'CNT({n},{m}): built {len(s_list)} wall atoms, expected {self.num_wall}'

        s = torch.tensor(s_list, dtype=torch.float)
        t = torch.tensor(t_list, dtype=torch.float)
        theta = 2 * Pi * s
        box_center = torch.tensor([[self.a_len / 2, self.b_len / 2]])
        cart_xy = torch.stack([self.radius * torch.cos(theta),
                               self.radius * torch.sin(theta)], dim=1) + box_center
        cell_xy = self.cell[:2, :2].detach().cpu()
        frac_xy = cart2frac(cart_xy, cell_xy)
        frac_z = ((t + self.frac_z) % 1.0).unsqueeze(1)  # frac_z acts as axial phase
        return torch.cat([frac_xy, frac_z], dim=-1)


# --- Database-template constraint (real 1D nanotube pinned as known skeleton) --

class SC_DBTemplate(SC_Base):
    """Pin a real nanotube structure from the Alexandria 1D database (Pathway 3).

    Unlike SC_Nanotube/SC_CarbonTube (which synthesize a geometry from a few
    parameters), this takes an actual structure loaded by
    gen_utils.nanotube_params_from_db: its per-atom atomic numbers, fractional
    coordinates, and 3x3 cell. All template atoms become known/pinned; the model
    decorates the remaining (num_atom - N) atoms. Because the templates are
    multi-element, atm_types_all is overridden to pin each atom's real species
    instead of a single type_known.

    Requires reduced_mask=False so mask_x stays (N, 3).
    """
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons,
                 reduced_mask, device, frac_known=None, atom_numbers_known=None,
                 cell=None):
        super().__init__(bond_len, num_atom, type_known, frac_z, c_vec_cons,
                         reduced_mask, device)
        # Known atomic numbers -> element-symbol indices (Z is the index into
        # chemical_symbols, so the two coincide) for atm_types_all.
        self.atom_numbers_known = torch.as_tensor(atom_numbers_known,
                                                  dtype=torch.long)
        self.frac_known = torch.as_tensor(frac_known, dtype=torch.float) % 1.0
        self.num_known = self.frac_known.shape[0]
        self.cell = torch.as_tensor(cell, dtype=torch.float, device=device)
        a, b = self.cell[0].norm().item(), self.cell[1].norm().item()
        self.a_len, self.b_len, self.c_len = a, b, self.cell[2].norm().item()

    def get_mask_l(self):
        # Pin the WHOLE real cell (a, b AND the c tube axis). The template is a real
        # DFT structure, so its cell is the ground truth; leaving c free lets a
        # bulk-trained model inflate the tube's period. Fixing all of it keeps the
        # generated cell template-faithful (guards the c-axis).
        return mask_l_allfixed

    def atm_types_all(self):
        # Pin each known atom to its real species; fill decorators randomly.
        types_idx_known = self.atom_numbers_known.tolist()
        n_unk = int(self.num_atom - self.num_known)
        types_unk = random.choices(chemical_symbols[1:MAX_ATOMIC_NUM + 1], k=n_unk)
        types_idx_unk = [chemical_symbols.index(elem) for elem in types_unk]
        types = torch.tensor(types_idx_known + types_idx_unk)
        mask = torch.zeros_like(types)
        mask[:self.num_known] = 1
        self.atom_types, self.mask_t = types, mask
        if not self.use_constraints:
            self.mask_t = torch.zeros_like(self.mask_t)


# NTGEN is nanotube-only: generalized skeleton tube ('ntb'), carbon nanotube
# ('cnt'), real database templates ('alx'), and the unconstrained vanilla model
# ('van'). Extend with new tube types here.
sc_dict = {'ntb': SC_Nanotube, 'cnt': SC_CarbonTube,
           'alx': SC_DBTemplate, 'van': SC_Vanilla}

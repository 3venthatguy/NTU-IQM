import torch
from tqdm import tqdm
from torch_geometric.data import Data
from torch.utils.data import Dataset
import numpy as np
import random   
import pickle as pkl
from sc_utils import *  
from sc_natm import natm_dist


# reference of metallic radii: https://www.sciencedirect.com/science/article/pii/0022190273803574
metallic_radius = {'Mn': 1.292, 'Fe': 1.277, 'Co': 1.250, 'Ni': 1.246, 'Ru': 1.339, 'Nd': 1.821, 'Gd': 1.802, 'Tb': 1.781, 'Dy': 1.773, 'Yb': 1.940}
with open('./data/kde_bond.pkl', 'rb') as file:
    kde_dict = pkl.load(file)

# --- 1D nanotube database (Pathway 3): real structures as pinned templates -----
# The Alexandria 1D-nanotube set (7002 ASE structures) is preprocessed once into a
# compact CSR-style npz by data/alx_1D/build_templates.py (no ase needed at run
# time). We mmap it here and serve one real structure per draw (sc='alx'):
# per-atom numbers + fractional coords + cell, pinned as the known skeleton for
# SC_DBTemplate. Templates are indexed by atom count for O(1) natm_range filtering.

class _NanotubeTemplateDB:
    """Lazy, mmap-backed store of database nanotube templates, bucketed by natoms."""
    def __init__(self, path='./data/alx_1D/nanotube_templates.npz'):
        self.ok = False
        try:
            z = np.load(path, mmap_mode='r')            # arrays stay on disk
        except FileNotFoundError:
            print(f'Warning: {path} not found; sc="alx" falls back to SC_Nanotube. '
                  'Run data/alx_1D/build_templates.py to create it.')
            return
        self._numbers = z['numbers']                    # (sum N,)   int16
        self._frac = z['frac_coords']                   # (sum N, 3) float32
        self._cells = z['cells']                        # (M, 3, 3)  float32
        self._splits = z['splits']                      # (M+1,)     int64
        self._natoms = np.asarray(z['natoms'])          # (M,)       int32
        # Per-structure provenance (0=synthetic, 1=real). Optional: older caches
        # (built before the tagged-DB change) have no `source` array -> treat every
        # template as one untagged pool so source_filter=None keeps working.
        self._source = np.asarray(z['source']) if 'source' in z.files else None
        self._source_codes = {'synthetic': 0, 'real': 1}
        # Bucket structure indices by atom count for cheap range queries.
        self._by_natm = {}
        for i, n in enumerate(self._natoms.tolist()):
            self._by_natm.setdefault(n, []).append(i)
        self.max_natm = int(z['max_natm'])
        self.ok = True
        src_note = ''
        if self._source is not None:
            n_real = int((self._source == 1).sum())
            src_note = f', source: {n_real} real / {len(self._source) - n_real} synthetic'
        print(f'Loaded {len(self._natoms)} nanotube templates '
              f'(natoms {int(self._natoms.min())}-{int(self._natoms.max())}{src_note})')

    def sample(self, natm_min, natm_max, source_filter=None):
        """Return one template with natm_min <= N <= natm_max, or None if none fit.
        The caller keeps room for decoration by passing natm_max = its ceiling - 1.

        source_filter: None -> any source; 'real'/'synthetic' (or the int code) ->
        restrict the pool to that provenance. Ignored when the cache is untagged."""
        if not self.ok:
            return None
        pool = [i for n in range(int(natm_min), int(natm_max) + 1)
                for i in self._by_natm.get(n, [])]
        if source_filter is not None and self._source is not None:
            code = self._source_codes.get(source_filter, source_filter)
            pool = [i for i in pool if int(self._source[i]) == code]
        if not pool:
            return None
        idx = random.choice(pool)
        lo, hi = int(self._splits[idx]), int(self._splits[idx + 1])
        return {
            'frac_known': np.asarray(self._frac[lo:hi]),           # (N, 3)
            'atom_numbers_known': np.asarray(self._numbers[lo:hi]),  # (N,)
            'cell': np.asarray(self._cells[idx]),                  # (3, 3)
        }

NANOTUBE_DB = _NanotubeTemplateDB()

def nanotube_template_from_db(natm_min, natm_max, source_filter=None):
    """Sample a 1D-nanotube template (sc='alx') that leaves room for >=1 decorating
    atom. Returns SC_DBTemplate kwargs, or {} to fall back to the parametric
    SC_Nanotube ('ntb').

    source_filter: None -> pin from any source; 'real'/'synthetic' -> restrict to
    that provenance (e.g. 'real' to pin only DFT structures). Ignored on untagged
    caches, so old npz files behave exactly as before."""
    # num_atom must exceed num_known (= N), so cap N at natm_max - 1.
    tmpl = NANOTUBE_DB.sample(natm_min, max(natm_min, natm_max - 1),
                              source_filter=source_filter)
    return tmpl if tmpl is not None else {}

def nanotube_params_from_db(bond_len=None, element=None):
    """Parametric SC_Nanotube geometry overrides ('ntb'). Empty dict -> use the
    NANOTUBE_DEFAULTS in sc_utils.py. (Kept as the fallback / synthetic path;
    real structures are served by nanotube_template_from_db above.)"""
    return {}

# Hybrid seam for carbon nanotubes (SC_CarbonTube, sc='cnt'): wire a CNT database
# here and sample {chirality, a_cc, vacuum}. Empty dict -> CARBONTUBE_DEFAULTS.
CARBONTUBE_DB = None

def carbontube_params_from_db(bond_len=None):
    """Return SC_CarbonTube geometry overrides sampled from a CNT database.

    Keys may include {'chirality', 'a_cc', 'vacuum'}. An empty dict means "use
    SC_CarbonTube's built-in defaults (random small chirality, a_cc=1.42 A)"."""
    if CARBONTUBE_DB is None:
        return {}
    return {}

# Bond length fallback for known species without an entry in the metallic-bond
# KDE (kde_bond.pkl only covers the magnetic metals above). SC_CarbonTube ignores
# bond_len anyway (it uses a_cc), but SampleDataset.process() still samples one.
fallback_bond_len = {'C': 1.42}

def convert_seconds_short(sec):
    minutes, seconds = divmod(int(sec), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return f"{days:02d}:{hours:02d}:{minutes:02d}:{seconds:02d}"

# I am trying to pass None keyword as a command line parameter to a script as follows, if I explicity mention Category=None it works but the moment I switch to sys.argv[1] it fails, any pointers on how to fix this? Please wriite a function that takes a list of strings and returns a dictionary with the key as the first letter of the string and the value as the string itself. If the first letter is already a key, add the string to the list of values.
def parse_none_or_value(argument, obj=float):
    if argument == 'None':
        return None
    return obj(argument)

class SampleDataset(Dataset):      
    def __init__(self, dataset, 
                 natm_range, 
                 total_num, 
                 bond_sigma_per_mu, 
                 use_min_bond_len, 
                 known_species, 
                 sc_list, 
                 frac_z,
                 c_vec_cons,
                 reduced_mask,
                 seed,
                 device,
                 max_decorators=None,
                 template_source=None,
                 r_lo_percentile=5.0):
        super().__init__()
        self.seed = seed
        set_seeds(self.seed)
        self.dataset = dataset
        self.natm_range = sorted([int(i) for i in natm_range])
        self.natm_min, self.natm_max = self.natm_range[0], self.natm_range[-1]
        print('natm_range: ', [self.natm_min, self.natm_max])
        self.total_num = total_num
        self.bond_sigma_per_mu = bond_sigma_per_mu
        self.use_min_bond_len = use_min_bond_len
        self.known_species = known_species
        self.sc_options = sc_list
        self.sc_list = random.choices(self.sc_options, k=self.total_num)
        self.frac_z = frac_z
        self.c_vec_cons = c_vec_cons
        self.reduced_mask = reduced_mask
        self.device = device
        # Template/skeleton-dominated mode: when set, num_atom = num_known + a small
        # random number of decorator atoms (0..max_decorators), bypassing the dataset
        # atom-count distribution. Required to pin whole real templates (num_known can
        # exceed the dataset's ~20-atom support, which would otherwise zero the
        # distribution and crash). None -> legacy distribution sampling.
        self.max_decorators = max_decorators
        # Provenance filter for sc='alx' pinning: None -> any template, 'real' or
        # 'synthetic' -> pin only that source (see nanotube_template_from_db).
        self.template_source = template_source
        # Inner band-edge percentile for the sc='shl' shell (r_min); ignored by
        # 'alx' (its envelope stays one-sided).
        self.r_lo_percentile = r_lo_percentile
        self.num_atom_distribution()
        self.process()
        self.generate_dataset()   
        
    def get_num_atoms(self):
        self.num_atom_list = []
        for (n_kn, sc) in zip(self.num_known_list, self.sc_list):    
            type_distribution = self.distributions_dict[sc].copy()
            # Set the first n elements to 0
            type_distribution[:n_kn+1] = [0] * (n_kn + 1)
            type_distribution[:self.natm_min] = [0] * self.natm_min
            sum_p = sum(type_distribution)
            assert sum_p > 0.0, f"sum_p is {sum_p}, type_distribution: {type_distribution}"
            type_distribution_norm = [p / sum_p for p in type_distribution] 
            num_atom = np.random.choice(len(type_distribution_norm), 1, p = type_distribution_norm)[0]
            self.num_atom_list.append(num_atom)


    def num_atom_distribution(self):
        if self.dataset == 'uniform':  
            self.distributions_dict = {sc: natm_dist[self.dataset][:self.natm_max+1] for sc in self.sc_options} 
        else:
            self.distributions_dict = {sc: (natm_dist[sc][:self.natm_max+1] if sc in natm_dist.keys() 
                                            else natm_dist[self.dataset][:self.natm_max+1]) for sc in self.sc_options}

    def process(self):
        self.type_known_list = random.choices(self.known_species, k=self.total_num)
        # self.num_known_list =[num_known_dict[sc] for sc in self.sc_list]
        if self.bond_sigma_per_mu is not None:  
            # print('Sample bond length from Gaussian')
            self.radii_list = [metallic_radius[s] for s in self.type_known_list]
            self.bond_mu_list = [2*r for r in self.radii_list]
            self.bond_sigma_list = [b*self.bond_sigma_per_mu for b in self.bond_mu_list]
            self.bond_len_list = [np.random.normal(self.bond_mu_list[i], self.bond_sigma_list[i]) for i in range(self.total_num)]
        else:
            # print('Sample bond length from KDE')
            self.min_bond_len_dict = {elem: 0 for elem in chemical_symbols}
            if self.use_min_bond_len:
                for k, v in metallic_radius.items():
                    self.min_bond_len_dict[k] = 2*v
            self.bond_len_list = [max(kde_dict[s].resample(1).item(), self.min_bond_len_dict[s])
                                  if s in kde_dict else fallback_bond_len.get(s, 1.5)
                                  for s in self.type_known_list]
        self.frac_z_known_list = [random.uniform(0, 1) for _ in range(self.total_num)]
        self.is_carbon = self.dataset == 'carbon_24' 

        self.num_known_list = []
        self.num_atom_list = []
        self.frac_coords_list = []
        self.atom_types_list = []
        self.lattice_list = []
        self.mask_x_list, self.mask_t_list, self.mask_l_list =  [], [], []
        # Per-template cylindrical bounds for the radial envelope (sc='alx' real
        # templates only); None disables the envelope for that sample.
        self.tube_bounds_list = []
        # self.get_num_atoms()
        for i, (sc, type_known, bond_len, frac_z_known) in enumerate(zip(self.sc_list, self.type_known_list, self.bond_len_list, self.frac_z_known_list)):
            sc_obj = sc_dict[sc]
            # Nanotubes take extra geometry overrides drawn from the DB (hybrid);
            # {} for all other constraints leaves their signature unchanged.
            if sc == 'alx':
                # Real 1D-nanotube template pinned as the known skeleton. If no
                # template fits natm_range, fall back to the parametric SC_Nanotube.
                extra_kwargs = nanotube_template_from_db(
                    self.natm_min, self.natm_max, source_filter=self.template_source)
                if not extra_kwargs:
                    sc_obj = sc_dict['ntb']
            elif sc == 'shl':
                # Real 1D-nanotube GEOMETRY only: the template's cell defines the
                # tube frame + radial band, but NO atoms are pinned (num_known=0);
                # the model generates every atom, confined to the shell. Same DB
                # draw as 'alx'; empty -> fall back to the parametric SC_Nanotube.
                extra_kwargs = nanotube_template_from_db(
                    self.natm_min, self.natm_max, source_filter=self.template_source)
                if not extra_kwargs:
                    sc_obj = sc_dict['ntb']
            elif sc == 'ntb':
                extra_kwargs = nanotube_params_from_db(bond_len, type_known)
            elif sc == 'cnt':
                extra_kwargs = carbontube_params_from_db(bond_len)
            else:
                extra_kwargs = {}
            sc_material = sc_obj(bond_len=bond_len,
                              num_atom=None,
                              type_known=type_known,
                              frac_z=frac_z_known,
                              c_vec_cons=self.c_vec_cons,
                              reduced_mask=self.reduced_mask,
                              device=self.device,
                              **extra_kwargs)
            num_known = sc_material.num_known
            shl_template = (sc == 'shl' and bool(extra_kwargs))
            if shl_template:
                # Geometric-shell mode: no atoms pinned (num_known=0). The target
                # atom count is the drawn tube's real nsites, so the generated tube
                # has a realistic wall density. Bypasses both the decorator logic and
                # the dataset distribution.
                num_atom = int(len(extra_kwargs['atom_numbers_known']))
                num_atom = max(num_atom, 1)
            elif self.max_decorators is not None:
                # Template/skeleton-dominated: pin the whole known structure and add
                # only a few model-placed decorator atoms. Bypasses the dataset
                # atom-count distribution (which has no mass above ~20 atoms and would
                # zero out for large pinned templates such as real Alexandria tubes).
                num_atom = int(num_known) + random.randint(0, int(self.max_decorators))
                num_atom = max(num_atom, 1)
            else:
                type_distribution = self.distributions_dict[sc].copy()
                type_distribution[:num_known+1] = [0] * (num_known + 1)
                type_distribution[:self.natm_min] = [0] * self.natm_min
                sum_p = sum(type_distribution)
                assert sum_p > 0.0, f"sum_p is {sum_p}, type_distribution: {type_distribution}"
                type_distribution_norm = [p / sum_p for p in type_distribution]
                num_atom = np.random.choice(len(type_distribution_norm), 1, p = type_distribution_norm)[0]
            sc_material.num_atom = num_atom
            sc_material.frac_coords_all()
            sc_material.atm_types_all()

            # Cylindrical bounds from a real template. 'alx' measures the band from
            # its pinned skeleton (frac_known); 'shl' measures it from the drawn
            # template's atoms (which are otherwise discarded, since num_known=0).
            # Fallback 'ntb' and all other constraints get None -> the radial
            # confinement is disabled downstream.
            used_real_template = (sc in ('alx', 'shl') and bool(extra_kwargs))
            if used_real_template:
                if sc == 'shl':
                    frac_np = np.asarray(extra_kwargs['frac_known'], dtype=float)
                else:
                    frac_np = sc_material.frac_known.detach().cpu().numpy()
                cell_np = sc_material.cell.detach().cpu().numpy()
                self.tube_bounds_list.append(estimate_template_bounds(
                    frac_np, cell_np, r_lo_percentile=self.r_lo_percentile))
            else:
                self.tube_bounds_list.append(None)

            self.num_known_list.append(num_known)
            self.num_atom_list.append(num_atom)
            self.frac_coords_list.append(sc_material.frac_coords)
            self.atom_types_list.append(sc_material.atom_types)
            self.lattice_list.append(sc_material.cell)
            self.mask_x_list.append(sc_material.mask_x)
            self.mask_t_list.append(sc_material.mask_t)
            self.mask_l_list.append(sc_material.mask_l)
            
    def generate_dataset(self):
        self.data_list = []
        for index in tqdm(range(self.total_num)):
            num_atom = np.round(self.num_atom_list[index]).astype(np.int64)
            num_known_ = np.round(self.num_known_list[index]).astype(np.int64)
            frac_coords_known_=self.frac_coords_list[index]
            lattice_known_=self.lattice_list[index].unsqueeze(0)
            atom_types_known_=self.atom_types_list[index]
            mask_x_, mask_l_, mask_t_ = [a[index] for a in [self.mask_x_list, self.mask_l_list, self.mask_t_list]]

            # Graph-level cylindrical-envelope metadata. Stored with a leading
            # batch dim of 1 so PyG concatenates them to (B, ...) exactly like
            # lattice_known; the sampler expands per-atom via batch.batch. When no
            # real template was pinned, is_alx=0 and r_max is a large sentinel so
            # the radial pull is a no-op (r > r_max is never true).
            bounds = self.tube_bounds_list[index]
            if bounds is not None:
                is_alx_, r_max_, axis_ = 1.0, float(bounds['r_max']), int(bounds['axis'])
                # Inner band edge: 'shl' confines all atoms into [r_min, r_max];
                # 'alx' leaves it 0 (one-sided envelope on decorators only).
                r_min_ = float(bounds['r_min']) if self.sc_list[index] == 'shl' else 0.0
                centroid_ = torch.as_tensor(bounds['centroid'], dtype=torch.float)
                a_hat_ = torch.as_tensor(bounds['a_hat'], dtype=torch.float)
                e1_ = torch.as_tensor(bounds['e1'], dtype=torch.float)
                e2_ = torch.as_tensor(bounds['e2'], dtype=torch.float)
            else:
                is_alx_, r_max_, r_min_, axis_ = 0.0, 1e6, 0.0, 0
                centroid_ = torch.zeros(3); a_hat_ = torch.zeros(3)
                e1_ = torch.zeros(3); e2_ = torch.zeros(3)

            data = Data(
                num_atoms=torch.LongTensor([num_atom]),
                num_nodes=num_atom,
                num_known=num_known_,
                frac_coords_known=frac_coords_known_,
                lattice_known=lattice_known_,
                atom_types_known=atom_types_known_,
                mask_x=mask_x_,
                mask_l=mask_l_,
                mask_t=mask_t_,
                is_alx=torch.tensor([is_alx_], dtype=torch.float),
                r_max=torch.tensor([r_max_], dtype=torch.float),
                r_min=torch.tensor([r_min_], dtype=torch.float),
                tube_axis=torch.tensor([axis_], dtype=torch.long),
                tube_centroid=centroid_.unsqueeze(0),
                tube_a_hat=a_hat_.unsqueeze(0),
                tube_e1=e1_.unsqueeze(0),
                tube_e2=e2_.unsqueeze(0),
            )
            if self.is_carbon:
                data.atom_types = torch.LongTensor([6] * num_atom)
            self.data_list.append(data) 


    def __len__(self):
        return self.total_num

    def __getitem__(self, index):
        return self.data_list[index]
    
def set_seeds(seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
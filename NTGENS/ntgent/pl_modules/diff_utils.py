import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps, beta_start, beta_end):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


# --- Progressive skeleton-pinning strength psi(t) -----------------------------
# For constrained (SCIGEN-style) sampling, the pinned skeleton can be introduced
# *gradually* over the reverse-diffusion loop instead of frozen from t=T. psi(t)
# is the pinning strength at timestep t: 0 -> skeleton has no influence (all
# atoms diffuse freely, matching the training distribution), 1 -> skeleton fully
# pinned. It ramps from psi_start at t=T to psi_end at t=0. The caller multiplies
# psi onto the binary mask each step (see sample_scigen), so psi==1 everywhere
# reproduces the original binary pinning exactly.

def pinning_strength_linear(t, T, psi_start=0.0, psi_end=1.0):
    """Linear ramp: psi = psi_end at t=0, psi_start at t=T."""
    frac = float(t) / float(T)              # 1.0 at t=T, 0.0 at t=0
    return psi_end + (psi_start - psi_end) * frac

def pinning_strength_sigmoid(t, T, alpha=10.0, t_mid_frac=0.6,
                             psi_start=0.0, psi_end=1.0):
    """Sigmoid ramp with inflection at t_mid_frac*T (slow-early / fast-mid /
    slow-late, mimicking nucleate -> grow -> anneal). alpha sets steepness."""
    t_mid = t_mid_frac * float(T)
    # s = 0 near t=T, s = 1 near t=0 (note the sign: t decreases over the loop).
    s = 1.0 / (1.0 + math.exp(alpha * (float(t) - t_mid) / float(T)))
    return psi_start + (psi_end - psi_start) * s

def pinning_strength(t, T, cfg):
    """Dispatch to the configured psi(t) schedule.

    cfg: dict-like with keys {enabled, schedule, alpha, t_mid_frac, psi_start,
    psi_end}. Returns a python float in [0, 1]. Returns 1.0 when cfg is None,
    disabled, or schedule == 'none' -> binary pinning (original behaviour)."""
    if not cfg or not cfg.get('enabled', False):
        return 1.0
    schedule = cfg.get('schedule', 'none')
    psi_start = cfg.get('psi_start', 0.0)
    psi_end = cfg.get('psi_end', 1.0)
    if schedule == 'linear':
        return pinning_strength_linear(t, T, psi_start, psi_end)
    if schedule == 'sigmoid':
        return pinning_strength_sigmoid(
            t, T, cfg.get('alpha', 10.0), cfg.get('t_mid_frac', 0.6),
            psi_start, psi_end)
    return 1.0


# --- Lattice-tracked tube frame -----------------------------------------------
# The reverse-diffusion loop starts from a lattice that is pure noise (l_T ~ N(0,1),
# a ~1 A cell) and only grows into the real template cell (~14 A) as C0 = sqrt(alphas_cumprod)
# rises. A tube frame frozen in absolute template Angstroms therefore sits far OUTSIDE
# the early cell: every atom ends up at nearly the same azimuth relative to a centroid
# that is several cell-widths away, and since the radial forces below are theta-invariant
# that single arc is then slid onto the wall and locked in. Rebuilding the frame from the
# CURRENT lattice each step keeps it self-consistent at every noise level; at t=0 the
# current lattice IS the template cell, so s_t -> 1 and this reduces exactly to the
# original fixed-frame behaviour.

# Row indices of the two non-axis (transverse) lattice vectors, keyed by axis index.
_OTHER_ROWS = torch.tensor([[1, 2], [0, 2], [0, 1]], dtype=torch.long)


def lattice_tube_frame(lat, axis, centroid_frac=None, eps=1e-8):
    """Per-graph transverse cylindrical frame reconstructed from the CURRENT lattice.

    Mirrors sc_utils.cylindrical_frame (Gram-Schmidt, so non-orthogonal cells are
    handled) but batched, on-device, and driven by the lattice at this diffusion
    step rather than the template's t=0 cell.

    lat: (B, 3, 3) row-vector lattices. axis: (B,) long, tube-axis lattice index
    (batch.tube_axis). centroid_frac: (B, 3) fractional centre of the tube cross
    section, or None -> the box transverse centre (0.5, 0.5 on the non-axis rows).

    Returns (a_hat, e1, e2, ctr, area, pn) with shapes (B,3), (B,3), (B,3), (B,3),
    (B,), (B,2): the axis unit vector, the transverse orthonormal basis, the
    cross-section centre (Cartesian, transverse), the transverse cell area, and the
    two transverse vector norms."""
    B = lat.shape[0]
    other = _OTHER_ROWS.to(lat.device)[axis]                   # (B, 2)
    gather = lambda rows: torch.gather(
        lat, 1, rows.view(B, 1, 1).expand(B, 1, 3)).squeeze(1)  # (B, 3)

    a = gather(axis)
    a_hat = a / a.norm(dim=-1, keepdim=True).clamp_min(eps)

    w0, w1 = gather(other[:, 0]), gather(other[:, 1])
    # Transverse components (project out the axial direction).
    p0 = w0 - (w0 * a_hat).sum(-1, keepdim=True) * a_hat
    p1 = w1 - (w1 * a_hat).sum(-1, keepdim=True) * a_hat

    e1 = p0 / p0.norm(dim=-1, keepdim=True).clamp_min(eps)
    e2 = torch.cross(a_hat, e1, dim=-1)                        # completes the frame

    if centroid_frac is None:
        f0 = f1 = torch.full((B, 1), 0.5, dtype=lat.dtype, device=lat.device)
    else:
        f0 = torch.gather(centroid_frac, 1, other[:, 0:1])     # (B, 1)
        f1 = torch.gather(centroid_frac, 1, other[:, 1:2])
    # The Cartesian centroid is sum_k f_k * lat[k]; transverse projection is linear
    # and kills the axial term, so only the two non-axis components contribute.
    ctr = f0 * p0 + f1 * p1

    area = (torch.cross(w0, w1, dim=-1) * a_hat).sum(-1).abs()  # (B,)
    pn = torch.stack([p0.norm(dim=-1), p1.norm(dim=-1)], dim=-1)  # (B, 2)
    return a_hat, e1, e2, ctr, area, pn


def transverse_scale(area, pn, area_ref, pn_ref, s_lo=0.25, s_hi=4.0,
                     ang_min=0.05, eps=1e-8):
    """Scale of the current transverse cell relative to the template's, plus a
    validity gate.

    Area-based (s = sqrt(A_t / A_ref)) so the band keeps the same fraction of the
    cross-section at every t -- i.e. r_hi_t / sqrt(A_t) is invariant. A pure area
    ratio can collapse toward 0 when the two transverse rows of a randn lattice come
    out near-parallel (~5% of draws have sin(angle) < 0.075), so it is clamped to a
    window around the never-degenerate linear ratio.

    Returns (s, ok): (B,) float scale and (B,) bool -- ok=False marks a cell so
    degenerate that "tube" is meaningless, and the caller should zero the guidance
    strength for that graph this step."""
    s_area = torch.sqrt((area / area_ref.clamp_min(eps)).clamp_min(0.0))
    s_lin = 0.5 * (pn[:, 0] / pn_ref[:, 0].clamp_min(eps)
                   + pn[:, 1] / pn_ref[:, 1].clamp_min(eps))
    s = torch.max(torch.min(s_area, s_hi * s_lin), s_lo * s_lin)
    s = s.clamp(1e-3, 1e3)
    sin_ang = area / (pn[:, 0] * pn[:, 1]).clamp_min(eps)
    ok = (sin_ang > ang_min) & (area > eps)
    return s, ok


def apply_radial_band(cart_xyz, r_lo, r_hi, centroid, a_hat, e1, e2, strength):
    """Softly confine atoms to a transverse radial band [r_lo, r_hi] (on device;
    used inside sample_scigen). Atoms with r > r_hi are pulled inward, r < r_lo are
    pushed outward, and atoms already inside the band are untouched. Purely
    geometric: only the transverse radius is rescaled; the axial coordinate and the
    angle are unchanged.

    The geometric-shell mode (sc='shl') confines ALL atoms into the wall band by
    passing the two-sided [r_lo, r_hi]; passing r_lo=0 degrades it to a one-sided
    ceiling (pull inward only) for any caller that wants that.

    cart_xyz: (M, 3) Cartesian coords. r_lo/r_hi: (M,) or scalar per-atom band
    edges (r_lo <= r_hi). centroid/a_hat/e1/e2: (M, 3) per-atom transverse frame
    (broadcast by caller). strength: (M,) or scalar in [0, 1] (0 = no pull, 1 = snap
    to the nearest edge). Returns new (M, 3) Cartesian coords."""
    z = (cart_xyz * a_hat).sum(-1, keepdim=True)          # (M, 1) axial
    perp = cart_xyz - z * a_hat                            # transverse component
    rel = perp - centroid
    u = (rel * e1).sum(-1)                                 # (M,)
    v = (rel * e2).sum(-1)
    r = torch.sqrt(u * u + v * v).clamp_min(1e-8)
    if not torch.is_tensor(strength):
        strength = torch.full_like(r, float(strength))
    # Blend the radius toward the nearest violated edge; leave in-band atoms alone.
    r_target = torch.where(r > r_hi, r + strength * (r_hi - r),
                           torch.where(r < r_lo, r + strength * (r_lo - r), r))
    scale = r_target / r                                   # (M,)
    u_new = (u * scale).unsqueeze(-1)
    v_new = (v * scale).unsqueeze(-1)
    perp_new = centroid + u_new * e1 + v_new * e2
    return perp_new + z * a_hat


def apply_density_force(cart_xyz, force_table, grid_lo, grid_dr, centroid, a_hat,
                        e1, e2, strength, r_lo=None, r_hi=None):
    """Log-density gradient guidance on the transverse radius (on device; v2).

    Steps each atom's radius up d/dr log rho(r), tabulated per atom in force_table
    (built at dataset time by sc_utils.estimate_radial_density_force). This nudges
    atoms toward the modes of the template's empirical radial distribution (the
    wall/shell), so they concentrate into a thin wall instead of filling the band.
    Purely radial: axial coordinate and angle are unchanged. Layered on top of the
    hard band (apply_radial_band); r_lo/r_hi (if given) clamp the result so guidance
    can never push an atom out of the confinement band.

    cart_xyz: (M, 3) Cartesian. force_table: (M, G) per-atom tabulated force. grid_lo
    /grid_dr: (M,) or scalar grid origin/spacing (r = grid_lo + i*grid_dr). frame
    (centroid/a_hat/e1/e2): (M, 3). strength: (M,) or scalar (= density_strength *
    psi * gate); 0 -> no-op. Returns new (M, 3) Cartesian coords."""
    z = (cart_xyz * a_hat).sum(-1, keepdim=True)          # (M, 1) axial
    perp = cart_xyz - z * a_hat
    rel = perp - centroid
    u = (rel * e1).sum(-1)                                 # (M,)
    v = (rel * e2).sum(-1)
    r = torch.sqrt(u * u + v * v).clamp_min(1e-8)

    # Linear interpolation of the per-atom force table at radius r.
    G = force_table.shape[-1]
    if not torch.is_tensor(grid_lo):
        grid_lo = torch.as_tensor(grid_lo, dtype=r.dtype, device=r.device)
    if not torch.is_tensor(grid_dr):
        grid_dr = torch.as_tensor(grid_dr, dtype=r.dtype, device=r.device)
    pos = (r - grid_lo) / grid_dr.clamp_min(1e-8)         # (M,) fractional bin
    pos = pos.clamp(0, G - 1)
    i0 = pos.floor().long()
    i1 = (i0 + 1).clamp(max=G - 1)
    w = (pos - i0.to(pos.dtype)).unsqueeze(-1)            # (M, 1)
    f0 = torch.gather(force_table, 1, i0.unsqueeze(-1))   # (M, 1)
    f1 = torch.gather(force_table, 1, i1.unsqueeze(-1))
    g = ((1 - w) * f0 + w * f1).squeeze(-1)              # (M,) interpolated force

    if not torch.is_tensor(strength):
        strength = torch.full_like(r, float(strength))
    r_new = r + strength * g
    if r_lo is not None and r_hi is not None:
        r_new = torch.max(torch.min(r_new, r_hi), r_lo)   # stay inside the band
    r_new = r_new.clamp_min(0.0)
    scale = r_new / r                                     # (M,)
    u_new = (u * scale).unsqueeze(-1)
    v_new = (v * scale).unsqueeze(-1)
    perp_new = centroid + u_new * e1 + v_new * e2
    return perp_new + z * a_hat


def apply_angular_spread(cart_xyz, centroid, a_hat, e1, e2, batch_idx, num_graphs,
                         strength, mode_weights=(1.0,), r_floors=(0.0,),
                         r_eps=1e-4, min_atoms=3, max_dtheta=0.5, jitter=0.0,
                         generator=None):
    """Rotate atoms about the tube axis to spread them in theta (on device).

    Complements apply_radial_band / apply_density_force, which are theta-INVARIANT by
    construction (they only rescale the radius), so any azimuthal clustering they
    inherit is preserved forever. This term descends the circular-uniformity energy
    L = sum_m w_m * Rbar_m^2, where Z_m = sum_j exp(i*m*theta_j) and Rbar_m = |Z_m|/N
    (0 = uniform, 1 = every atom at one angle):

        dtheta_i = strength * sum_m w_m * relu(Rbar_m - rho_m) * sin(m*theta_i - arg Z_m)

    The `+` sign pushes atoms AWAY from the resultant angle. Self-limiting (the drive
    is proportional to Rbar_m, so it vanishes at uniformity), rotation-invariant (only
    theta_i - arg Z_m appears, so a frame whose e1/e2 rotate step to step is harmless),
    and O(M*N). Note this is exactly a band-limited pairwise angular repulsion --
    sum_m w_m Rbar_m^2 = (1/N^2) sum_ij K(theta_i - theta_j) with K(phi) = sum_m w_m
    cos(m*phi) -- evaluated in O(M*N) instead of O(N^2).

    Each atom's transverse radius r and axial coordinate z are preserved EXACTLY: the
    update is applied as a displacement in the (e1, e2) plane, so strength=0 returns
    the input unchanged bit for bit.

    Mode 1 alone is the right default: real templates measure Rbar_1 ~ 0.01 (median),
    so driving it to 0 matches ground truth, whereas Rbar_2 ~ 0.10 is genuine n-fold
    structure that must NOT be optimised away -- enable mode 2 only with a low weight
    and a deadband (r_floors). Two known fixed points: EXACTLY co-located atoms
    (every sin term is 0 -- use `jitter`), and two antipodal clusters, where mode 1
    already sees an optimum (Rbar_1 ~ 0 while Rbar_2 ~ 1). Measured on finite-width
    clusters: mode 1 alone leaves Rbar_2 = 0.96 / gap = 160 deg, while adding mode 2
    at w=0.25 recovers Rbar_2 = 0.10 / gap = 66 deg, stopping exactly at the deadband.
    On the single-arc case this term actually targets, mode 1 alone is enough
    (Rbar_1 0.98 -> 0.00, largest angular gap 322 -> 88 deg).

    cart_xyz: (M, 3) Cartesian. centroid/a_hat/e1/e2: (M, 3) per-atom frame.
    batch_idx: (M,) long graph id (batch.batch). num_graphs: B.
    strength: (M,) or scalar = ang_strength * psi * gate; 0 -> exact no-op.
    mode_weights/r_floors: per-Fourier-mode weight w_m and deadband rho_m.
    Returns new (M, 3) Cartesian coords."""
    z = (cart_xyz * a_hat).sum(-1, keepdim=True)          # (M, 1) axial
    perp = cart_xyz - z * a_hat
    rel = perp - centroid
    u = (rel * e1).sum(-1)                                 # (M,)
    v = (rel * e2).sum(-1)
    r = torch.sqrt(u * u + v * v)

    if not torch.is_tensor(strength):
        strength = torch.full_like(r, float(strength))

    # theta is undefined on the axis; those atoms neither vote nor rotate.
    valid = (r > r_eps).to(r.dtype)
    r_safe = r.clamp_min(1e-12)
    c1, s1 = u / r_safe, v / r_safe                        # cos(theta), sin(theta)

    zeros = lambda: torch.zeros(num_graphs, dtype=r.dtype, device=r.device)
    n_eff = zeros().index_add_(0, batch_idx, valid)        # (B,)

    dtheta = torch.zeros_like(r)
    cm, sm = c1, s1
    for m, (w, floor) in enumerate(zip(mode_weights, r_floors), start=1):
        if m > 1:                                          # Chebyshev: no atan2 needed
            cm, sm = cm * c1 - sm * s1, sm * c1 + cm * s1
        C = zeros().index_add_(0, batch_idx, cm * valid)
        S = zeros().index_add_(0, batch_idx, sm * valid)
        z_mod = torch.sqrt(C * C + S * S).clamp_min(1e-12)  # |Z_m|
        r_bar = z_mod / n_eff.clamp_min(1.0)
        drive = float(w) * torch.relu(r_bar - float(floor))  # (B,), 0 at uniformity
        # sin(m*theta_i - arg Z_m) = (sin(m*th_i)*C - cos(m*th_i)*S) / |Z_m|
        sin_term = (sm * C[batch_idx] - cm * S[batch_idx]) / z_mod[batch_idx]
        dtheta = dtheta + drive[batch_idx] * sin_term

    dtheta = dtheta * strength
    if jitter:
        noise = torch.randn(dtheta.shape, dtype=dtheta.dtype, device=dtheta.device,
                            generator=generator)
        dtheta = dtheta + float(jitter) * strength * noise
    dtheta = dtheta.clamp(-float(max_dtheta), float(max_dtheta))
    # Too few atoms to define an angular distribution -> leave the graph alone.
    dtheta = dtheta * valid * (n_eff >= float(min_atoms)).to(r.dtype)[batch_idx]

    cd, sd = torch.cos(dtheta), torch.sin(dtheta)
    du = (u * cd - v * sd) - u                             # exactly 0 when dtheta == 0
    dv = (u * sd + v * cd) - v
    return cart_xyz + du.unsqueeze(-1) * e1 + dv.unsqueeze(-1) * e2


def p_wrapped_normal(x, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += torch.exp(-(x + T * i) ** 2 / 2 / sigma ** 2)
    return p_

def d_log_p_wrapped_normal(x, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += (x + T * i) / sigma ** 2 * torch.exp(-(x + T * i) ** 2 / 2 / sigma ** 2)
    return p_ / p_wrapped_normal(x, sigma, N, T)

def sigma_norm(sigma, T=1.0, sn = 10000):
    sigmas = sigma[None, :].repeat(sn, 1)
    x_sample = sigma * torch.randn_like(sigmas)
    x_sample = x_sample % T
    normal_ = d_log_p_wrapped_normal(x_sample, sigmas, T = T)
    return (normal_ ** 2).mean(dim = 0)




class BetaScheduler(nn.Module):

    def __init__(
        self,
        timesteps,
        scheduler_mode,
        beta_start = 0.0001,
        beta_end = 0.02
    ):
        super(BetaScheduler, self).__init__()
        self.timesteps = timesteps
        if scheduler_mode == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif scheduler_mode == 'linear':
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif scheduler_mode == 'quadratic':
            betas = quadratic_beta_schedule(timesteps, beta_start, beta_end)
        elif scheduler_mode == 'sigmoid':
            betas = sigmoid_beta_schedule(timesteps, beta_start, beta_end)


        betas = torch.cat([torch.zeros([1]), betas], dim=0)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        sigmas = torch.zeros_like(betas)

        sigmas[1:] = betas[1:] * (1. - alphas_cumprod[:-1]) / (1. - alphas_cumprod[1:])

        sigmas = torch.sqrt(sigmas)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sigmas', sigmas)

    def uniform_sample_t(self, batch_size, device):
        ts = np.random.choice(np.arange(1, self.timesteps+1), batch_size)
        return torch.from_numpy(ts).to(device)

class SigmaScheduler(nn.Module):

    def __init__(
        self,
        timesteps,
        sigma_begin = 0.01,
        sigma_end = 1.0
    ):
        super(SigmaScheduler, self).__init__()
        self.timesteps = timesteps
        self.sigma_begin = sigma_begin
        self.sigma_end = sigma_end
        sigmas = torch.FloatTensor(np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), timesteps)))

        sigmas_norm_ = sigma_norm(sigmas)

        self.register_buffer('sigmas', torch.cat([torch.zeros([1]), sigmas], dim=0))
        self.register_buffer('sigmas_norm', torch.cat([torch.ones([1]), sigmas_norm_], dim=0))

    def uniform_sample_t(self, batch_size, device):
        ts = np.random.choice(np.arange(1, self.timesteps+1), batch_size)
        return torch.from_numpy(ts).to(device)



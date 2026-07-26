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


def apply_radial_band(cart_xyz, r_lo, r_hi, centroid, a_hat, e1, e2, strength):
    """Softly confine atoms to a transverse radial band [r_lo, r_hi] (on device;
    used inside sample_scigen). Atoms with r > r_hi are pulled inward, r < r_lo are
    pushed outward, and atoms already inside the band are untouched. Purely
    geometric: only the transverse radius is rescaled; the axial coordinate and the
    angle are unchanged.

    This generalizes a one-sided ceiling to a two-sided shell: the geometric-shell
    mode (sc='shl') confines ALL atoms into the wall band, while the sc='alx'
    envelope passes r_lo=0 so it only ever pulls decorators inward (never pushes an
    atom out of a filled core).

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


def apply_radial_pull(cart_xyz, r_max, centroid, a_hat, e1, e2, strength):
    """One-sided radial ceiling: pull atoms with r > r_max back toward r_max, leave
    r <= r_max untouched. Thin wrapper over apply_radial_band with r_lo=0 (kept for
    the sc='alx' envelope and any existing callers)."""
    r_lo = torch.zeros_like(r_max) if torch.is_tensor(r_max) else 0.0
    return apply_radial_band(cart_xyz, r_lo, r_max, centroid, a_hat, e1, e2, strength)


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



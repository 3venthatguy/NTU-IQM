import math, copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from typing import Any, Dict

import hydra
import omegaconf
import pytorch_lightning as pl
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from tqdm import tqdm

from scigen.common.utils import PROJECT_ROOT
from scigen.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume, lattice_params_to_matrix_torch,
    frac_to_cart_coords, min_distance_sqr_pbc)

from scigen.pl_modules.diff_utils import (d_log_p_wrapped_normal, pinning_strength,
                                          apply_radial_band, apply_density_force,
                                          apply_angular_spread,
                                          lattice_tube_frame, transverse_scale)


MAX_ATOMIC_NUM=100


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()
        if hasattr(self.hparams, "model"):
            self._hparams = self.hparams.model


    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}


### Model definition

class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CSPDiffusion(BaseModule):   
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.decoder = hydra.utils.instantiate(self.hparams.decoder, latent_dim = self.hparams.latent_dim + self.hparams.time_dim, pred_type = True, smooth = True)
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)
        self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.keep_lattice = self.hparams.cost_lattice < 1e-5
        self.keep_coords = self.hparams.cost_coord < 1e-5



    def forward(self, batch):

        batch_size = batch.num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        beta = self.beta_scheduler.betas[times]

        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords

        rand_l, rand_x = torch.randn_like(lattices), torch.randn_like(frac_coords)

        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l
        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.

        gt_atom_types_onehot = F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()

        rand_t = torch.randn_like(gt_atom_types_onehot)

        atom_type_probs = (c0.repeat_interleave(batch.num_atoms)[:, None] * gt_atom_types_onehot + c1.repeat_interleave(batch.num_atoms)[:, None] * rand_t)

        if self.keep_coords:
            input_frac_coords = frac_coords

        if self.keep_lattice:
            input_lattice = lattices

        pred_l, pred_x, pred_t = self.decoder(time_emb, atom_type_probs, input_frac_coords, input_lattice, batch.num_atoms, batch.batch)

        tar_x = d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)

        loss_lattice = F.mse_loss(pred_l, rand_l)
        loss_coord = F.mse_loss(pred_x, tar_x)
        loss_type = F.mse_loss(pred_t, rand_t)


        loss = (
            self.hparams.cost_lattice * loss_lattice +
            self.hparams.cost_coord * loss_coord + 
            self.hparams.cost_type * loss_type)

        return {
            'loss' : loss,
            'loss_lattice' : loss_lattice,
            'loss_coord' : loss_coord,
            'loss_type' : loss_type
        }

    @torch.no_grad()
    def sample(self, batch, diff_ratio = 1.0, step_lr = 1e-5):
        pass


    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_type = output_dict['loss_type']
        loss = output_dict['loss']


        self.log_dict(
            {'train_loss': loss,
            'lattice_loss': loss_lattice,
            'coord_loss': loss_coord,
            'type_loss': loss_type},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        if loss.isnan():
            return None

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='val')

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, output_dict, prefix):

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_type = output_dict['loss_type']
        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_lattice_loss': loss_lattice,
            f'{prefix}_coord_loss': loss_coord,
            f'{prefix}_type_loss': loss_type,
        }

        return log_dict, loss



@torch.no_grad()
def sample_scigen(self, batch, diff_ratio = 1.0, step_lr = 1e-5):     

    batch_size = batch.num_graphs

    l_T_unk, x_T_unk = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)    

    t_T_unk = torch.randn([batch.num_nodes, MAX_ATOMIC_NUM]).to(self.device)    

    l_T_known, x_T_known = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)       
    t_T_known = torch.randn([batch.num_nodes, MAX_ATOMIC_NUM]).to(self.device)    
    
    l_0_known, x_0_known = batch.lattice_known, batch.frac_coords_known
    t_0_known = F.one_hot(batch.atom_types_known - 1, num_classes=MAX_ATOMIC_NUM).float()
    
    mask_l, mask_x, mask_t = batch.mask_l, batch.mask_x, batch.mask_t   
    mask_t = mask_t.unsqueeze(-1)
    if len(mask_l.shape) == 2:
        print('Use reduce mask')
        mask_x, mask_l = mask_x.unsqueeze(-1), mask_l.unsqueeze(-1)
    mask_x_inv, mask_l_inv, mask_t_inv = 1 - mask_x, 1 - mask_l, 1 - mask_t

    # --- Progressive skeleton pinning + cylindrical radial envelope -----------
    # Both are opt-in via runtime attributes set on the model instance right after
    # the sample_scigen monkey-patch (model.pin_cfg / model.cyl_cfg). When absent
    # or disabled, psi(t)==1 at every step and the radial pull is a no-op, so this
    # block reproduces the original binary-pinning sampler exactly.
    pin_cfg = getattr(self, 'pin_cfg', None)
    cyl_cfg = getattr(self, 'cyl_cfg', None)
    dens_cfg = getattr(self, 'dens_cfg', None)
    ang_cfg = getattr(self, 'ang_cfg', None)
    # psi(t) does double duty: it blends the pinned lattice/skeleton AND scales the
    # geometric guidance. Those want different schedules -- a hard-pinned lattice
    # (psi=1 from t=T) gives the cleanest monotone cell-scale ramp, but running the
    # guidance at full strength against a noise-scale cell wrecks the structure. An
    # optional geom_pin_cfg ramps the forces independently; absent, behaviour is
    # unchanged.
    geom_cfg = getattr(self, 'geom_pin_cfg', None) or pin_cfg
    T_total = self.beta_scheduler.timesteps

    cyl_on = bool(cyl_cfg and cyl_cfg.get('enabled', False)) and hasattr(batch, 'is_alx')
    # v2: radial log-density guidance (concentrate atoms onto the wall within the band).
    dens_on = bool(dens_cfg and dens_cfg.get('enabled', False)) and hasattr(batch, 'dens_force')
    # v3: angular dispersion (the radial terms above are theta-invariant, so nothing
    # else ever redistributes atoms around the circumference).
    ang_on = bool(ang_cfg and ang_cfg.get('enabled', False)) and hasattr(batch, 'is_alx')
    geom_on = cyl_on or dens_on or ang_on
    if geom_on:
        node_batch = batch.batch
        pull_gate = batch.is_alx[node_batch]                      # (N,) 1 for tube graphs (legacy flag name)
        r_hi_atom = batch.r_max[node_batch] * (1.0 + float((cyl_cfg or {}).get('margin', 0.0)))
        # Inner band edge: r_min for the 'shl' shell (two-sided, confines every atom
        # into the wall). Older batches without r_min fall back to a one-sided
        # ceiling (r_lo=0).
        r_lo_atom = (batch.r_min[node_batch] if hasattr(batch, 'r_min')
                     else torch.zeros_like(r_hi_atom))
        centroid_atom = batch.tube_centroid[node_batch]           # (N, 3)
        a_hat_atom = batch.tube_a_hat[node_batch]
        e1_atom = batch.tube_e1[node_batch]
        e2_atom = batch.tube_e2[node_batch]
        max_strength = float((cyl_cfg or {}).get('max_strength', 1.0))
        # Rebuild the tube frame from the CURRENT lattice each step instead of
        # freezing the template's t=0 frame. The cell starts as noise (~1 A) and
        # only grows into the template cell, so a fixed absolute-Angstrom frame sits
        # outside the early cell and collapses every atom onto one azimuth; since
        # the radial forces are theta-invariant, that arc is then locked in.
        track_frame = bool((cyl_cfg or {}).get('track_frame', True))
        tube_axis = getattr(batch, 'tube_axis', None)
        if tube_axis is None:
            tube_axis = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        ctr_frac = getattr(batch, 'tube_centroid_frac', None)   # None -> box centre
        s_lo_cfg = float((cyl_cfg or {}).get('scale_lo', 0.25))
        s_hi_cfg = float((cyl_cfg or {}).get('scale_hi', 4.0))
        ang_min_cfg = float((cyl_cfg or {}).get('ang_min', 0.05))
        if track_frame:
            # Reference scale from the template cell; s_t == 1 there by construction,
            # so at t=0 this reduces exactly to the original fixed-frame behaviour.
            _, _, _, _, area_ref, pn_ref = lattice_tube_frame(
                batch.lattice_known, tube_axis, ctr_frac)
    if dens_on:
        dens_force_atom = batch.dens_force[node_batch]            # (N, G)
        dens_grid_lo_atom = batch.dens_grid_lo[node_batch]        # (N,)
        dens_grid_dr_atom = batch.dens_grid_dr[node_batch]
        density_strength = float(dens_cfg.get('density_strength', 0.05))
    if ang_on:
        ang_strength = float(ang_cfg.get('ang_strength', 0.05))
        ang_modes = tuple(ang_cfg.get('mode_weights', (1.0,)))
        ang_floors = tuple(ang_cfg.get('r_floors', (0.0,) * len(ang_modes)))
        ang_min_atoms = int(ang_cfg.get('min_atoms', 3))
        ang_max_dtheta = float(ang_cfg.get('max_dtheta', 0.5))
        ang_jitter = float(ang_cfg.get('jitter', 0.0))

    def radial_envelope(x_frac, lat, psi):
        """Shape tube-graph atoms transversely. (1) Hard band [r_lo, r_hi] confines
        them (v1); (2) log-density guidance nudges them up d/dr log rho(r) toward the
        template's wall (v2), concentrating a filled disk into a thin wall. Both
        scale with psi so early (exploratory) steps are free and late steps enforce
        the shape. No-op unless a mode is on.

        With track_frame (default) the (r, theta, z) frame is rebuilt from the
        CURRENT lattice each step and the band / density grid are scaled by the
        transverse cell scale s_t, so the constraint stays self-consistent while the
        cell grows from noise into the template cell. s_t -> 1 at t=0, where this is
        exactly the legacy fixed-frame behaviour."""
        if not geom_on:
            return x_frac
        cart = frac_to_cart_coords(x_frac, None, None, batch.num_atoms, lattices=lat)
        cart_in = cart
        ctr_a, ah_a, e1_a, e2_a = centroid_atom, a_hat_atom, e1_atom, e2_atom
        r_lo_a, r_hi_a, gate, s_atom = r_lo_atom, r_hi_atom, pull_gate, None
        gl_a = dens_grid_lo_atom if dens_on else None
        gd_a = dens_grid_dr_atom if dens_on else None
        if track_frame:
            a_h, e1_, e2_, ctr_, area, pn = lattice_tube_frame(lat, tube_axis, ctr_frac)
            s, ok = transverse_scale(area, pn, area_ref, pn_ref,
                                     s_lo_cfg, s_hi_cfg, ang_min_cfg)
            rep = lambda x: torch.repeat_interleave(x, batch.num_atoms, dim=0)
            ah_a, e1_a, e2_a, ctr_a = rep(a_h), rep(e1_), rep(e2_), rep(ctr_)
            s_atom = rep(s)                                      # (N,)
            # Degenerate cells (near-parallel transverse rows) carry no tube frame.
            gate = pull_gate * rep(ok.to(pull_gate.dtype))
            r_lo_a, r_hi_a = r_lo_atom * s_atom, r_hi_atom * s_atom
            if dens_on:
                gl_a, gd_a = dens_grid_lo_atom * s_atom, dens_grid_dr_atom * s_atom
        if cyl_on:
            strength = (max_strength * psi) * gate               # (N,)
            cart = apply_radial_band(cart, r_lo_a, r_hi_a, ctr_a,
                                     ah_a, e1_a, e2_a, strength)
        if dens_on:
            d_strength = (density_strength * psi) * gate         # (N,)
            if s_atom is not None:
                # r -> s*r means the displacement must scale too, else the force is
                # 1/s_t too strong while the cell is still small.
                d_strength = d_strength * s_atom
            cart = apply_density_force(cart, dens_force_atom, gl_a, gd_a,
                                       ctr_a, ah_a, e1_a, e2_a, d_strength,
                                       r_lo=r_lo_a, r_hi=r_hi_a)
        if ang_on:
            # Angle-only, so it commutes with the radial terms above; applied last
            # so it acts on the radii they just settled.
            a_strength = (ang_strength * psi) * gate         # (N,)
            cart = apply_angular_spread(cart, ctr_a, ah_a, e1_a, e2_a,
                                        node_batch, batch_size, a_strength,
                                        mode_weights=ang_modes, r_floors=ang_floors,
                                        min_atoms=ang_min_atoms,
                                        max_dtheta=ang_max_dtheta, jitter=ang_jitter)
        # Graphs with no tube (is_alx=0) carry a zero frame, which would otherwise
        # collapse every one of their atoms onto the origin; leave them untouched.
        cart = torch.where((gate > 0).unsqueeze(-1), cart, cart_in)
        inv_nodes = torch.repeat_interleave(torch.linalg.pinv(lat), batch.num_atoms, dim=0)
        return torch.einsum('ni,nij->nj', cart, inv_nodes) % 1.0

    print('num_atoms: ', batch.num_atoms)
    print('mask_x, x_T_known, x_T_unk: ', mask_x.shape, x_T_known.shape, x_T_unk.shape) 
    print('mask_l, l_T_known, l_T_unk: ', mask_l.shape, l_T_known.shape, l_T_unk.shape)
    print('mask_t, t_T_known, t_T_unk: ', mask_t.shape, t_T_known.shape, t_T_unk.shape)

    # At t=T the pinning strength is psi_start (~0 for a ramp), so known atoms
    # start from noise too -> the whole batch begins in-distribution.
    psi_T = pinning_strength(self.beta_scheduler.timesteps, T_total, pin_cfg)
    x_T = (psi_T * mask_x) * x_T_known + (1 - psi_T * mask_x) * x_T_unk
    l_T = (psi_T * mask_l) * l_T_known + (1 - psi_T * mask_l) * l_T_unk
    t_T = (psi_T * mask_t) * t_T_known + (1 - psi_T * mask_t) * t_T_unk

        
    if self.keep_coords:
        x_T = batch.frac_coords

    if self.keep_lattice:
        l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
    

    traj = {self.beta_scheduler.timesteps : {
        'num_atoms' : batch.num_atoms,
        'atom_types' : t_T,
        'frac_coords' : x_T % 1.,
        'lattices' : l_T
    }}

    for t in tqdm(range(self.beta_scheduler.timesteps, 0, -1)):

        times = torch.full((batch_size, ), t, device = self.device)

        time_emb = self.time_embedding(times)

        # Pinning strength at this step and the psi-scaled effective masks. psi==1
        # (schedule disabled) recovers the original binary masks exactly.
        psi = pinning_strength(t, T_total, pin_cfg)
        psi_geom = pinning_strength(t, T_total, geom_cfg)   # == psi unless decoupled
        mask_x_e, mask_l_e, mask_t_e = psi * mask_x, psi * mask_l, psi * mask_t
        mask_x_ei, mask_l_ei, mask_t_ei = 1 - mask_x_e, 1 - mask_l_e, 1 - mask_t_e
        
        alphas = self.beta_scheduler.alphas[t]
        alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

        sigmas = self.beta_scheduler.sigmas[t]
        sigma_x = self.sigma_scheduler.sigmas[t]
        sigma_norm = self.sigma_scheduler.sigmas_norm[t]

        c0 = 1.0 / torch.sqrt(alphas)
        c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)
        
        C0_minus_0 = torch.sqrt(alphas_cumprod) 
        C1_minus_0 = torch.sqrt(1. - alphas_cumprod)    
        sigma_x_minus_0 = sigma_x    
        alphas_cumprod_minus_1 = self.beta_scheduler.alphas_cumprod[t-1]
        C0_minus_1 = torch.sqrt(alphas_cumprod_minus_1) 
        C1_minus_1 = torch.sqrt(1. - alphas_cumprod_minus_1)    
        sigma_x_minus_1 = self.sigma_scheduler.sigmas[t-1]    

        x_t_unk = traj[t]['frac_coords']   
        l_t_unk = traj[t]['lattices']   
        t_t_unk = traj[t]['atom_types']   

        if self.keep_coords:
            x_t = x_T

        if self.keep_lattice:
            l_t = l_T

        # Corrector

        rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
        rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
        rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

        step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
        std_x = torch.sqrt(2 * step_size)

        pred_l, pred_x, pred_t = self.decoder(time_emb, t_t_unk, x_t_unk, l_t_unk, batch.num_atoms, batch.batch)    

        pred_x = pred_x * torch.sqrt(sigma_norm)

        x_t_minus_05_unk = x_t_unk - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t   

        l_t_minus_05_unk = l_t_unk   

        t_t_minus_05_unk = t_t_unk   

        rand_l_known = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)    
        rand_t_known = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)    
        rand_x_known = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)   
        
        x_t_minus_05_known = (x_0_known + sigma_x_minus_0*rand_x_known)%1   
        l_t_minus_05_known = C0_minus_0 * l_0_known + C1_minus_0 * rand_l_known   
        t_t_minus_05_known = C0_minus_0 * t_0_known + C1_minus_0 * rand_t_known   

        x_t_minus_05 = mask_x_e * x_t_minus_05_known + mask_x_ei * x_t_minus_05_unk
        l_t_minus_05 = mask_l_e * l_t_minus_05_known + mask_l_ei * l_t_minus_05_unk
        t_t_minus_05 = mask_t_e * t_t_minus_05_known + mask_t_ei * t_t_minus_05_unk
        x_t_minus_05 = radial_envelope(x_t_minus_05, l_t_minus_05, psi_geom)
        
    
        # Predictor

        rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
        rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
        rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

        adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
        step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
        std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   


        pred_l, pred_x, pred_t = self.decoder(time_emb, t_t_minus_05, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)

        pred_x = pred_x * torch.sqrt(sigma_norm)

        x_t_minus_1_unk = x_t_minus_05 - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t   

        l_t_minus_1_unk = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t   

        t_t_minus_1_unk = c0 * (t_t_minus_05 - c1 * pred_t) + sigmas * rand_t   

        rand_l_known = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)    
        rand_t_known = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)    
        rand_x_known = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)    
        
        x_t_minus_1_known = (x_0_known + sigma_x_minus_1*rand_x_known)%1    
        l_t_minus_1_known = C0_minus_1 * l_0_known + C1_minus_1 * rand_l_known   
        t_t_minus_1_known = C0_minus_1 * t_0_known + C1_minus_1 * rand_t_known   

        x_t_minus_1 = mask_x_e * x_t_minus_1_known + mask_x_ei * x_t_minus_1_unk
        l_t_minus_1 = mask_l_e * l_t_minus_1_known + mask_l_ei * l_t_minus_1_unk
        t_t_minus_1 = mask_t_e * t_t_minus_1_known + mask_t_ei * t_t_minus_1_unk
        x_t_minus_1 = radial_envelope(x_t_minus_1, l_t_minus_1, psi_geom)

        traj[t - 1] = {
            'num_atoms' : batch.num_atoms,
            'num_known' : torch.LongTensor([batch.num_known]),  
            'atom_types' : t_t_minus_1,
            'frac_coords' : x_t_minus_1 % 1.,
            'lattices' : l_t_minus_1              
        }

    traj_stack = {
        'num_atoms' : batch.num_atoms,
        'num_known' : torch.LongTensor([batch.num_known]),  
        'atom_types' : torch.stack([traj[i]['atom_types'] for i in range(self.beta_scheduler.timesteps, -1, -1)]).argmax(dim=-1) + 1,
        'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(self.beta_scheduler.timesteps, -1, -1)]),
        'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(self.beta_scheduler.timesteps, -1, -1)])
    }

    return traj[0], traj_stack

#%%
import time
import argparse
import torch
from pathlib import Path
from torch_geometric.data import DataLoader
from eval_utils import load_model, lattices_to_params_shape, recommand_step_lr
import random   
from scigen.pl_modules.diffusion_w_type import sample_scigen
from gen_utils import SampleDataset, convert_seconds_short, parse_none_or_value
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   


def diffusion(loader, model, step_lr, save_traj):
    
    frac_coords = []
    num_atoms = []
    num_known = []  
    atom_types = []
    lattices = []
    all_frac_coords = []
    all_atom_types = []
    all_lattices = []
    all_lengths, all_angles = [], []    # ignore if we save traj
    for idx, batch in enumerate(loader):

        if torch.cuda.is_available():
            batch.cuda()
        outputs, traj = model.sample_scigen(batch, step_lr = step_lr)
        frac_coords.append(outputs['frac_coords'].detach().cpu())
        num_atoms.append(outputs['num_atoms'].detach().cpu())
        num_known.append(outputs['num_known'].detach().cpu()) 
        atom_types.append(outputs['atom_types'].detach().cpu())
        lattices.append(outputs['lattices'].detach().cpu())
        if save_traj:
            all_frac_coords.append(traj['all_frac_coords'].detach().cpu())
            all_atom_types.append(traj['atom_types'].detach().cpu())
            all_lattices.append(traj['all_lattices'].detach().cpu())
        print(f'batch {idx+1}/{len(loader)} done') 

    frac_coords = torch.cat(frac_coords, dim=0)
    num_atoms = torch.cat(num_atoms, dim=0)
    num_known = torch.cat(num_known, dim=0)
    atom_types = torch.cat(atom_types, dim=0)
    lattices = torch.cat(lattices, dim=0)
    lengths, angles = lattices_to_params_shape(lattices)
    if save_traj: 
        all_frac_coords = torch.cat(all_frac_coords, dim=1)
        all_atom_types = torch.cat(all_atom_types, dim=1)
        all_lattices = torch.cat(all_lattices, dim=1)
        all_lengths, all_angles = lattices_to_params_shape(all_lattices)    # works for all-time outputs

    print('num_atoms: ', num_atoms.shape, num_atoms)
    print('num_known: ', num_known.shape, num_known)
    
    return (
        frac_coords, atom_types, lattices, lengths, angles, num_atoms, num_known, all_frac_coords, all_atom_types, all_lattices, all_lengths, all_angles 
    )



def main(args):
    # randomly generate seed for each run
    # Generate a random seed by combining time and randomness
    seed = int(time.time() * random.random())
    print(f"Generated seed: {seed}")
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    print('args: ', args)   
    model, _, cfg = load_model(
        model_path, load_data=False)
    if torch.cuda.is_available():
        model.to('cuda')
    model.sample_scigen = sample_scigen.__get__(model)
    # Constrained-sampling controls read by sample_scigen off these runtime attrs
    # (a pretrained checkpoint's self.hparams are frozen, so we don't rely on them).
    # Defaults reproduce the original binary-pinning behaviour.
    model.pin_cfg = {
        'enabled': args.pin_schedule != 'none',
        'schedule': args.pin_schedule,
        'alpha': args.pin_alpha,
        't_mid_frac': args.pin_tmid,
        'psi_start': args.pin_psi_start,
        'psi_end': args.pin_psi_end,
    }
    model.cyl_cfg = {
        'enabled': args.cyl_masking,
        'margin': args.cyl_margin,
        'r_lo_percentile': args.cyl_r_lo_pct,
        'max_strength': args.cyl_strength,
    }
    # v2 radial density guidance (concentrate atoms onto the wall within the band).
    model.dens_cfg = {
        'enabled': args.density_guidance,
        'density_strength': args.density_strength,
        'bandwidth_scale': args.bandwidth_scale,
        'grid_size': args.density_grid_size,
    }
    print('pin_cfg:', model.pin_cfg)
    print('cyl_cfg:', model.cyl_cfg)
    print('dens_cfg:', model.dens_cfg)
    print('Evaluate the diffusion model.')
    c_vec_cons = {'scale': args.c_scale, 'vert': args.c_vert}
    # print('c_vec_cons: ', c_vec_cons)
    test_set = SampleDataset(dataset=args.dataset, 
                            natm_range=args.natm_range,
                            total_num=args.batch_size * args.num_batches_to_samples, 
                            bond_sigma_per_mu=args.bond_sigma_per_mu,
                            use_min_bond_len=args.use_min_bond_len,
                            known_species=args.known_species, 
                            sc_list=args.sc, 
                            frac_z=args.frac_z,
                            c_vec_cons=c_vec_cons,
                            reduced_mask=args.reduced_mask,
                            seed = seed,
                            device=device,
                            max_decorators=args.max_decorators,
                            template_source=args.template_source,
                            r_lo_percentile=args.cyl_r_lo_pct,
                            bandwidth_scale=args.bandwidth_scale,
                            density_grid_size=args.density_grid_size)
    test_loader = DataLoader(test_set, batch_size = args.batch_size)
    step_lr = args.step_lr if args.step_lr >= 0 else recommand_step_lr['gen'][args.dataset]
    start_time = time.time()
    (frac_coords, atom_types, lattices, lengths, angles, num_atoms, num_known, \
        all_frac_coords, all_atom_types, all_lattices, all_lengths, all_angles) = diffusion(test_loader, model, step_lr, args.save_traj)    

    if args.label == '':
        gen_out_name = 'eval_gen.pt'
    else:
        gen_out_name = f'eval_gen_{args.label}.pt'
    print(f'gen_out_name: {gen_out_name}')

    run_time = time.time() - start_time
    total_num = args.batch_size * args.num_batches_to_samples
    print('args: ', args)   
    print(f'Total outputs: {args.num_batches_to_samples} samples x {args.batch_size} batches = {total_num} materials')
    print(f'run time: {run_time} sec = {convert_seconds_short(run_time)}')
    print(f'{run_time/args.num_batches_to_samples} sec/sample')
    print(f'{run_time/total_num} sec/material')
    
    config = {
        'eval_setting': args,
        'num_atoms': num_atoms,
        'num_known': num_known,
        'frac_coords': frac_coords,
        'atom_types': atom_types,
        'lengths': lengths,
        'angles': angles,
        'all_frac_coords': all_frac_coords,
        'all_atom_types': all_atom_types,
        'all_lengths': all_lengths,
        'all_angles': all_angles,
        'c_vec_cons': c_vec_cons,
        'pin_cfg': model.pin_cfg,
        'cyl_cfg': model.cyl_cfg,
        'dens_cfg': model.dens_cfg,
        'seed': seed,
        'time': run_time,
    }
    torch.save(config, model_path / gen_out_name)

      
#%%

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--step_lr', default=-1, type=float)
    parser.add_argument('--num_batches_to_samples', default=20, type=int)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--label', default='')
    parser.add_argument('--save_traj', type=lambda x: x.lower() == 'true', default=False, help="Save trajectory during generation")
    parser.add_argument('--natm_range', nargs='+', default=[1, 20])
    parser.add_argument('--bond_sigma_per_mu', default=None)   
    parser.add_argument('--use_min_bond_len', type=lambda x: x.lower() == 'true', default=False, help="Use minimum bond length with metallic radius")
    parser.add_argument('--known_species', nargs='+', default=['Mn', 'Fe', 'Co', 'Ni', 'Ru', 'Nd', 'Gd', 'Tb', 'Dy', 'Yb']) 
    parser.add_argument('--sc', nargs='+', default=['shl'], help="Nanotube constraints: 'shl' (real tube geometry, free de-novo generation), 'van' (unconstrained)")
    parser.add_argument('--c_scale', default=None, help="Scale to multiply bond length for LZ constraint") 
    parser.add_argument('--c_vert', type=lambda x: x.lower() == 'true', default=False, help="Use LZ constraint for vertical bonds")
    parser.add_argument('--frac_z', default=None, type=parse_none_or_value, help="Fraction of z-coordinate of the 2D geometric pattern. If None, return random frac_z in [0, 1).")
    parser.add_argument('--reduced_mask', type=lambda x: x.lower() == 'true', default=False, help="Use reduced mask")
    # Decorator-count override: num_atom = num_known + a few decorators ('shl'
    # ignores this and uses the drawn tube's nsites).
    parser.add_argument('--max_decorators', default=None, type=lambda x: parse_none_or_value(x, int),
                        help="num_atom = num_known + randint(0, max_decorators). None -> distribution sampling.")
    parser.add_argument('--template_source', default=None, type=lambda x: parse_none_or_value(x, str),
                        help="shl template provenance filter: 'real' | 'synthetic' | None (any).")
    # Progressive skeleton pinning: ramp the skeleton mask via psi(t).
    parser.add_argument('--pin_schedule', default='none', choices=['none', 'linear', 'sigmoid'],
                        help="Progressive-pinning schedule ('none' -> original binary pinning).")
    parser.add_argument('--pin_alpha', default=10.0, type=float, help="Sigmoid steepness for psi(t).")
    parser.add_argument('--pin_tmid', default=0.6, type=float, help="Sigmoid inflection as fraction of T.")
    parser.add_argument('--pin_psi_start', default=0.0, type=float, help="Pinning strength at t=T.")
    parser.add_argument('--pin_psi_end', default=1.0, type=float, help="Pinning strength at t=0.")
    # Cylindrical radial band confining atoms to the tube wall (sc='shl').
    parser.add_argument('--cyl_masking', type=lambda x: x.lower() == 'true', default=False,
                        help="Softly confine atoms to the template's radial band (sc='shl').")
    parser.add_argument('--cyl_margin', default=0.1, type=float, help="r_hi = r_max * (1 + margin).")
    parser.add_argument('--cyl_r_lo_pct', default=5.0, type=float, help="Inner band-edge percentile for sc='shl' shell confinement.")
    parser.add_argument('--cyl_strength', default=1.0, type=float, help="Peak radial pull (scaled by psi(t)).")
    # v2 radial density guidance: concentrate atoms onto the wall within the band.
    parser.add_argument('--density_guidance', type=lambda x: x.lower() == 'true', default=False,
                        help="Nudge atoms up d/dr log rho(r) toward the template's wall (sc='shl').")
    parser.add_argument('--density_strength', default=0.05, type=float, help="Density-guidance radial step scale (scaled by psi(t)).")
    parser.add_argument('--bandwidth_scale', default=1.0, type=float, help="KDE bandwidth multiplier = wall thickness (<1 sharpens).")
    parser.add_argument('--density_grid_size', default=96, type=int, help="Per-template density force-table resolution.")
    args = parser.parse_args()
    main(args)

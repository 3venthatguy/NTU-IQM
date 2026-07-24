# GNN Screening: `gnn_eval/`

> **Audience:** programmer. **Purpose:** the near-independent classifier sub-project used to pre-screen generated structures before DFT.
> ⬅ Back to [docs hub](../README.md) · Related: [generation-scripts.md](generation-scripts.md)

`gnn_eval/` is a **near-standalone sub-project** with its own README, its own config template, and its own dependency set. Its job: train GNN classifiers that cheaply filter SCIGEN/NTGEN's raw output so only promising structures reach expensive DFT.

```
gnn_eval/
├── README.md                  its own docs (model list, Figshare data/weights)
├── config_eval_template.py    copy → config_eval.py (api_key, data dirs, seedn)
├── train_mag.py               train magnetic / non-magnetic classifier
├── train_stab_diff.py         train pristine-vs-diffused classifier
├── train_stab_ehull.py        train e_hull<0.1eV stability classifier
└── utils/
    ├── common.py        chemical_symbols, magnetic_atoms, make_dict()
    ├── data.py          Dataset_Cls, augment_data_diffuse
    ├── data_mp.py       Materials-Project API pull/filter
    ├── model_class.py   GraphNetworkClassifier (e3nn-based)
    ├── model_class_mag.py  GraphNetworkClassifierMag
    ├── model_train.py   train_classifier() loop
    ├── output.py        generate_dataframe(), CIF export
    ├── plot_data.py     plot_confusion_matrices()
    └── record.py        log_buffer (StringIO) + logger
```

## The three classifiers
| Trainer | Classifies | Notes |
|---|---|---|
| `train_stab_diff.py` | pristine vs. diffused structure | catches unphysical "melted" outputs |
| `train_stab_ehull.py` | e_above_hull < 0.1 eV (stable) | needs Matbench Discovery data |
| `train_mag.py` | magnetic vs. non-magnetic | magnetic-materials workflow |

Each uses an **e3nn equivariant GNN** (`GraphNetworkClassifier`) with hyperparameters `mul`, `irreps_out`, `lmax`, `nlayers`, `number_of_basis`, `radial_layers`, `radial_neurons`, `node_dim`, `node_embed_dim`; trained via `utils.model_train.train_classifier`, logged with `wandb`, and evaluated with confusion matrices (`utils.plot_data`).

## How generation output gets screened
`script/eval_screen.py` ([generation-scripts.md](generation-scripts.md)) consumes trained classifier weights by name from `config_scigen.py` (`stab_pred_name_A/B`, `mag_pred_name`) and runs a **sequential filter cascade** on the structures in `eval_gen_<label>.pt`:

```
all generated structures
  → [1] SMACT validity          (charge-neutrality / electronegativity)
  → [2] occupancy ratio < 1.7   (reject over-dense cells)
  → [3] GNN stability (A)       pristine-vs-diffused
  → [4] GNN stability (B)       e_hull
  → [5] GNN magnetism (optional, --screen_mag)
  → surviving structures → CIF files + text log
```

Each stage narrows the surviving DataFrame; the survivors are written as CIFs (`generate_cif_files`) with a log captured via `gnn_eval.utils.record.log_buffer`.

## Setup (only if retraining classifiers)
```bash
cp gnn_eval/config_eval_template.py gnn_eval/config_eval.py
# edit api_key, model_dir, data_dir, seedn
# download training data + pretrained weights per gnn_eval/README.md (Figshare)
cd gnn_eval && python train_stab_diff.py   # or train_stab_ehull.py / train_mag.py
```
`gnn_eval/README.md` lists the model architectures (VGNN, E3NN PDOS, E3NN Magnetic Order) with paper links and its own dependencies (`e3nn==0.5.1`, `plotly`, `mp_api`, `mendeleev`, `seaborn`, …).

> To just *screen* (not retrain), you only need the pretrained classifier weights + `config_scigen.py` pointing at them; run `script/eval_screen.py`. See [usage/workflows.md](../usage/workflows.md).

## Next

- The screening CLI → [generation-scripts.md](generation-scripts.md)
- Config files → [configuration.md](configuration.md)

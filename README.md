# MEP-LDM

## PPP interface update

MEP-LDM provides latent diffusion tooling for porous-media microstructure modeling, including data loading, feature extraction, model training, sampling, and evaluation utilities.

## Package Layout

```text
MEP-LDM/
├── mep_ldm/                 # Project-specific data, feature, model, metric, and trainer code
│   ├── data/                # Binary volume datasets, datamodules, and transforms
│   ├── features/            # Morphology and physical-property feature extraction
│   ├── metrics/             # Evaluation metrics and reporting helpers
│   ├── models/              # Model construction helpers
│   └── trainers/            # Training, loading, and evaluation orchestration
├── mep_ldm_core/            # Diffusion, network, scheduler, and autoencoder components
├── scripts/training/        # Public training and inference entry points
├── tests/                   # Lightweight tests
├── requirements.txt
└── setup.py
```

## Installation

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Training

Raw datasets and checkpoints are not included in this repository. Pass a local dataset path explicitly, or set `data.path` in the YAML file before running training.

```bash
python scripts/training/train_mep_ldm_vae.py \
  --cfgpath /path/to/vae.yaml \
  --datapath /path/to/volume.raw \
  --fast_dev_run
```

```bash
python scripts/training/train_mep_ldm.py \
  --cfgpath /path/to/ldm.yaml \
  --datapath /path/to/volume.raw \
  --fast_dev_run
```

The LDM configuration expects a locally trained VAE checkpoint at `model.autoencoder.checkpoint_path` before full training.

## Physical Property Predictor

The differentiable physical-property predictor is provided as `mep_ldm.models.physical_property_predictor`. It predicts porosity, specific surface area, and fractal dimension from binary volumes or logits using analytic operators and MLP calibrators.

```bash
python scripts/training/train_property_predictor.py \
  --loader tiff \
  --data_path /path/to/tiff_dataset \
  --output_dir outputs/property_predictor
```

For TIFF input, use `train/` and `val/` subfolders. Filenames should include labels in the form `phi_0.113_D_2.160_Sv_0.4302.tif`.

## Tests

```bash
python tests/run_tests.py
```

## Data Sources

This project has used public porous-media resources such as:

- The Eleven Sandstones database at Digital Rocks Portal: <https://www.digitalrocksportal.org/projects/317>
- Micro-CT Images and Networks from Imperial College London: <https://www.imperial.ac.uk/earth-science/research/research-groups/pore-scale-modelling/micro-ct-images-and-networks/>


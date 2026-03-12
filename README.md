<div align="center" style="line-height:1">
  <a href="https://github.com/Gewu-Intelligence" target="_blank"><img alt="github" src="https://img.shields.io/badge/Github-Gewu-blue?logo=github"/></a>
  <a href="https://github.com/Gewu-Intelligence/Huntianling"><img alt="Homepage" src="https://img.shields.io/badge/🤖Skills-Huntianling-blue"/></a>
  <a href="https://github.com/Gewu-Intelligence/Suiren-Foundation-Model"><img alt="Homepage" src="https://img.shields.io/badge/Base-Suiren-blue"/></a>
  <a href="https://drive.google.com/file/d/1vUMYzhmhCeNU18WE5D_xV4gQWxfU7kI7/view?usp=sharing"><img alt="slides" src="https://img.shields.io/badge/Slides-Suiren-white?logo=slideshare"/></a>
</div>

<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/moonshotai/Kimi-K2.5/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Modified_MIT-f5de53?&color=f5de53"/></a>
</div>


# Suiren Pre-trained Molecular Model Fine-tuning Framework

A PyTorch-based fine-tuning framework for adapting Suiren pre-trained molecular models to various downstream property prediction tasks.

## Overview

[suiren-family](suiren-family.jpg)

The Suiren family offers multiple backbone variants, among which **Suiren-ConfAvg** is designed to learn conformational averaging features. Its embeddings can assist in predicting various macroscopic molecular properties, such as density, melting point, ADMET, and more.

In this repository, we provide a complete set of fine-tuning models and a training framework, allowing you to train on your own data. Notably, the Suiren-ConfAvg pre-trained model offers significant benefits for tasks with limited data.

[finetune-model](finetune-model.png)

This project provides tools to fine-tune pre-trained molecular encoders for two types of tasks:
- **Regression**: Predicting continuous molecular properties (e.g., density, solubility)
- **Classification**: Predicting binary/multi-class molecular properties (e.g., ADMET BBB, ADMET toxicity)

## Repository Structure

```
├── main.py                      # Main fine-tuning training script
├── inference.py                 # Inference script for predictions
├── engine.py                    # Training and evaluation loops
├── optim_factory.py            # Optimizer and scheduler factory
├── logger.py                   # Custom logging utilities
├── utils.py                    # Utility functions
├── data/                       # Dataset directory
├── models/                     # Model definitions
├── checkpoints/               # Saved model checkpoints
└── logs/                      # Training logs
```

## Installation

### Environment Setup

**Install Dependencies**:

- Python 3.8+
- PyTorch 2.0+
- numpy
- torch_geometric
- pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv
- timm==0.4.12
- torcheval

## Data Preparation

### Dataset Formats

Datasets must be CSV files containing two columns: `SMILES` and `value`

- **SMILES**: SMILES string representation of the molecule
- **value**: The property value (continuous for regression, discrete for classification)

example:
```
SMILES,value
O=C(O)c1ccccc1,-4.0050001
```

or:
```
SMILES,value
Cc1ccccc1,1
```

### Data Organization

#### Option 1: Random Train/Val Split
```
data/
└── {property_name}/
    └── raw/
        └── {property_name}.csv
```

Use: `--data-mode smiles_random` (default)

#### Option 2: Predefined Train/Val Split  
```
data/
└── {property_name}/
    └── raw/
        ├── {property_name}_train.csv
        └── {property_name}_valid.csv
```

Use: `--data-mode smiles_deined`

#### Option 3: Predefined Train/Test Split (Split Train to Train/Val)
```
data/
└── {property_name}/
    └── raw/
        ├── {property_name}.csv
        └── {property_name}_test.csv
```

Use: `--data-mode smiles_deined --tvt`

## Pretrained Models

Download pretrained model checkpoints:

| Model | Purpose | Download Link |
|-------|---------|---------------|
| Suiren-ConfAvg | Pretrain graph neural network for SMILES | `https://huggingface.co/ajy112/Suiren-ConfAvg` |

## Usage

### Basic Regression Fine-tuning

```bash
python -m torch.distributed.launch \
  --nproc_per_node={num_gpu} main.py \
  --name {property_name} \
  --mode regression \
  --data-mode smiles_defined \
  --tvt \
  --epochs 100 \
  --batch-size 8 \
  --lr 1e-4 \
  --main-metric MAE \
  --checkpoint-pretrain /path/to/model.pt
```


```bash
python -m torch.distributed.launch \
  --nproc_per_node={num_gpu} main.py \
  --name {property_name} \
  --mode classification \
  --data-mode smiles_defined \
  --epochs 100 \
  --batch-size 8 \
  --lr 1e-4 \
  --main-metric ACC \
  --checkpoint-pretrain /path/to/model.pt
```

```bash
python -m torch.distributed.launch \
  --nproc_per_node={num_gpu} main.py \
  --name {property_name} \
  --mode classification \
  --data-mode smiles_random \
  --epochs 100 \
  --batch-size 8 \
  --lr 1e-4 \
  --main-metric ACC \
  --checkpoint-pretrain /path/to/model.pt
```



## Quick Inference

Once you have a fine-tuned model, use the inference script for fast predictions:

### Single SMILES Prediction

```bash
python inference.py {task_name}
```

Interactive input:
```bash
$ python inference.py acentric_factor
Enter SMILES or CSV path: c1cc2c3c(cccc3c1)CC2
{
  "SMILES": "c1cc2c3c(cccc3c1)CC2",
  "prediction": 0.381
}
```

### Batch Prediction from CSV

Provide a CSV file with a `SMILES` column:

```bash
$ python inference.py LD50
In SMILES or CSV path: test_data.csv
```

The source CSV will be updated with predictions in a `value` column.

### Inference Options

```bash
python inference.py {task_name} [--checkpoint /path/to/model.pt] [--batch-size 32] [--device auto]
```

## Key Arguments

### Task Configuration
- `--mode`: regression or classification
- `--name`: Property name (used for logging and checkpoint naming and identify data directory)
- `--loss`: Loss function - l1 or l2 (regression only, classification default is cross_entropy)
- `--main-metric`: Primary metric for model selection (MAE, R2, ACC, AUPRC, AUROC)

### Data Configuration
- `--data-mode`: smiles_random or smiles_deined
- `--tvt`: Use train/val/test split, must be used with `--data-mode smiles_deined`
- `--ratio`: Train/val split ratio (default: 0.8)

### Training Configuration
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size per GPU (default: 2)
- `--lr`: Learning rate (default: 2e-4)
- `--weight-decay`: L2 regularization (default: 0.01)
- `--warmup-epochs`: LR warmup epochs (default: 0)

## Output Structure

Results are organized by property name and experiment timestamp:

```
checkpoints/{property_name}/{timestamp}/
├── {property_name}_best.pt          # Best model checkpoint
└── {property_name}_ema_best.pt     # Best EMA model (if enabled)

logs/{property_name}/
└── {timestamp}.log                  # Training log
```

## Notes

- Data preprocessing and dataset splitting are performed automatically on first run
- Preprocessing results are cached for faster subsequent runs
- To reset preprocessing cache: `rm -r data/{property_name}/processed/`
- Different property datasets should use different names to avoid conflicts

<!-- ## Citation

If you use this framework, please cite the relevant papers for the underlying models. -->

## License

See LICENSE file for details.

## Support

For issues, questions, or contributions, please open an issue on GitHub or contact the development team (junyian@gmail.com).

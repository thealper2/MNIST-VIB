# Deep Variational Information Bottleneck (VIB) Implementation

A PyTorch implementation of the Deep Variational Information Bottleneck (VIB) for MNIST classification, featuring latent space visualization and model checkpointing.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/thealper2/MNIST-VIB.git
cd MNIST-VIB
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
python3 main.py --epochs 10 --batch-size 256 --latent-dim 128 --beta 0.1 --output-dir my_results
```

### Resume Training

```bash
python3 main.py --resume --checkpoint-dir checkpoints
```

### Evaluation Only

```bash
python3 main.py --evaluate-only --checkpoint-dir checkpoints
```

### All Available Options

```bash
python3 main.py --help

# Output
Usage: main.py [OPTIONS]

Options:
  --epochs INTEGER               Number of training epochs  [default: 50]
  --batch-size INTEGER           Batch size for training  [default: 128]
  --latent-dim INTEGER           Dimension of latent space  [default: 256]
  --hidden-dim INTEGER           Dimension of hidden layers  [default: 1024]
  --learning-rate FLOAT          Learning rate  [default: 0.001]
  --beta FLOAT                   Weight for KL term in VIB loss  [default: 0.001]
  --output-dir TEXT              Directory to save results  [default: results]
  --seed INTEGER                 Random seed
  --evaluate-only                Only evaluate without training
  --resume                       Resume training from checkpoint
  --checkpoint-dir TEXT          Directory to save/load checkpoints  [default: checkpoints]
  --help                         Show this message and exit.
```

## References

1. Original VIB Paper:
[Deep Variational Information Bottleneck](https://arxiv.org/abs/1612.00410)
Alemi et al., ICLR 2017

2. PyTorch:
[https://pytorch.org](https://pytorch.org)
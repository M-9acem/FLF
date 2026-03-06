"""
Generate canonical initial weights (w_0) for each model architecture.
Run this ONCE before any experiments. All subsequent experiments will
load from init_weights/<model>_w0.pt to guarantee identical starting points.

Usage:
    python generate_init_weights.py
"""

import torch
from pathlib import Path
from src.models import SimpleCNN, LeNet5, ResNet8, ResNet18, ResNet50

INIT_WEIGHTS_DIR = Path('init_weights')
INIT_WEIGHTS_DIR.mkdir(exist_ok=True)

MODELS = {
    'simple_cnn': lambda: SimpleCNN(num_classes=10, num_channels=3),
    'lenet5':     lambda: LeNet5(num_classes=10, num_channels=3),
    'resnet8':    lambda: ResNet8(num_classes=10, num_channels=3),
    'resnet18':   lambda: ResNet18(num_classes=10, num_channels=3),
    'resnet50':   lambda: ResNet50(num_classes=10, num_channels=3),
}

print('Generating canonical initial weights...')
for model_name, factory in MODELS.items():
    out_path = INIT_WEIGHTS_DIR / f'{model_name}_w0.pt'
    if out_path.exists():
        print(f'  [skip] {out_path} already exists — delete it to regenerate')
        continue
    model = factory()
    torch.save({k: v.cpu().clone() for k, v in model.state_dict().items()}, out_path)
    print(f'  [ok]   {out_path}')

print(f'\nDone. All experiments will now load from init_weights/<model>_w0.pt')

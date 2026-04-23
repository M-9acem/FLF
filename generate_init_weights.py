"""
Generate canonical initial weights (w_0) for each model/dataset combination.
Run this ONCE before experiments. De/centralized runs can then load
init_weights/<model>_<dataset>_w0.pt to guarantee identical starting points.

Legacy model-only files (<model>_w0.pt) are also generated for compatibility.

Usage:
    python generate_init_weights.py
"""

import torch
from pathlib import Path
from src.models import SimpleCNN, LeNet5, ResNet8, ResNet18, ResNet50

INIT_WEIGHTS_DIR = Path('init_weights')
INIT_WEIGHTS_DIR.mkdir(exist_ok=True)

MODEL_FACTORIES = {
    'simple_cnn': SimpleCNN,
    'lenet5': LeNet5,
    'resnet8': ResNet8,
    'resnet18': ResNet18,
    'resnet50': ResNet50,
}

DATASET_CHANNELS = {
    'mnist': 1,
    'fashion_mnist': 1,
    'cifar10': 3,
}

print('Generating canonical initial weights...')
for model_name, model_cls in MODEL_FACTORIES.items():
    # Keep legacy model-only file using 3 channels (historical default).
    legacy_path = INIT_WEIGHTS_DIR / f'{model_name}_w0.pt'
    if not legacy_path.exists():
        legacy_model = model_cls(num_classes=10, num_channels=3)
        torch.save({k: v.cpu().clone() for k, v in legacy_model.state_dict().items()}, legacy_path)
        print(f'  [ok]   {legacy_path}')
    else:
        print(f'  [skip] {legacy_path} already exists — delete it to regenerate')

    for dataset_name, num_channels in DATASET_CHANNELS.items():
        out_path = INIT_WEIGHTS_DIR / f'{model_name}_{dataset_name}_w0.pt'
        if out_path.exists():
            print(f'  [skip] {out_path} already exists — delete it to regenerate')
            continue
        model = model_cls(num_classes=10, num_channels=num_channels)
        torch.save({k: v.cpu().clone() for k, v in model.state_dict().items()}, out_path)
        print(f'  [ok]   {out_path}')

print('\nDone. Dataset-aware weights are available under init_weights/<model>_<dataset>_w0.pt')

"""Generate canonical data partitions once for reproducible experiments.

Pre-generates the default partition(s) used by run_all_mixing_methods.py.
Partitions are saved to  data_partition/  and loaded automatically by main.py
on subsequent runs, so every experiment sees identical client data splits.

Usage:
    python generate_partition.py
"""
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from src.utils import get_dataset, partition_data

# Configs to pre-generate: (dataset, data_dir, num_clients, partition_type, alpha, classes_per_client)
CONFIGS = [
    ('cifar10', './data', 40, 'dirichlet', 0.5, None),
    ('cifar10', './data', 10, 'dirichlet', 0.5, None),
]

OUT_DIR = Path('data_partition')
OUT_DIR.mkdir(exist_ok=True)

for dataset, data_dir, num_clients, partition_type, alpha, cpc in CONFIGS:
    if partition_type == 'pathological':
        fname = f'{dataset}_N{num_clients}_pathological_c{cpc}.pkl'
    else:
        fname = f'{dataset}_N{num_clients}_{partition_type}_a{alpha}.pkl'
    out_path = OUT_DIR / fname

    if out_path.exists():
        print(f'[SKIP] Already exists: {out_path}')
        continue

    print(f'[GEN]  {out_path} ...')
    train_dataset, _ = get_dataset(dataset, data_dir)
    indices = partition_data(
        train_dataset,
        num_clients,
        partition_type=partition_type,
        alpha=alpha,
        num_classes_per_client=cpc or 2,
    )
    with open(out_path, 'wb') as f:
        pickle.dump(indices, f)
    sizes = [len(idx) for idx in indices]
    print(f'       Saved {num_clients} client splits — min {min(sizes)}, max {max(sizes)}, total {sum(sizes)} samples')

print('Done.')

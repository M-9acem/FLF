"""Generate canonical data partitions once for reproducible experiments.

Pre-generates the default partition(s) used by run_all_mixing_methods.py.
Partitions are saved to  data_partition/  and loaded automatically by main.py
on subsequent runs, so every experiment sees identical client data splits.

Usage:
    python generate_partition.py
    python generate_partition.py --dataset cifar10 --num_clients 6 --partition dirichlet --alpha 0.5
"""
import argparse
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from src.utils import get_dataset, partition_data

# Configs to pre-generate in bulk mode:
# (dataset, data_dir, num_clients, partition_type, alpha, classes_per_client)
CONFIGS = [
    ('cifar10', './data', 40, 'dirichlet', 0.5, None),
    ('cifar10', './data', 10, 'dirichlet', 0.5, None),
]

OUT_DIR = Path('data_partition')
OUT_DIR.mkdir(exist_ok=True)

def _partition_filename(dataset: str, num_clients: int, partition_type: str, alpha: float, cpc: int | None) -> str:
    if partition_type == 'pathological':
        return f'{dataset}_N{num_clients}_pathological_c{cpc}.pkl'
    return f'{dataset}_N{num_clients}_{partition_type}_a{alpha}.pkl'


def generate_one(
    dataset: str,
    data_dir: str,
    num_clients: int,
    partition_type: str,
    alpha: float,
    classes_per_client: int | None,
) -> Path:
    fname = _partition_filename(dataset, num_clients, partition_type, alpha, classes_per_client)
    out_path = OUT_DIR / fname

    if out_path.exists():
        print(f'[SKIP] Already exists: {out_path}')
        return out_path

    print(f'[GEN]  {out_path} ...')
    train_dataset, _ = get_dataset(dataset, data_dir)
    indices = partition_data(
        train_dataset,
        num_clients,
        partition_type=partition_type,
        alpha=alpha,
        num_classes_per_client=classes_per_client or 2,
    )
    with open(out_path, 'wb') as f:
        pickle.dump(indices, f)
    sizes = [len(idx) for idx in indices]
    print(f'       Saved {num_clients} client splits — min {min(sizes)}, max {max(sizes)}, total {sum(sizes)} samples')
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Generate canonical dataset partitions.')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name (e.g., cifar10).')
    parser.add_argument('--data_dir', type=str, default='./data', help='Dataset root directory.')
    parser.add_argument('--num_clients', type=int, default=None, help='Number of clients.')
    parser.add_argument('--partition', type=str, default='dirichlet', choices=['iid', 'dirichlet', 'pathological'])
    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet alpha.')
    parser.add_argument('--classes_per_client', type=int, default=2, help='Classes/client for pathological split.')
    args = parser.parse_args()

    # If dataset and num_clients are provided, generate exactly one requested partition.
    if args.dataset is not None and args.num_clients is not None:
        generate_one(
            dataset=args.dataset,
            data_dir=args.data_dir,
            num_clients=args.num_clients,
            partition_type=args.partition,
            alpha=args.alpha,
            classes_per_client=args.classes_per_client if args.partition == 'pathological' else None,
        )
        print('Done.')
        return

    # Otherwise run bulk mode (backward-compatible behavior).
    for dataset, data_dir, num_clients, partition_type, alpha, cpc in CONFIGS:
        generate_one(dataset, data_dir, num_clients, partition_type, alpha, cpc)

    print('Done.')


if __name__ == '__main__':
    main()

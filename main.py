"""Main entry point for the Federated Learning Framework."""

import argparse
import pickle
import sys
import time
import torch
import numpy as np
import random
from pathlib import Path

from src.models import SimpleCNN, LeNet5, ResNet8, ResNet18, ResNet50
from src.utils import ComprehensiveLogger, get_dataset, partition_data, create_dataloaders
from src.centralized import FedAvgClient, FedAvgServer
from src.decentralized import P2PClient, P2PRunner, create_two_cluster_topology


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_model(model_name: str, num_classes: int, num_channels: int):
    """Create a neural network model.
    
    Args:
        model_name: Name of the model ('simple_cnn', 'lenet5', 'resnet18', 'resnet50')
        num_classes: Number of output classes
        num_channels: Number of input channels
        
    Returns:
        PyTorch model
    """
    if model_name.lower() in ['simple_cnn', 'simplecnn', 'cnn']:
        return SimpleCNN(num_classes=num_classes, num_channels=num_channels)
    elif model_name.lower() == 'lenet5':
        return LeNet5(num_classes=num_classes, num_channels=num_channels)
    elif model_name.lower() == 'resnet8':
        return ResNet8(num_classes=num_classes, num_channels=num_channels)
    elif model_name.lower() == 'resnet18':
        return ResNet18(num_classes=num_classes, num_channels=num_channels)
    elif model_name.lower() == 'resnet50':
        return ResNet50(num_classes=num_classes, num_channels=num_channels)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_centralized(args):
    """Run centralized federated learning (FedAvg).
    
    Args:
        args: Command-line arguments
    """
    print("\n" + "="*60)
    print("CENTRALIZED FEDERATED LEARNING (FedAvg)")
    print("="*60)
    
    # Set device(s)
    if torch.cuda.is_available() and not args.no_cuda:
        num_gpus = torch.cuda.device_count()
        devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]
        print(f"Using {num_gpus} GPU(s): {[str(d) for d in devices]}")
    else:
        devices = [torch.device("cpu")]
        print("Using device: cpu")
    device = devices[0]  # Primary device for global model
    
    # Set seed
    set_seed(args.seed)
    
    # Initialize logger
    logger = ComprehensiveLogger(
        log_dir=args.log_dir,
        experiment_name=args.experiment_name or "centralized_fedavg",
        mode="centralized"
    )
    print(f"Logging to: {logger.get_log_dir()}")
    
    # Save configuration
    config = vars(args)
    logger.save_config(config)
    
    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    train_dataset, test_dataset = get_dataset(args.dataset, args.data_dir)
    
    # Determine number of classes and channels
    if args.dataset.lower() in ['mnist', 'fashion_mnist']:
        num_classes = 10
        num_channels = 1
    elif args.dataset.lower() == 'cifar10':
        num_classes = 10
        num_channels = 3
    else:
        num_classes = 10
        num_channels = 3
    
    # Partition data — load canonical file or generate+save
    _part_fname = (
        f'{args.dataset}_N{args.num_clients}_pathological_c{args.classes_per_client}.pkl'
        if args.partition == 'pathological'
        else f'{args.dataset}_N{args.num_clients}_{args.partition}_a{args.alpha}.pkl'
    )
    _part_path = Path(args.partition_file) if args.partition_file else Path('data_partition') / _part_fname
    if _part_path.exists():
        print(f'Loading canonical partition from: {_part_path}')
        with open(_part_path, 'rb') as _f:
            client_indices = pickle.load(_f)
    else:
        print(f'Partition file not found at {_part_path} — partitioning and saving ...')
        client_indices = partition_data(
            train_dataset,
            args.num_clients,
            partition_type=args.partition,
            alpha=args.alpha,
            num_classes_per_client=args.classes_per_client
        )
        _part_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_part_path, 'wb') as _f:
            pickle.dump(client_indices, _f)
        print(f'Partition saved to: {_part_path} (will be reused in future runs)')
    _sizes = [len(idx) for idx in client_indices]
    print(f'{len(client_indices)} client splits — min {min(_sizes)}, max {max(_sizes)}, total {sum(_sizes)} samples')

    # Create dataloaders
    client_loaders = create_dataloaders(
        train_dataset,
        test_dataset,
        client_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create global model
    print(f"Creating {args.model} model...")
    global_model = create_model(args.model, num_classes, num_channels)
    
    # Create clients (distributed across GPUs round-robin)
    print(f"Initializing {args.num_clients} clients across {len(devices)} device(s)...")
    clients = []
    for i in range(args.num_clients):
        client_device = devices[i % len(devices)]
        train_loader, test_loader = client_loaders[i]
        model = create_model(args.model, num_classes, num_channels)
        client = FedAvgClient(
            client_id=i,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=client_device,
            learning_rate=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        clients.append(client)
    if len(devices) > 1:
        for d in devices:
            count = sum(1 for c in clients if c.device == d)
            print(f"  {d}: {count} clients")
    
    # Create server
    server = FedAvgServer(
        model=global_model,
        clients=clients,
        device=device,
        logger=logger
    )
    
    # Train
    print(f"\nStarting training: {args.rounds} rounds, {args.epochs} local epochs")
    print("-" * 60)
    start_time = time.time()
    server.train(num_rounds=args.rounds, local_epochs=args.epochs)
    total_training_time = time.time() - start_time
    
    # Generate final report
    print("\nGenerating final summary report...")
    print(f"Training completed in {total_training_time:.2f} seconds")
    print(f"\nResults saved to: {logger.get_log_dir()}")
    print(f"Simplified centralized metrics logged to:")
    print(f"  - centralized_global_metrics.csv (global model metrics per round)")
    print(f"  - centralized_global_per_class_metrics.csv (global model per-class metrics)")
    print(f"  - centralized_client_metrics.csv (per-client metrics per round)")
    print(f"  - centralized_client_per_class_metrics.csv (per-client per-class metrics)")


def run_decentralized(args):
    """Run decentralized federated learning (P2P).
    
    Args:
        args: Command-line arguments
    """
    print("\n" + "="*60)
    print("DECENTRALIZED FEDERATED LEARNING (P2P)")
    print("="*60)
    
    # Set device(s)
    if torch.cuda.is_available() and not args.no_cuda:
        num_gpus = torch.cuda.device_count()
        devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]
        print(f"Using {num_gpus} GPU(s): {[str(d) for d in devices]}")
    else:
        devices = [torch.device("cpu")]
        print("Using device: cpu")
    device = devices[0]  # Primary device
    
    # Set seed
    set_seed(args.seed)
    
    # Initialize logger
    logger = ComprehensiveLogger(
        log_dir=args.log_dir,
        experiment_name=args.experiment_name or "decentralized_p2p",
        mode="decentralized"
    )
    print(f"Logging to: {logger.get_log_dir()}")
    
    # Save configuration
    config = vars(args)
    logger.save_config(config)
    
    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    train_dataset, test_dataset = get_dataset(args.dataset, args.data_dir)
    
    # Determine number of classes and channels
    if args.dataset.lower() in ['mnist', 'fashion_mnist']:
        num_classes = 10
        num_channels = 1
    elif args.dataset.lower() == 'cifar10':
        num_classes = 10
        num_channels = 3
    else:
        num_classes = 10
        num_channels = 3
    
    # Partition data — load canonical file or generate+save
    _part_fname = (
        f'{args.dataset}_N{args.num_clients}_pathological_c{args.classes_per_client}.pkl'
        if args.partition == 'pathological'
        else f'{args.dataset}_N{args.num_clients}_{args.partition}_a{args.alpha}.pkl'
    )
    _part_path = Path(args.partition_file) if args.partition_file else Path('data_partition') / _part_fname
    if _part_path.exists():
        print(f'Loading canonical partition from: {_part_path}')
        with open(_part_path, 'rb') as _f:
            client_indices = pickle.load(_f)
    else:
        print(f'Partition file not found at {_part_path} — partitioning and saving ...')
        client_indices = partition_data(
            train_dataset,
            args.num_clients,
            partition_type=args.partition,
            alpha=args.alpha,
            num_classes_per_client=args.classes_per_client
        )
        _part_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_part_path, 'wb') as _f:
            pickle.dump(client_indices, _f)
        print(f'Partition saved to: {_part_path} (will be reused in future runs)')
    _sizes = [len(idx) for idx in client_indices]
    print(f'{len(client_indices)} client splits — min {min(_sizes)}, max {max(_sizes)}, total {sum(_sizes)} samples')

    # Create dataloaders
    client_loaders = create_dataloaders(
        train_dataset,
        test_dataset,
        client_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Log partition stats (class distribution per client)
    logger.log_partition_stats(client_indices, train_dataset.targets)
    print('Partition stats saved to partition_stats.csv')

    # Create topology
    if args.topology_file:
        # Load pre-generated topology
        print(f"Loading pre-generated topology from: {args.topology_file}")
        with open(args.topology_file, 'rb') as f:
            graph = pickle.load(f)
        print(f"Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    else:
        # Create new topology
        print(f"Creating two-cluster topology...")
        graph = create_two_cluster_topology(
            num_clients=args.num_clients,
            main_link_prob=args.main_link_prob,
            border_link_prob=args.border_link_prob,
            intra_cluster_prob=args.intra_cluster_prob,
            intra_cluster_communication=args.intra_cluster_communication
        )
        print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Load shared initial weights w_0 so every client starts identically
    w0_path = Path(args.init_weights) if args.init_weights else Path('init_weights') / f'{args.model}_w0.pt'
    if not w0_path.exists():
        print(f'Initial weights not found at {w0_path} — running generate_init_weights.py ...')
        import subprocess as _sp
        _sp.run([sys.executable, 'generate_init_weights.py'], check=True)
    shared_w0 = torch.load(w0_path, map_location='cpu', weights_only=True)
    print(f'Loaded shared initial weights from: {w0_path}')

    # Create clients (distributed across GPUs round-robin)
    print(f"Initializing {args.num_clients} P2P clients across {len(devices)} device(s)...")
    clients = []
    for i in range(args.num_clients):
        client_device = devices[i % len(devices)]
        train_loader, test_loader = client_loaders[i]
        model = create_model(args.model, num_classes, num_channels)
        model.load_state_dict(shared_w0)
        client = P2PClient(
            client_id=i,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=client_device,
            learning_rate=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        clients.append(client)
    if len(devices) > 1:
        for d in devices:
            count = sum(1 for c in clients if c.device == d)
            print(f"  {d}: {count} clients")
    
    # Create runner
    # Parse gossip schedule if provided: "5:0,3:100,1:200" -> [(0,5),(100,3),(200,1)]
    gossip_schedule = None
    if args.gossip_schedule:
        gossip_schedule = []
        for entry in args.gossip_schedule.split(','):
            steps_str, round_str = entry.strip().split(':')
            gossip_schedule.append((int(round_str), int(steps_str)))

    runner = P2PRunner(
        clients=clients,
        graph=graph,
        logger=logger,
        seed=args.seed,
        mixing_method=args.mixing_method,
        gossip_steps=args.gossip_steps,
        gossip_schedule=gossip_schedule,
        delay_d=args.delay_d
    )
    
    # Save network topology visualization
    print("\nGenerating network topology visualization...")
    try:
        runner.save_topology_visualization(
            output_dir=logger.get_log_dir(),
            experiment_name="network_topology"
        )
    except Exception as e:
        print(f"Warning: topology visualization skipped ({e})")
    
    # Train
    if args.gossip_steps > 1:
        print(f"\nStarting training: {args.rounds} rounds, {args.epochs} local epochs, {args.gossip_steps} gossip steps/round")
    else:
        print(f"\nStarting training: {args.rounds} rounds, {args.epochs} local epochs")
    print("-" * 60)
    start_time = time.time()
    runner.train(num_rounds=args.rounds, local_epochs=args.epochs)
    total_training_time = time.time() - start_time
    
    # Generate final report
    print("\nGenerating final summary report...")
    print(f"Training completed in {total_training_time:.2f} seconds")
    print(f"\nResults saved to: {logger.get_log_dir()}")
    print(f"Simplified P2P metrics logged to:")
    print(f"  - p2p_metrics.csv (post-gossip metrics per client per round)")
    print(f"  - p2p_per_class_metrics.csv (post-gossip per-class metrics)")
    print(f"  - pre_gossip_metrics.csv (pre-gossip metrics per client per round)")
    print(f"  - pre_gossip_per_class_metrics.csv (pre-gossip per-class metrics)")
    print(f"  - pre_gossip_weights/ (model weights before gossip, per round)")
    print(f"  - global_aggregated_metrics.csv (weighted-avg model pre-gossip, per round)")
    print(f"  - global_aggregated_per_class_metrics.csv (per-class metrics of aggregated model)")
    print(f"  - client_final_weights.pt (final model state dicts after gossip)")
    print(f"  - network_topology.html (interactive network visualization)")
    print(f"  - topology_info.txt (network statistics)")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Federated Learning Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training mode
    parser.add_argument(
        '--type',
        type=str,
        required=True,
        choices=['centralized', 'decentralized'],
        help='Type of federated learning (centralized or decentralized)'
    )
    
    # Training parameters
    parser.add_argument('--rounds', type=int, default=10, help='Number of federated rounds')
    parser.add_argument('--epochs', type=int, default=5, help='Number of local epochs per round')
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients')
    
    # Model parameters
    parser.add_argument(
        '--model',
        type=str,
        default='lenet5',
        choices=['simple_cnn', 'lenet5', 'resnet8', 'resnet18', 'resnet50'],
        help='Model architecture'
    )
    
    # Dataset parameters
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        choices=['mnist', 'cifar10', 'fashion_mnist'],
        help='Dataset to use'
    )
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    
    # Data partitioning
    parser.add_argument(
        '--partition',
        type=str,
        default='dirichlet',
        choices=['iid', 'dirichlet', 'pathological'],
        help='Data partitioning strategy'
    )
    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet alpha parameter')
    parser.add_argument('--classes_per_client', type=int, default=2, help='Classes per client (pathological)')
    
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    # Decentralized parameters
    parser.add_argument('--main_link_prob', type=float, default=1.0, help='Main bridge link probability')
    parser.add_argument('--border_link_prob', type=float, default=1.0, help='Border link probability')
    parser.add_argument('--intra_cluster_prob', type=float, default=0.8, help='Intra-cluster link probability')
    parser.add_argument('--topology_file', type=str, default=None, help='Path to pre-generated topology file (.pkl)')
    parser.add_argument(
        '--mixing_method',
        type=str,
        default='metropolis_hastings',
        choices=['metropolis_hastings', 'max_degree', 'jaccard', 'jaccard_dissimilarity', 'matcha'],
        help='Mixing matrix method for gossip aggregation'
    )
    parser.add_argument('--gossip_steps', type=int, default=1, help='Number of gossip iterations per round before next training')
    parser.add_argument('--gossip_schedule', type=str, default=None,
        help='Gossip drop schedule as "steps:from_round" pairs, e.g. "5:0,3:100,1:200" '
             'means 5 gossip steps until round 100, 3 until round 200, then 1. '
             'Overrides --gossip_steps when provided.')
    parser.add_argument('--delay_d', type=int, default=0,
        help='Delayed aggregation depth d. If >0, uses current self and delayed neighbors: '
             'w_i(t+1)=alpha_ii*w_i(t)+sum_{j!=i}a_ij*w_j(t-d).')
    parser.add_argument(
        '--intra_cluster_communication',
        action='store_true',
        help='If set, all nodes in a cluster are interconnected (default). If not set, nodes in a cluster only communicate with the edge node.'
    )
    
    # System parameters
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--init_weights', type=str, default=None,
        help='Path to a w_0.pt file. Defaults to init_weights/<model>_w0.pt. '
             'Run generate_init_weights.py once to create these files.')
    parser.add_argument('--partition_file', type=str, default=None,
        help='Path to a pre-generated partition .pkl file. '
             'Defaults to data_partition/<dataset>_N<num_clients>_<partition>_a<alpha>.pkl. '
             'Run generate_partition.py once to create these files.')
    
    # Logging parameters
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name')
    
    args = parser.parse_args()
    
    # Create directories
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    
    # Run appropriate training
    if args.type == 'centralized':
        run_centralized(args)
    elif args.type == 'decentralized':
        run_decentralized(args)
    else:
        raise ValueError(f"Unknown type: {args.type}")


if __name__ == "__main__":
    main()

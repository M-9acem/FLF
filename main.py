"""Main entry point for the Federated Learning Framework."""

import argparse
import time
import torch
import numpy as np
import random
from pathlib import Path

from src.models import SimpleCNN, LeNet5, ResNet18, ResNet50
from src.utils import ComprehensiveLogger as MetricsLogger, get_dataset, partition_data, create_dataloaders
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
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Set seed
    set_seed(args.seed)
    
    # Initialize logger
    logger = MetricsLogger(
        log_dir=args.log_dir,
        experiment_name=args.experiment_name or "centralized_fedavg"
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
    
    # Partition data
    print(f"Partitioning data among {args.num_clients} clients ({args.partition})...")
    client_indices = partition_data(
        train_dataset,
        args.num_clients,
        partition_type=args.partition,
        alpha=args.alpha,
        num_classes_per_client=args.classes_per_client
    )
    
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
    
    # Create clients
    print(f"Initializing {args.num_clients} clients...")
    clients = []
    for i in range(args.num_clients):
        train_loader, test_loader = client_loaders[i]
        model = create_model(args.model, num_classes, num_channels)
        client = FedAvgClient(
            client_id=i,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            learning_rate=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        clients.append(client)
    
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
    
    # Extract necessary data from logger's all_metrics
    global_metrics = logger.all_metrics.get('global_metrics', [])
    per_client_metrics = logger.all_metrics.get('per_client_metrics', [])
    
    # Find best global accuracy
    best_global_accuracy = 0.0
    best_round = 0
    if global_metrics:
        for metric in global_metrics:
            if metric.get('global_test_accuracy', 0) > best_global_accuracy:
                best_global_accuracy = metric['global_test_accuracy']
                best_round = metric['round']
    
    # Get final per-client accuracies (last round, last epoch)
    final_per_client_accuracies = {}
    if per_client_metrics:
        last_round = max(m['round'] for m in per_client_metrics)
        for metric in per_client_metrics:
            if metric['round'] == last_round:
                client_id = metric['client_id']
                # Get the most recent entry for this client in the final round
                if client_id not in final_per_client_accuracies:
                    final_per_client_accuracies[client_id] = metric.get('train_accuracy', 0.0)
    
    # Calculate total communication overhead (in centralized, this is simpler)
    # Each round has one upload and one download per client
    total_communication_overhead = 0.0  # Can be refined based on actual measurements
    
    logger.save_final_report(
        best_global_accuracy=best_global_accuracy,
        best_round=best_round,
        final_per_client_accuracies=final_per_client_accuracies,
        total_training_time=total_training_time,
        total_communication_overhead=total_communication_overhead
    )
    
    print(f"\nResults saved to: {logger.get_log_dir()}")


def run_decentralized(args):
    """Run decentralized federated learning (P2P).
    
    Args:
        args: Command-line arguments
    """
    print("\n" + "="*60)
    print("DECENTRALIZED FEDERATED LEARNING (P2P)")
    print("="*60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Set seed
    set_seed(args.seed)
    
    # Initialize logger
    logger = MetricsLogger(
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
    
    # Partition data
    print(f"Partitioning data among {args.num_clients} clients ({args.partition})...")
    client_indices = partition_data(
        train_dataset,
        args.num_clients,
        partition_type=args.partition,
        alpha=args.alpha,
        num_classes_per_client=args.classes_per_client
    )
    
    # Create dataloaders
    client_loaders = create_dataloaders(
        train_dataset,
        test_dataset,
        client_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create topology
    print(f"Creating two-cluster topology...")
    graph = create_two_cluster_topology(
        num_clients=args.num_clients,
        main_link_prob=args.main_link_prob,
        border_link_prob=args.border_link_prob,
        intra_cluster_prob=args.intra_cluster_prob
    )
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Create clients
    print(f"Initializing {args.num_clients} P2P clients...")
    clients = []
    for i in range(args.num_clients):
        train_loader, test_loader = client_loaders[i]
        model = create_model(args.model, num_classes, num_channels)
        client = P2PClient(
            client_id=i,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            learning_rate=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        clients.append(client)
    
    # Create runner
    runner = P2PRunner(
        clients=clients,
        graph=graph,
        logger=logger,
        seed=args.seed,
        mixing_method=args.mixing_method
    )
    
    # Train
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
    print(f"  - p2p_metrics.csv (overall metrics per client per round)")
    print(f"  - p2p_per_class_metrics.csv (per-class metrics)")


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
        default='simple_cnn',
        choices=['simple_cnn', 'lenet5', 'resnet18', 'resnet50'],
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
        default='iid',
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
    parser.add_argument(
        '--mixing_method',
        type=str,
        default='metropolis_hastings',
        choices=['metropolis_hastings', 'max_degree', 'jaccard', 'matcha'],
        help='Mixing matrix method for gossip aggregation'
    )
    
    # System parameters
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
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

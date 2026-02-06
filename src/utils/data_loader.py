"""Data loading and partitioning utilities for federated learning."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import List, Tuple, Dict
from collections import defaultdict


def get_dataset(dataset_name: str = "cifar10", data_dir: str = "./data"):
    """Get train and test datasets.
    
    Args:
        dataset_name: Name of the dataset ('mnist', 'cifar10', 'fashion_mnist')
        data_dir: Directory to store/load data
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    if dataset_name.lower() == "mnist":
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform
        )
        
    elif dataset_name.lower() == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform
        )
        
    elif dataset_name.lower() == "fashion_mnist":
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.FashionMNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            root=data_dir, train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_dataset, test_dataset


def partition_data(
    dataset,
    num_clients: int,
    partition_type: str = "iid",
    alpha: float = 0.5,
    num_classes_per_client: int = 2
) -> List[np.ndarray]:
    """Partition dataset among clients.
    
    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        partition_type: 'iid', 'dirichlet', or 'pathological'
        alpha: Dirichlet concentration parameter (for 'dirichlet')
        num_classes_per_client: Number of classes per client (for 'pathological')
        
    Returns:
        List of index arrays, one per client
    """
    labels = np.array(dataset.targets)
    
    if partition_type == "iid":
        return _iid_partition(num_clients, labels)
    elif partition_type == "dirichlet":
        return _dirichlet_partition(labels, num_clients, alpha)
    elif partition_type == "pathological":
        return _pathological_partition(labels, num_clients, num_classes_per_client)
    else:
        raise ValueError(f"Unknown partition type: {partition_type}")


def _iid_partition(num_clients: int, labels: np.ndarray) -> List[np.ndarray]:
    """IID partitioning - randomly shuffle and split evenly."""
    indices = np.random.permutation(len(labels))
    splits = np.array_split(indices, num_clients)
    return [np.array(split, dtype=np.int64) for split in splits]


def _dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    min_partition_size: int = 10
) -> List[np.ndarray]:
    """Non-IID partitioning using Dirichlet distribution."""
    num_classes = labels.max() + 1
    class_indices = [np.where(labels == y)[0] for y in range(num_classes)]
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for y in range(num_classes):
        np.random.shuffle(class_indices[y])
        class_size = len(class_indices[y])

        # Sample proportions until valid
        trial = 0
        while True:
            proportions = np.random.dirichlet(alpha=[alpha] * num_clients)
            splits = (np.cumsum(proportions) * class_size).astype(int)[:-1]
            shards = np.split(class_indices[y], splits)

            # Check if all shards are large enough
            if all(len(shard) >= min_partition_size or len(shard) == 0 for shard in shards):
                break
        
            trial += 1
            if trial == 10:
                raise ValueError(
                    "Max attempts (10) reached. Please adjust alpha."
                )
        
        # Assign shards to clients
        for client_id, shard in enumerate(shards):
            client_indices[client_id].extend(shard.tolist())

    return [np.array(sorted(idxs), dtype=np.int64) for idxs in client_indices]


def _pathological_partition(
    labels: np.ndarray,
    num_clients: int,
    num_classes_per_client: int
) -> List[np.ndarray]:
    """Pathological non-IID partitioning - each client has limited classes."""
    num_classes = len(np.unique(labels))
    
    # Group indices by class
    class_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_to_indices[int(label)].append(idx)
    
    # Calculate shards per class
    shard_per_class = num_classes_per_client * num_clients // num_classes
    
    if (num_clients * num_classes_per_client) % num_classes != 0:
        raise ValueError(
            "num_clients * num_classes_per_client must be divisible by num_classes"
        )
    
    # Create shards
    shards = []
    for label in range(num_classes):
        indices = class_to_indices[label]
        np.random.shuffle(indices)
        
        # Split into shards
        shard_size = len(indices) // shard_per_class
        for i in range(shard_per_class):
            start = i * shard_size
            end = (i + 1) * shard_size if i < shard_per_class - 1 else len(indices)
            shards.append(indices[start:end])
    
    # Randomly assign shards to clients
    np.random.shuffle(shards)
    client_indices = [[] for _ in range(num_clients)]
    
    for client_id in range(num_clients):
        for _ in range(num_classes_per_client):
            if shards:
                client_indices[client_id].extend(shards.pop())
    
    return [np.array(sorted(idxs), dtype=np.int64) for idxs in client_indices]


def create_dataloaders(
    train_dataset,
    test_dataset,
    client_indices: List[np.ndarray],
    batch_size: int = 32,
    num_workers: int = 0
) -> List[Tuple[DataLoader, DataLoader]]:
    """Create train and test dataloaders for each client.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        client_indices: List of index arrays for each client
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        
    Returns:
        List of (train_loader, test_loader) tuples, one per client
    """
    loaders = []
    
    for indices in client_indices:
        # Create training loader
        train_subset = Subset(train_dataset, indices)
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        
        # Use full test set for each client
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        loaders.append((train_loader, test_loader))
    
    return loaders


def get_class_distribution(dataset, indices: np.ndarray) -> Dict[int, int]:
    """Get class distribution for a subset of data.
    
    Args:
        dataset: PyTorch dataset
        indices: Indices of the subset
        
    Returns:
        Dictionary mapping class labels to counts
    """
    labels = np.array(dataset.targets)
    subset_labels = labels[indices]
    
    distribution = {}
    unique, counts = np.unique(subset_labels, return_counts=True)
    for label, count in zip(unique, counts):
        distribution[int(label)] = int(count)
    
    return distribution

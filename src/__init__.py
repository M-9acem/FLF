"""Federated Learning Framework - Main Package"""

__version__ = "1.0.0"
__author__ = "Federated Learning Team"

# Import main components for easy access
from src.centralized import FedAvgClient, FedAvgServer
from src.decentralized import P2PClient, P2PRunner, create_two_cluster_topology
from src.models import SimpleCNN, Net, LeNet5, ResNet18, ResNet50
from src.utils import MetricsLogger, get_dataset, partition_data, create_dataloaders

__all__ = [
    'FedAvgClient',
    'FedAvgServer',
    'P2PClient',
    'P2PRunner',
    'create_two_cluster_topology',
    'SimpleCNN',
    'Net',
    'LeNet5',
    'ResNet18',
    'ResNet50',
    'MetricsLogger',
    'get_dataset',
    'partition_data',
    'create_dataloaders',
]

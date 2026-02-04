"""Centralized federated learning components."""

from .client import FedAvgClient
from .server import FedAvgServer

__all__ = ['FedAvgClient', 'FedAvgServer']

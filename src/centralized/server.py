"""FedAvg server implementation."""

import torch
import torch.nn as nn
from typing import List, Dict
import copyimport numpy as npfrom src.centralized.client import FedAvgClient
from src.utils.logger import MetricsLogger


class FedAvgServer:
    """Federated Averaging server."""
    
    def __init__(
        self,
        model: nn.Module,
        clients: List[FedAvgClient],
        device: torch.device,
        logger: MetricsLogger = None
    ):
        """Initialize FedAvg server.
        
        Args:
            model: Global model
            clients: List of clients
            device: Device to run computations
            logger: Metrics logger
        """
        self.global_model = model.to(device)
        self.clients = clients
        self.device = device
        self.logger = logger
        
        # Initialize all clients with global model
        self._broadcast_model()
    
    def _broadcast_model(self):
        """Send global model to all clients."""
        global_state = self.global_model.state_dict()
        for client in self.clients:
            client.set_parameters(copy.deepcopy(global_state))
    
    def aggregate(self, client_states: List[Dict[str, torch.Tensor]], client_weights: List[int]):
        """Aggregate client models using weighted averaging.
        
        Args:
            client_states: List of client model state dicts
            client_weights: List of sample counts for weighting
        """
        total_samples = sum(client_weights)
        
        # Initialize aggregated state
        aggregated_state = {}
        for key in client_states[0].keys():
            aggregated_state[key] = torch.zeros_like(client_states[0][key])
        
        # Weighted averaging
        for state, weight in zip(client_states, client_weights):
            for key in state.keys():
                aggregated_state[key] += state[key] * (weight / total_samples)
        
        # Update global model
        self.global_model.load_state_dict(aggregated_state)
    
    def train_round(self, round_num: int, local_epochs: int = 1) -> Dict[str, any]:
        """Execute one round of federated learning.
        
        Args:
            round_num: Current round number
            local_epochs: Number of local training epochs
            
        Returns:
            Dictionary with round metrics
        """
        print(f"\n=== Round {round_num} ===")
        
        # Broadcast current global model
        self._broadcast_model()
        
        # Collect metrics
        client_states = []
        client_weights = []
        client_losses = []
        client_accuracies = []
        gradient_norms_list = []
        all_class_metrics = {}
        
        # Each client trains locally
        for idx, client in enumerate(self.clients):
            print(f"Training client {client.client_id}...", end=' ')
            metrics = client.train(epochs=local_epochs)
            
            # Collect model updates
            client_states.append(client.get_parameters())
            client_weights.append(metrics['num_samples'])
            client_losses.append(metrics['final_loss'])
            client_accuracies.append(metrics['final_accuracy'])
            
            # Log comprehensive per-client metrics
            if self.logger:
                for epoch_idx, (loss, acc, grad_norm) in enumerate(
                    zip(metrics['losses'], metrics['accuracies'], metrics['gradient_norms'])
                ):
                    # Use new comprehensive logging methods
                    self.logger.log_per_client_metrics(
                        client_id=client.client_id,
                        round_num=round_num,
                        epoch=epoch_idx,
                        train_loss=loss,
                        train_accuracy=acc,
                        test_loss=None,
                        test_accuracy=None
                    )
                
                # Evaluate client on test set after training
                test_metrics = client.evaluate(compute_per_class_metrics=True)
                
                # Log per-class metrics with precision/recall/f1
                if 'class_metrics' in test_metrics:
                    self.logger.log_per_class_metrics(
                        client_id=client.client_id,
                        round_num=round_num,
                        class_metrics=test_metrics['class_metrics']
                    )
                    all_class_metrics[client.client_id] = test_metrics['class_metrics']
            
            gradient_norms_list.append(metrics['gradient_norms'][-1])
            
            print(f"Loss: {metrics['final_loss']:.4f}, Acc: {metrics['final_accuracy']:.2f}%")
        
        # Aggregate models
        print("Aggregating models...")
        import time
        agg_start = time.time()
        self.aggregate(client_states, client_weights)
        agg_time = time.time() - agg_start
        
        # Evaluate global model
        global_metrics = self._evaluate_global_model()
        
        print(f"Global Model - Loss: {global_metrics['loss']:.4f}, Acc: {global_metrics['accuracy']:.2f}%")
        
        # Log comprehensive global and convergence metrics
        if self.logger:
            # Global metrics with aggregation time
            self.logger.log_global_metrics(
                round_num=round_num,
                global_train_loss=sum(client_losses) / len(client_losses),
                global_train_accuracy=sum(client_accuracies) / len(client_accuracies),
                global_test_loss=global_metrics['loss'],
                global_test_accuracy=global_metrics['accuracy'],
                num_participating_clients=len(self.clients),
                aggregation_time=agg_time
            )
            
            # Convergence metrics
            self.logger.log_convergence_metrics(
                round_num=round_num,
                client_accuracies=client_accuracies,
                client_losses=client_losses,
                straggler_threshold=10.0  # 10% below mean
            )
            
            # Round summary
            self.logger.log_round_summary(
                round_num=round_num,
                num_clients_participated=len(self.clients),
                client_accuracies=client_accuracies,
                client_losses=client_losses,
                gradient_norms=gradient_norms_list
            )
        
        return {
            'round': round_num,
            'client_losses': client_losses,
            'client_accuracies': client_accuracies,
            'global_loss': global_metrics['loss'],
            'global_accuracy': global_metrics['accuracy'],
            'gradient_norms': gradient_norms_list
        }
    
    def _evaluate_global_model(self) -> Dict[str, any]:
        """Evaluate global model on all clients' test data.
        
        Returns:
            Dictionary with global evaluation metrics
        """
        # Broadcast global model
        self._broadcast_model()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for client in self.clients:
            metrics = client.evaluate()
            total_loss += metrics['loss'] * len(client.test_loader.dataset)
            total_correct += (metrics['accuracy'] / 100.0) * len(client.test_loader.dataset)
            total_samples += len(client.test_loader.dataset)
        
        avg_loss = total_loss / total_samples
        avg_accuracy = 100.0 * total_correct / total_samples
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy
        }
    
    def train(self, num_rounds: int, local_epochs: int = 1):
        """Train for multiple rounds.
        
        Args:
            num_rounds: Number of federated rounds
            local_epochs: Local epochs per round
        """
        # Log data distribution at start
        if self.logger:
            for client in self.clients:
                # Compute class distribution
                class_dist = {}
                num_classes = 10
                for _, targets in client.train_loader:
                    for target in targets:
                        label = target.item()
                        class_dist[label] = class_dist.get(label, 0) + 1
                
                total_samples = len(client.train_loader.dataset)
                
                # Compute heterogeneity score (KL divergence from uniform)
                uniform_dist = 1.0 / num_classes
                kl_div = 0.0
                for class_id in range(num_classes):
                    p = class_dist.get(class_id, 0) / total_samples
                    if p > 0:
                        kl_div += p * np.log(p / uniform_dist)
                
                self.logger.log_data_distribution(
                    client_id=client.client_id,
                    total_samples=total_samples,
                    class_distribution=class_dist,
                    data_heterogeneity_score=float(kl_div)
                )
        
        # Training loop
        for round_num in range(1, num_rounds + 1):
            self.train_round(round_num, local_epochs)
        
        print("\n=== Training Complete ===")

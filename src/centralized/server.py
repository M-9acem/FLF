"""FedAvg server implementation."""

import torch
import torch.nn as nn
from typing import List, Dict
import copy
import numpy as np
from src.centralized.client import FedAvgClient
from src.utils.logger import MetricsLogger


class FedAvgServer:
    """Federated Averaging server."""
    
    def __init__(
        self,
        model: nn.Module,
        clients: List[FedAvgClient],
        device: torch.device,
        logger: 'ComprehensiveLogger' = None
    ):
        """Initialize FedAvg server.
        
        Args:
            model: Global model
            clients: List of clients
            device: Device to run computations
            logger: Comprehensive metrics logger
        """
        self.global_model = model.to(device)
        self.clients = clients
        self.device = device
        self.logger = logger
        self.prev_global_gradient_norm = None
        
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
        
        # Each client trains locally
        for idx, client in enumerate(self.clients):
            print(f"Training client {client.client_id}...", end=' ')
            
            # Store previous gradient for change calculation
            prev_grad_norm = getattr(client, 'prev_gradient_norm', None)
            
            metrics = client.train(epochs=local_epochs)
            
            # Collect model updates
            client_states.append(client.get_parameters())
            client_weights.append(metrics['num_samples'])
            client_losses.append(metrics['final_loss'])
            client_accuracies.append(metrics['final_accuracy'])
            
            # Get gradient norm from last epoch
            grad_norm = metrics['gradient_norms'][-1] if metrics['gradient_norms'] else 0.0
            gradient_norms_list.append(grad_norm)
            
            # Calculate gradient change
            gradient_change = 0.0
            if prev_grad_norm is not None:
                gradient_change = abs(grad_norm - prev_grad_norm)
            client.prev_gradient_norm = grad_norm
            
            # Evaluate client and log simplified metrics
            if self.logger:
                test_metrics = client.evaluate(compute_per_class_metrics=True)
                
                self.logger.log_centralized_client_metrics(
                    client_id=client.client_id,
                    round_num=round_num,
                    test_accuracy=test_metrics['accuracy'],
                    test_loss=test_metrics['loss'],
                    gradient_norm=grad_norm,
                    gradient_change=gradient_change,
                    class_metrics=test_metrics.get('class_metrics', {})
                )
            
            print(f"Loss: {metrics['final_loss']:.4f}, Acc: {metrics['final_accuracy']:.2f}%")
        
        # Aggregate models
        print("Aggregating models...")
        import time
        agg_start = time.time()
        self.aggregate(client_states, client_weights)
        agg_time = time.time() - agg_start
        
        # Evaluate global model
        global_metrics = self._evaluate_global_model(compute_per_class=True)
        
        print(f"Global Model - Loss: {global_metrics['loss']:.4f}, Acc: {global_metrics['accuracy']:.2f}%")
        
        # Calculate global gradient norm and change
        global_grad_norm = np.mean(gradient_norms_list)
        global_gradient_change = 0.0
        if self.prev_global_gradient_norm is not None:
            global_gradient_change = abs(global_grad_norm - self.prev_global_gradient_norm)
        self.prev_global_gradient_norm = global_grad_norm
        
        # Log simplified global metrics
        if self.logger:
            self.logger.log_centralized_global_metrics(
                round_num=round_num,
                test_accuracy=global_metrics['accuracy'],
                test_loss=global_metrics['loss'],
                gradient_norm=global_grad_norm,
                gradient_change=global_gradient_change,
                class_metrics=global_metrics.get('class_metrics', {})
            )
        
        return {
            'round': round_num,
            'client_losses': client_losses,
            'client_accuracies': client_accuracies,
            'global_loss': global_metrics['loss'],
            'global_accuracy': global_metrics['accuracy'],
            'gradient_norms': gradient_norms_list
        }
    
    def _evaluate_global_model(self, compute_per_class: bool = False) -> Dict[str, any]:
        """Evaluate global model on all clients' test data.
        
        Args:
            compute_per_class: Whether to compute per-class metrics
            
        Returns:
            Dictionary with global evaluation metrics
        """
        # Broadcast global model
        self._broadcast_model()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_class_metrics = {}
        
        for client in self.clients:
            metrics = client.evaluate(compute_per_class_metrics=compute_per_class)
            total_loss += metrics['loss'] * len(client.test_loader.dataset)
            total_correct += (metrics['accuracy'] / 100.0) * len(client.test_loader.dataset)
            total_samples += len(client.test_loader.dataset)
            
            # Aggregate per-class metrics
            if compute_per_class and 'class_metrics' in metrics:
                for class_id, class_metric in metrics['class_metrics'].items():
                    if class_id not in all_class_metrics:
                        all_class_metrics[class_id] = {
                            'accuracy': 0.0,
                            'loss': 0.0,
                            'precision': 0.0,
                            'recall': 0.0,
                            'f1_score': 0.0,
                            'count': 0
                        }
                    all_class_metrics[class_id]['accuracy'] += class_metric.get('accuracy', 0.0)
                    all_class_metrics[class_id]['loss'] += class_metric.get('loss', 0.0)
                    all_class_metrics[class_id]['precision'] += class_metric.get('precision', 0.0)
                    all_class_metrics[class_id]['recall'] += class_metric.get('recall', 0.0)
                    all_class_metrics[class_id]['f1_score'] += class_metric.get('f1_score', 0.0)
                    all_class_metrics[class_id]['count'] += 1
        
        result = {
            'loss': total_loss / total_samples if total_samples > 0 else 0.0,
            'accuracy': (total_correct / total_samples * 100.0) if total_samples > 0 else 0.0
        }
        
        # Average per-class metrics
        if compute_per_class and all_class_metrics:
            averaged_class_metrics = {}
            for class_id, metrics in all_class_metrics.items():
                count = metrics['count']
                if count > 0:
                    averaged_class_metrics[class_id] = {
                        'accuracy': metrics['accuracy'] / count,
                        'loss': metrics['loss'] / count,
                        'precision': metrics['precision'] / count,
                        'recall': metrics['recall'] / count,
                        'f1_score': metrics['f1_score'] / count
                    }
            result['class_metrics'] = averaged_class_metrics
        
        return result
        
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
        # Training loop
        for round_num in range(1, num_rounds + 1):
            self.train_round(round_num, local_epochs)
        
        print("\n=== Training Complete ===")

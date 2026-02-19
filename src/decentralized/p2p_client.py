"""P2P client implementation for decentralized federated learning."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
import copy
import numpy as np


class P2PClient:
    """Peer-to-peer federated learning client."""
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0
    ):
        """Initialize P2P client.
        
        Args:
            client_id: Unique client identifier
            model: Neural network model
            train_loader: Training data loader
            test_loader: Test data loader
            device: Device to run computations
            learning_rate: Learning rate for optimizer
            momentum: Momentum for SGD
            weight_decay: Weight decay for regularization
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # Store neighbor models for gossip
        self.neighbor_models: Dict[int, Dict[str, torch.Tensor]] = {}
    
    def get_state(self) -> Dict[str, torch.Tensor]:
        """Get current model state.
        
        Returns:
            Model state dictionary
        """
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
    
    def set_state(self, state_dict: Dict[str, torch.Tensor]):
        """Set model state.
        
        Args:
            state_dict: Model state dictionary
        """
        self.model.load_state_dict(state_dict)
    
    def store_neighbor_models(self, neighbor_states: Dict[int, Dict[str, torch.Tensor]]):
        """Store models from neighbors.
        
        Args:
            neighbor_states: Dictionary mapping neighbor IDs to their model states
        """
        self.neighbor_models = copy.deepcopy(neighbor_states)
    
    def train(self, epochs: int = 1) -> Dict[str, any]:
        """Train the local model.
        
        Args:
            epochs: Number of local training epochs
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        
        epoch_losses = []
        epoch_accuracies = []
        epoch_gradient_norms = []
        
        # Per-class metrics
        num_classes = 10
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            gradient_norms = []
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Calculate gradient norm (flatten all grads into one vector)
                grad_vec = torch.cat([p.grad.data.flatten() for p in self.model.parameters() if p.grad is not None])
                total_norm = grad_vec.norm(2).item()
                gradient_norms.append(total_norm)
                
                optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                # Per-class accuracy
                for i in range(len(target)):
                    label = target[i].item()
                    class_correct[label] += (pred[i] == target[i]).item()
                    class_total[label] += 1
            
            avg_loss = total_loss / len(self.train_loader)
            accuracy = 100.0 * correct / total
            avg_gradient_norm = np.mean(gradient_norms)
            
            epoch_losses.append(avg_loss)
            epoch_accuracies.append(accuracy)
            epoch_gradient_norms.append(avg_gradient_norm)
        
        # Calculate per-class accuracies
        class_accuracies = {}
        for i in range(num_classes):
            if class_total[i] > 0:
                class_accuracies[i] = 100.0 * class_correct[i] / class_total[i]
            else:
                class_accuracies[i] = 0.0
        
        gradient_variance = np.var(gradient_norms) if len(gradient_norms) > 1 else 0.0
        
        return {
            'client_id': self.client_id,
            'num_samples': len(self.train_loader.dataset),
            'losses': epoch_losses,
            'accuracies': epoch_accuracies,
            'gradient_norms': epoch_gradient_norms,
            'gradient_variance': gradient_variance,
            'class_accuracies': class_accuracies,
            'final_loss': epoch_losses[-1],
            'final_accuracy': epoch_accuracies[-1]
        }
    
    def gossip_aggregate(self, weights: Dict[int, float]) -> float:
        """Aggregate model with neighbors using gossip protocol.
        
        Args:
            weights: Dictionary mapping neighbor IDs to mixing weights
            
        Returns:
            weight_diff: L2 norm of the parameter difference before/after aggregation
        """
        # Get current model state
        current_state = self.get_state()
        
        # Flatten pre-aggregation weights into a single vector
        pre_vec = torch.cat([v.flatten().float() for v in current_state.values()])
        
        # Initialize aggregated state
        aggregated_state = {}
        for key in current_state.keys():
            aggregated_state[key] = torch.zeros_like(current_state[key], dtype=torch.float32)
        
        # Add weighted contributions from self and neighbors
        for neighbor_id, weight in weights.items():
            if neighbor_id == self.client_id:
                # Self weight
                for key in current_state.keys():
                    aggregated_state[key] += current_state[key].float() * weight
            elif neighbor_id in self.neighbor_models:
                # Neighbor weight
                neighbor_state = self.neighbor_models[neighbor_id]
                for key in current_state.keys():
                    aggregated_state[key] += neighbor_state[key].float() * weight
        
        # Flatten post-aggregation weights into a single vector
        post_vec = torch.cat([v.flatten().float() for v in aggregated_state.values()])
        
        # Compute L2 norm of weight difference (model divergence due to gossip)
        weight_diff = (post_vec - pre_vec).norm(2).item()
        
        # Update model (cast back to original dtypes, e.g. Long for num_batches_tracked)
        self.set_state({
            k: v.to(dtype=current_state[k].dtype, device=self.device)
            for k, v in aggregated_state.items()
        })
        
        return weight_diff
    
    def evaluate(self, compute_per_class_metrics: bool = False) -> Dict[str, any]:
        """Evaluate the model on test data.
        
        Args:
            compute_per_class_metrics: Whether to compute precision/recall/F1 per class
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Per-class metrics
        num_classes = 10
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        
        # For sklearn metrics
        all_targets = []
        all_preds = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                # Store for sklearn metrics
                if compute_per_class_metrics:
                    all_targets.extend(target.cpu().numpy())
                    all_preds.extend(pred.cpu().numpy())
                
                # Per-class accuracy
                for i in range(len(target)):
                    label = target[i].item()
                    class_correct[label] += (pred[i] == target[i]).item()
                    class_total[label] += 1
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100.0 * correct / total
        
        # Calculate per-class accuracies
        class_accuracies = {}
        for i in range(num_classes):
            if class_total[i] > 0:
                class_accuracies[i] = 100.0 * class_correct[i] / class_total[i]
            else:
                class_accuracies[i] = 0.0
        
        result = {
            'client_id': self.client_id,
            'loss': avg_loss,
            'accuracy': accuracy,
            'class_accuracies': class_accuracies
        }
        
        # Add comprehensive per-class metrics if requested
        if compute_per_class_metrics and len(all_targets) > 0:
            from sklearn.metrics import precision_recall_fscore_support
            
            precision, recall, f1, support = precision_recall_fscore_support(
                all_targets, all_preds, average=None, zero_division=0
            )
            
            class_metrics = {}
            for class_id in range(num_classes):
                class_metrics[class_id] = {
                    'accuracy': class_accuracies.get(class_id, 0.0),
                    'precision': float(precision[class_id]) * 100,
                    'recall': float(recall[class_id]) * 100,
                    'f1_score': float(f1[class_id]) * 100,
                    'samples': int(support[class_id]),
                    'correct_predictions': class_correct[class_id]
                }
            
            result['class_metrics'] = class_metrics
        
        return result

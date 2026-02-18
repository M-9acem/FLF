"""FedAvg client implementation."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, List
import copy
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


class FedAvgClient:
    """Federated Averaging client."""
    
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
        """Initialize FedAvg client.
        
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
        
        # Track initial model state
        self.initial_state = None
    
    def set_parameters(self, state_dict: Dict[str, torch.Tensor]):
        """Set model parameters from server.
        
        Args:
            state_dict: Model state dictionary
        """
        self.model.load_state_dict(state_dict)
        self.initial_state = copy.deepcopy(state_dict)
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current model parameters.
        
        Returns:
            Model state dictionary
        """
        return copy.deepcopy(self.model.state_dict())
    
    def train(
        self,
        epochs: int = 1,
        log_metrics: bool = True
    ) -> Dict[str, any]:
        """Train the local model.
        
        Args:
            epochs: Number of local training epochs
            log_metrics: Whether to return detailed metrics
            
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
        num_classes = 10  # Default for CIFAR10/MNIST
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
        
        # Calculate gradient variance
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
    
    def evaluate(self, compute_per_class_metrics: bool = True) -> Dict[str, any]:
        """Evaluate the model on test data with comprehensive metrics.
        
        Args:
            compute_per_class_metrics: Whether to compute detailed per-class metrics
        
        Returns:
            Dictionary with evaluation metrics including per-class precision/recall/f1
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
        
        # For precision/recall/f1
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                # Store predictions for sklearn metrics
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                # Per-class accuracy
                for i in range(len(target)):
                    label = target[i].item()
                    class_correct[label] += (pred[i] == target[i]).item()
                    class_total[label] += 1
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100.0 * correct / total
        
        # Calculate comprehensive per-class metrics
        class_metrics = {}
        if compute_per_class_metrics:
            # Compute precision, recall, f1 using sklearn
            precision, recall, f1, support = precision_recall_fscore_support(
                all_targets, all_preds, average=None, zero_division=0
            )
            
            for i in range(num_classes):
                class_acc = 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
                class_metrics[i] = {
                    'accuracy': class_acc,
                    'precision': float(precision[i]) * 100 if i < len(precision) else 0.0,
                    'recall': float(recall[i]) * 100 if i < len(recall) else 0.0,
                    'f1_score': float(f1[i]) * 100 if i < len(f1) else 0.0,
                    'samples': class_total[i],
                    'correct_predictions': class_correct[i]
                }
        else:
            # Just accuracy
            for i in range(num_classes):
                if class_total[i] > 0:
                    class_metrics[i] = {'accuracy': 100.0 * class_correct[i] / class_total[i]}
                else:
                    class_metrics[i] = {'accuracy': 0.0}
        
        return {
            'client_id': self.client_id,
            'loss': avg_loss,
            'accuracy': accuracy,
            'class_accuracies': {k: v['accuracy'] for k, v in class_metrics.items()},
            'class_metrics': class_metrics  # Comprehensive per-class metrics
        }
    
    def get_gradient_dict(self) -> Dict[str, torch.Tensor]:
        """Get gradients as a dictionary of tensors.
        
        Returns:
            Dictionary mapping parameter names to gradient tensors
        """
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
        return gradients

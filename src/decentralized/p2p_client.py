"""P2P client implementation for decentralized federated learning."""

import copy
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Clone tensors so saved history is never mutated by later updates."""
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


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
        ssos_enabled: bool = False,
        g: float = 0.0,
        optimizer_name: str = 'sgd',
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
            ssos_enabled: Whether to use SSOS accelerated gossip
            g: SSOS acceleration coefficient
            optimizer_name: Optimizer to use ('sgd' or 'adam')
            momentum: Momentum for SGD
            weight_decay: Weight decay for regularization
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.learning_rate = learning_rate
        self.ssos_enabled = bool(ssos_enabled)
        self.g = float(g) if self.ssos_enabled else 0.0
        self.optimizer_name = optimizer_name.lower()
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.prev_model: Optional[Dict[str, torch.Tensor]] = None

        if self.optimizer_name not in {'sgd', 'adam'}:
            raise ValueError(
                f"Unsupported optimizer '{optimizer_name}'. Supported values: 'sgd', 'adam'"
            )
        
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

    def commit_prev_model(self):
        """Store the current model as the previous-round snapshot."""
        self.prev_model = self.get_state()

    def reset_prev_model(self):
        """Clear SSOS history so the next gossip step falls back to first-order."""
        self.prev_model = None
    
    def train(
        self,
        epochs: int = 1,
        round_num: int = None,
        total_epochs: int = 0,
        global_epoch_start: int = 0,
        use_lr_schedule: bool = False,
        warmup_epochs: int = 5,
        warmup_start_lr: float = 0.001,
    ) -> Dict[str, any]:
        """Train the local model.
        
        Args:
            epochs: Number of local training epochs
            round_num: Current round number
            total_epochs: Total training epochs across all rounds
            global_epoch_start: Global epoch index where this local train call starts
            use_lr_schedule: Whether to apply the custom 4-phase LR schedule
            warmup_epochs: Warmup length used by the LR schedule
            warmup_start_lr: Starting LR used by the warmup phase
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        if self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        
        epoch_losses = []
        epoch_accuracies = []
        epoch_gradient_norms = []
        lr_history = []
        last_grad_vec = None  # will hold the mean gradient vector over all batches of the last epoch
        
        # Per-class metrics
        num_classes = 10
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        
        for epoch in range(epochs):
            current_epoch = global_epoch_start + epoch
            if use_lr_schedule and total_epochs > 0:
                scheduled_lr = self._compute_scheduled_lr(
                    base_lr=self.learning_rate,
                    current_epoch=current_epoch,
                    total_epochs=total_epochs,
                    warmup_epochs=warmup_epochs,
                    warmup_start_lr=warmup_start_lr,
                )
                for param_group in optimizer.param_groups:
                    param_group['lr'] = scheduled_lr
                current_lr = scheduled_lr
            else:
                current_lr = self.learning_rate

            lr_history.append({
                'local_epoch': epoch,
                'global_epoch': current_epoch,
                'lr': current_lr,
                'phase': self._get_schedule_phase(current_epoch, total_epochs, warmup_epochs)
            })

            total_loss = 0.0
            correct = 0
            total = 0
            gradient_norms = []
            grad_vec_accumulator = None
            epoch_batch_count = 0
            
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
                # Accumulate flat gradient vector for mean over epoch
                grad_vec_accumulator = grad_vec.detach().clone() if grad_vec_accumulator is None else grad_vec_accumulator + grad_vec.detach()
                epoch_batch_count += 1
                
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
            # Compute mean gradient vector for this epoch; last epoch's value is kept after the loop
            if grad_vec_accumulator is not None:
                last_grad_vec = (grad_vec_accumulator / epoch_batch_count).cpu().numpy()
        
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
            'final_accuracy': epoch_accuracies[-1],
            'last_grad_vec': last_grad_vec,  # raw gradient vector: last batch of last epoch
            'lr_history': lr_history,
            'current_lr': lr_history[-1]['lr'] if lr_history else self.learning_rate,
        }

    @staticmethod
    def _compute_scheduled_lr(
        base_lr: float,
        current_epoch: int,
        total_epochs: int,
        warmup_epochs: int,
        warmup_start_lr: float,
    ) -> float:
        half_epoch = total_epochs * 0.5
        three_quarter_epoch = total_epochs * 0.75

        # Phase 1 - Warmup: linearly increase from warmup_start_lr to base_lr.
        if current_epoch <= warmup_epochs:
            return warmup_start_lr + (base_lr - warmup_start_lr) * (current_epoch / max(1, warmup_epochs))

        # Phase 2 - Fixed: keep learning rate at base_lr until halfway point.
        if current_epoch < half_epoch:
            return base_lr

        # Phase 3 - First decay: reduce learning rate to base_lr / 10.
        if current_epoch < three_quarter_epoch:
            return base_lr / 10.0

        # Phase 4 - Second decay: reduce learning rate to base_lr / 100 until the end.
        return base_lr / 100.0

    @staticmethod
    def _get_schedule_phase(current_epoch: int, total_epochs: int, warmup_epochs: int) -> str:
        if total_epochs <= 0:
            return 'constant'

        half_epoch = total_epochs * 0.5
        three_quarter_epoch = total_epochs * 0.75

        if current_epoch <= warmup_epochs:
            return 'warmup'
        if current_epoch < half_epoch:
            return 'fixed'
        if current_epoch < three_quarter_epoch:
            return 'decay_1'
        return 'decay_2'
    
    def gossip_aggregate(self, weights: Dict[int, float]) -> float:
        """Aggregate model with neighbors using gossip or SSOS.
        
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
        aggregated_state = {
            key: torch.zeros_like(current_state[key], dtype=torch.float32)
            for key in current_state.keys()
        }
        
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

        # Debug: log aggregated state norm before momentum
        agg_vec = torch.cat([v.flatten().float() for v in aggregated_state.values()])
        agg_norm = agg_vec.norm(2).item()
        
        if self.ssos_enabled and self.prev_model is not None:
            accelerated_state = {}
            for key in current_state.keys():
                accelerated_state[key] = (
                    (1.0 + self.g) * aggregated_state[key].float()
                    - self.g * self.prev_model[key].float()
                )
            aggregated_state = accelerated_state
            
            # Debug: log momentum calculation
            accel_vec = torch.cat([v.flatten().float() for v in accelerated_state.values()])
            accel_norm = accel_vec.norm(2).item()
            momentum_contribution = accel_norm - agg_norm
            import os
            debug_mode = os.getenv('SSOS_DEBUG', '').strip().lower() in {'1', 'true', 'yes'}
            if debug_mode:
                print(f"[SSOS] Client {self.client_id}: "
                      f"g={self.g:.6f}, "
                      f"aggregated_norm={agg_norm:.6f}, "
                      f"accelerated_norm={accel_norm:.6f}, "
                      f"momentum_delta={momentum_contribution:+.6f}")
        elif self.ssos_enabled and self.prev_model is None:
            import os
            debug_mode = os.getenv('SSOS_DEBUG', '').strip().lower() in {'1', 'true', 'yes'}
            if debug_mode:
                print(f"[SSOS] Client {self.client_id}: "
                      f"NO momentum (prev_model=None, first gossip step), "
                      f"aggregated_norm={agg_norm:.6f}")
        
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

    def close(self) -> None:
        """Release per-round SSOS snapshot state."""
        self.prev_model = None
    
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

"""P2P client implementation for decentralized federated learning."""

import os
import tempfile
from pathlib import Path
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Tuple, Union, Any
import copy
import numpy as np


def _clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Clone tensors so buffered history is never mutated by later updates."""
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


class OutdatedAgreementAggregator:
    """Method 1: Outdated Agreement Feedback with per-node model buffers.

    Update rule:
        w_i(t+1) = alpha_ii * w_i(t) + sum_{j != i} a_ij * w_j(t-d)

    Notes:
      - Keeps the last d+1 states for each participant (self + neighbors).
      - For early rounds (t < d), uses the oldest available buffered state.
    """

    def __init__(self, local_id: int):
        self.local_id = local_id
        # node_id -> list of historical snapshot entries (oldest ... newest)
        self.state_buffer: Dict[int, List[Dict[str, Any]]] = {}
        self._step_counter = 0

        # By default, use disk-backed buffering to cap RAM usage for large models.
        to_disk_env = os.getenv('DELAY_BUFFER_TO_DISK', '1').strip().lower()
        self.use_disk_buffer = to_disk_env in {'1', 'true', 'yes', 'on'}

        root_env = os.getenv('DELAY_BUFFER_DIR', '').strip()
        if root_env:
            root = Path(root_env)
        else:
            root = Path(tempfile.gettempdir()) / 'flf_delay_buffer'
        self._buffer_root = root / f'client_{local_id}'
        if self.use_disk_buffer:
            self._buffer_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _state_scalar(state: Dict[str, torch.Tensor]) -> float:
        """Return one representative scalar for concise delay debug traces."""
        first_key = next(iter(state.keys()))
        flat = state[first_key].detach().float().view(-1)
        return float(flat[0].item()) if flat.numel() > 0 else 0.0

    def _save_state_to_disk(self, node_id: int, state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        node_dir = self._buffer_root / f'node_{node_id}'
        node_dir.mkdir(parents=True, exist_ok=True)
        self._step_counter += 1
        file_path = node_dir / f'step_{self._step_counter:08d}.pt'
        torch.save(state, file_path)
        return {
            'kind': 'disk',
            'path': str(file_path),
            'scalar': self._state_scalar(state),
        }

    def _make_entry(self, node_id: int, state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        cloned = _clone_state_dict(state)
        if self.use_disk_buffer:
            return self._save_state_to_disk(node_id, cloned)
        return {
            'kind': 'ram',
            'state': cloned,
            'scalar': self._state_scalar(cloned),
        }

    @staticmethod
    def _entry_state(entry: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if entry.get('kind') == 'disk':
            path = entry['path']
            try:
                return torch.load(path, map_location='cpu', weights_only=True)
            except TypeError:
                return torch.load(path, map_location='cpu')
        return entry['state']

    @staticmethod
    def _entry_scalar(entry: Dict[str, Any]) -> float:
        scalar = entry.get('scalar', None)
        if scalar is not None:
            return float(scalar)
        return 0.0

    @staticmethod
    def _delete_entry(entry: Dict[str, Any]) -> None:
        if entry.get('kind') == 'disk':
            path = entry.get('path', '')
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass

    def _append_state(self, node_id: int, state: Dict[str, torch.Tensor], delay: int) -> None:
        history = self.state_buffer.setdefault(node_id, [])
        history.append(self._make_entry(node_id, state))
        keep = max(delay + 1, 1)
        if len(history) > keep:
            stale = history[:-keep]
            for entry in stale:
                self._delete_entry(entry)
            del history[:-keep]

    def _get_delayed_entry(self, node_id: int, delay: int) -> Dict[str, Any]:
        history = self.state_buffer.get(node_id, [])
        if not history:
            raise KeyError(f"No buffered state for node {node_id}")
        if len(history) > delay:
            return history[-(delay + 1)]
        # For t < d, fall back to oldest available (initial buffered state).
        return history[0]

    def _get_delayed_state(self, node_id: int, delay: int) -> Dict[str, torch.Tensor]:
        entry = self._get_delayed_entry(node_id, delay)
        return self._entry_state(entry)

    def cleanup(self) -> None:
        for history in self.state_buffer.values():
            for entry in history:
                self._delete_entry(entry)
        self.state_buffer.clear()
        if self.use_disk_buffer:
            try:
                shutil.rmtree(self._buffer_root, ignore_errors=True)
            except OSError:
                pass

    def aggregate(
        self,
        local_model: nn.Module,
        neighbor_models: Union[Dict[int, Dict[str, torch.Tensor]], List[Tuple[int, Dict[str, torch.Tensor]]]],
        mixing_weights: Dict[int, float],
        delay: int,
        debug: bool = False,
        round_num: Optional[int] = None,
        gossip_step: Optional[int] = None,
        client_id: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate with delayed neighbor states and current local state.

        Args:
            local_model: Current local model.
            neighbor_models: Neighbor states as {neighbor_id: state_dict}
                or [(neighbor_id, state_dict), ...].
            mixing_weights: Mapping j -> W_ij for all contributors (including self).
            delay: Delay d for selecting state at t-d.

        Returns:
            Aggregated state_dict for w_i(t+1).
        """
        if delay < 0:
            raise ValueError("delay must be >= 0")

        local_state = _clone_state_dict(local_model.state_dict())
        self._append_state(self.local_id, local_state, delay)

        if isinstance(neighbor_models, list):
            neighbor_items = neighbor_models
        else:
            neighbor_items = list(neighbor_models.items())
        neighbor_lookup = dict(neighbor_items)

        for neighbor_id, neighbor_state in neighbor_items:
            self._append_state(neighbor_id, neighbor_state, delay)

        aggregated_state: Dict[str, torch.Tensor] = {
            k: torch.zeros_like(v, dtype=torch.float32) for k, v in local_state.items()
        }

        debug_chunks: List[str] = []

        for node_id, weight in mixing_weights.items():
            # Method 1 requested by user: self uses current state w_i(t),
            # neighbors use delayed states w_j(t-d).
            if node_id == self.local_id:
                source_state = local_state
                if debug:
                    debug_chunks.append(
                        f"self(node={node_id}):src=current,w={float(weight):.6f},"
                        f"value={self._state_scalar(source_state):.6f}"
                    )
            else:
                history = self.state_buffer.get(node_id, [])
                if len(history) > delay:
                    source_idx = -(delay + 1)
                    source_label = f"delayed(t-d),idx={source_idx}"
                    source_entry = history[source_idx]
                else:
                    source_label = "oldest_fallback(t<d)"
                    source_entry = history[0]
                source_state = self._entry_state(source_entry)
                if debug:
                    current_neighbor = _clone_state_dict(neighbor_lookup[node_id])
                    debug_chunks.append(
                        f"nbr(node={node_id}):src={source_label},w={float(weight):.6f},"
                        f"used={self._entry_scalar(source_entry):.6f},"
                        f"current={self._state_scalar(current_neighbor):.6f},"
                        f"hist_len={len(history)}"
                    )
            for key in aggregated_state.keys():
                aggregated_state[key] += source_state[key].float() * float(weight)

        if debug and debug_chunks:
            rid = round_num if round_num is not None else -1
            gid = gossip_step if gossip_step is not None else -1
            cid = client_id if client_id is not None else self.local_id
            line = (
                "[DELAY_DEBUG] "
                f"round={rid},gossip_step={gid},client={cid},d={delay} | "
                + " ; ".join(debug_chunks)
            )
            print(line)
            debug_file = os.getenv('DELAY_DEBUG_FILE', '').strip()
            if debug_file:
                try:
                    with open(debug_file, 'a', encoding='utf-8') as f:
                        f.write(line + "\n")
                except OSError:
                    pass

        # Cast back to local model dtypes to keep buffers/BN trackers valid.
        return {
            k: aggregated_state[k].to(dtype=local_state[k].dtype)
            for k in aggregated_state.keys()
        }


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
        # Buffered delayed aggregator (Method 1 from "Fast model averaging via buffered states").
        self.outdated_feedback = OutdatedAgreementAggregator(local_id=client_id)
    
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
    
    def train(self, epochs: int = 1, round_num: int = None) -> Dict[str, any]:
        """Train the local model.
        
        Args:
            epochs: Number of local training epochs
            round_num: Current round number
            
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
        last_grad_vec = None  # will hold the mean gradient vector over all batches of the last epoch
        
        # Per-class metrics
        num_classes = 10
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        
        for epoch in range(epochs):
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
            'last_grad_vec': last_grad_vec  # raw gradient vector: last batch of last epoch
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

    def gossip_aggregate_with_delay(
        self,
        weights: Dict[int, float],
        delay: int,
        round_num: Optional[int] = None,
        gossip_step: Optional[int] = None,
    ) -> float:
        """Aggregate model with delayed states w_j(t-d) using buffered history.

        Args:
            weights: Dictionary mapping node IDs to mixing weights W_ij.
            delay: Delay d in rounds/steps for outdated agreement feedback.

        Returns:
            weight_diff: L2 norm between pre/post aggregation parameter vectors.
        """
        if delay < 0:
            raise ValueError("delay must be >= 0")

        current_state = self.get_state()
        pre_vec = torch.cat([v.flatten().float() for v in current_state.values()])

        aggregated_state = self.outdated_feedback.aggregate(
            local_model=self.model,
            neighbor_models=self.neighbor_models,
            mixing_weights=weights,
            delay=delay,
            debug=os.getenv('DELAY_DEBUG', '').strip().lower() in {'1', 'true', 'yes', 'on'},
            round_num=round_num,
            gossip_step=gossip_step,
            client_id=self.client_id,
        )

        post_vec = torch.cat([v.flatten().float() for v in aggregated_state.values()])
        weight_diff = (post_vec - pre_vec).norm(2).item()

        self.set_state({k: v.to(device=self.device) for k, v in aggregated_state.items()})
        return weight_diff

    def close(self) -> None:
        """Release delayed-buffer resources (disk snapshots and in-memory index)."""
        self.outdated_feedback.cleanup()
    
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

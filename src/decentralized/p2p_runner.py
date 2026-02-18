"""P2P runner for decentralized federated learning."""

import torch
import time
from collections import defaultdict
from typing import List, Dict
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.decentralized.p2p_client import P2PClient
from src.decentralized.topology import (
    get_active_edges,
    get_active_mixing_matrix,
    MixingMethod
)
import networkx as nx


class P2PRunner:
    """Runner for peer-to-peer federated learning."""
    
    def __init__(
        self,
        clients: List[P2PClient],
        graph: nx.Graph,
        logger = None,
        seed: int = 42,
        mixing_method: MixingMethod = 'metropolis_hastings',
        gossip_steps: int = 1
    ):
        """Initialize P2P runner.
        
        Args:
            clients: List of P2P clients
            graph: Network topology graph
            logger: Comprehensive metrics logger
            seed: Random seed for reproducibility
            mixing_method: Mixing method for gossip aggregation
            gossip_steps: Number of gossip iterations per round (before next training)
        """
        self.clients = clients
        self.graph = graph
        self.logger = logger
        self.seed = seed
        self.mixing_method = mixing_method
        self.gossip_steps = gossip_steps
        self.num_clients = len(clients)
        
        # Compute cluster assignments (for two-cluster topology)
        self.cluster_assignments = self._compute_clusters()
        
        print(f"P2P Runner initialized with mixing method: {mixing_method}, gossip_steps: {gossip_steps}")
    
    def save_topology_visualization(self, output_dir: str, experiment_name: str = "p2p_topology"):
        """Save network topology as interactive HTML.
        
        Args:
            output_dir: Directory to save the visualization
            experiment_name: Name for the output file
        """
        from pathlib import Path
        from src.utils.visualization import plot_network_topology, save_topology_info
        
        output_path = Path(output_dir) / experiment_name
        
        # Save interactive HTML visualization
        plot_network_topology(
            graph=self.graph,
            output_path=str(output_path),
            title=f"P2P Network Topology - {self.mixing_method}",
            cluster_assignments=self.cluster_assignments
        )
        
        # Save topology information
        save_topology_info(
            graph=self.graph,
            output_dir=Path(output_dir),
            cluster_assignments=self.cluster_assignments
        )
        
    def _compute_clusters(self) -> Dict[int, int]:
        """Compute cluster assignment for each client (for two-cluster topology).
        
        Returns:
            Dictionary mapping client_id to cluster_id (0 or 1)
        """
        # Simple heuristic: first half in cluster 0, second half in cluster 1
        clusters = {}
        for i in range(self.num_clients):
            clusters[i] = 0 if i < self.num_clients // 2 else 1
        return clusters
    
    def train_round(self, round_num: int, local_epochs: int = 1) -> Dict[str, any]:
        """Execute one round of P2P federated learning.
        
        Args:
            round_num: Current round number
            local_epochs: Number of local training epochs
            
        Returns:
            Dictionary with round metrics
        """
        print(f"\n=== Round {round_num} ===")
        
        # Store previous gradients for computing gradient changes
        prev_gradients = {}
        if round_num > 1:
            for client in self.clients:
                if hasattr(client, 'prev_gradient_norm'):
                    prev_gradients[client.client_id] = client.prev_gradient_norm
        
        # Phase 1: Local training
        print("Phase 1: Local training...")
        client_states = {}
        client_losses = []
        client_accuracies = []
        gradient_norms_list = []
        
        # Determine number of GPUs in use
        unique_devices = list(set(c.device for c in self.clients))
        num_workers = len(unique_devices)
        
        def _train_p2p_client(client):
            """Train a single P2P client and return results."""
            metrics = client.train(epochs=local_epochs)
            state = client.get_state()
            
            grad_norm = metrics['gradient_norms'][-1] if metrics['gradient_norms'] else 0.0
            gradient_change = 0.0
            if client.client_id in prev_gradients:
                gradient_change = abs(grad_norm - prev_gradients[client.client_id])
            client.prev_gradient_norm = grad_norm
            
            test_metrics = None
            if self.logger:
                test_metrics = client.evaluate(compute_per_class_metrics=True)
            
            return {
                'client_id': client.client_id,
                'state': state,
                'final_loss': metrics['final_loss'],
                'final_accuracy': metrics['final_accuracy'],
                'grad_norm': grad_norm,
                'gradient_change': gradient_change,
                'test_metrics': test_metrics
            }
        
        if num_workers > 1:
            # Group clients by device so each GPU is used by exactly one thread
            gpu_groups = defaultdict(list)
            for client in self.clients:
                gpu_groups[str(client.device)].append(client)
            
            def _train_group(group_clients):
                group_results = {}
                for client in group_clients:
                    result = _train_p2p_client(client)
                    group_results[result['client_id']] = result
                return group_results
            
            print(f"Training {len(self.clients)} clients in parallel across {len(gpu_groups)} GPU(s)...")
            with ThreadPoolExecutor(max_workers=len(gpu_groups)) as executor:
                futures = [
                    executor.submit(_train_group, group)
                    for group in gpu_groups.values()
                ]
                results = {}
                for future in as_completed(futures):
                    results.update(future.result())
        else:
            results = {}
            for client in self.clients:
                results[client.client_id] = _train_p2p_client(client)
        
        # Collect results in order
        for client in self.clients:
            result = results[client.client_id]
            client_states[client.client_id] = result['state']
            client_losses.append(result['final_loss'])
            client_accuracies.append(result['final_accuracy'])
            gradient_norms_list.append(result['grad_norm'])
            
            print(f"Client {client.client_id}: Loss={result['final_loss']:.4f}, Acc={result['final_accuracy']:.2f}%")
        
        # Phase 2: Gossip aggregation (repeated gossip_steps times)
        print(f"Phase 2: Gossip aggregation ({self.gossip_steps} step(s))...")
        
        communication_start = time.time()
        
        # Accumulate total weight diff across all gossip steps
        weight_diffs = {c.client_id: 0.0 for c in self.clients}
        
        for gossip_step in range(self.gossip_steps):
            # Determine active edges for this round
            active_edges = get_active_edges(self.graph, round_num, self.seed)
            
            # Create mixing matrix based on active edges and mixing method
            W = get_active_mixing_matrix(
                self.graph,
                self.num_clients,
                active_edges,
                method=self.mixing_method
            )
            
            # Get fresh client states (updated after each gossip step)
            if gossip_step > 0:
                for client in self.clients:
                    client_states[client.client_id] = client.get_state()
            
            # Share models with neighbors
            for client in self.clients:
                neighbors = list(self.graph.neighbors(client.client_id))
                active_neighbors = [n for n in neighbors if active_edges.get((client.client_id, n), False)]
                
                # Collect neighbor models
                neighbor_states = {}
                for neighbor_id in active_neighbors:
                    neighbor_states[neighbor_id] = client_states[neighbor_id]
                
                # Store neighbor models
                client.store_neighbor_models(neighbor_states)
            
            # Perform gossip aggregation and accumulate weight differences
            for client in self.clients:
                weights = {}
                for neighbor_id in range(self.num_clients):
                    if W[client.client_id, neighbor_id] > 0:
                        weights[neighbor_id] = W[client.client_id, neighbor_id]
                
                step_weight_diff = client.gossip_aggregate(weights)
                weight_diffs[client.client_id] += step_weight_diff
        
        communication_time = time.time() - communication_start
        print(f"Communication took {communication_time:.2f}s")
        
        # Log metrics (after gossip, so weight_diff is available)
        if self.logger:
            for client in self.clients:
                result = results[client.client_id]
                if result['test_metrics']:
                    cluster_id = self.cluster_assignments.get(client.client_id)
                    self.logger.log_p2p_round_metrics(
                        client_id=client.client_id,
                        round_num=round_num,
                        test_accuracy=result['test_metrics']['accuracy'],
                        test_loss=result['test_metrics']['loss'],
                        gradient_norm=result['grad_norm'],
                        gradient_change=result['gradient_change'],
                        class_metrics=result['test_metrics'].get('class_metrics', {}),
                        cluster_id=cluster_id,
                        train_accuracy=result['final_accuracy'],
                        train_loss=result['final_loss'],
                        weight_diff=weight_diffs.get(client.client_id, 0.0)
                    )
        
        # Phase 3: Evaluation
        print("Phase 3: Evaluation...")
        eval_losses = []
        eval_accuracies = []
        
        for client in self.clients:
            eval_metrics = client.evaluate()
            eval_losses.append(eval_metrics['loss'])
            eval_accuracies.append(eval_metrics['accuracy'])
        
        avg_loss = np.mean(eval_losses)
        avg_accuracy = np.mean(eval_accuracies)
        std_accuracy = np.std(eval_accuracies)
        
        print(f"Average - Loss: {avg_loss:.4f}, Acc: {avg_accuracy:.2f}% (±{std_accuracy:.2f}%)")
        
        return {
            'round': round_num,
            'client_losses': client_losses,
            'client_accuracies': client_accuracies,
            'eval_losses': eval_losses,
            'eval_accuracies': eval_accuracies,
            'avg_loss': avg_loss,
            'avg_accuracy': avg_accuracy,
            'gradient_norms': gradient_norms_list
        }
    
    def train(self, num_rounds: int, local_epochs: int = 1):
        """Train for multiple rounds.
        
        Args:
            num_rounds: Number of federated rounds
            local_epochs: Local epochs per round
        """
        for round_num in range(1, num_rounds + 1):
            self.train_round(round_num, local_epochs)
        
        print("\n=== Training Complete ===")
        
        # Save final client weights for comparison
        if self.logger:
            self.logger.save_client_final_weights(self.clients)

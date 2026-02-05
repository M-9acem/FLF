"""P2P runner for decentralized federated learning."""

import torch
import time
from typing import List, Dict
import numpy as np
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
        mixing_method: MixingMethod = 'metropolis_hastings'
    ):
        """Initialize P2P runner.
        
        Args:
            clients: List of P2P clients
            graph: Network topology graph
            logger: Comprehensive metrics logger
            seed: Random seed for reproducibility
            mixing_method: Mixing method for gossip aggregation
        """
        self.clients = clients
        self.graph = graph
        self.logger = logger
        self.seed = seed
        self.mixing_method = mixing_method
        self.num_clients = len(clients)
        
        # Compute cluster assignments (for two-cluster topology)
        self.cluster_assignments = self._compute_clusters()
        
        print(f"P2P Runner initialized with mixing method: {mixing_method}")
    
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
        
        for client in self.clients:
            metrics = client.train(epochs=local_epochs)
            client_states[client.client_id] = client.get_state()
            client_losses.append(metrics['final_loss'])
            client_accuracies.append(metrics['final_accuracy'])
            
            # Get gradient norm
            grad_norm = metrics['gradient_norms'][-1]
            gradient_norms_list.append(grad_norm)
            
            # Compute gradient change if available
            gradient_change = 0.0
            if client.client_id in prev_gradients:
                gradient_change = abs(grad_norm - prev_gradients[client.client_id])
            
            # Store current gradient for next round
            client.prev_gradient_norm = grad_norm
            
            # SIMPLIFIED LOGGING: Only log what was requested
            if self.logger:
                # 1. Accuracy of each client overall for each round (test accuracy)
                test_metrics = client.evaluate(compute_per_class_metrics=True)
                test_accuracy = test_metrics['accuracy']
                test_loss = test_metrics['loss']
                
                # 2. Accuracy of each client per class for each round
                class_metrics = test_metrics.get('class_metrics', {})
                
                # 3. Loss per round of each client (test loss)
                # 4. Gradient norm of each client per round
                # 5. Gradient changes per round of each client
                
                # Get cluster assignment for this client
                cluster_id = self.cluster_assignments.get(client.client_id)
                
                # Log to CSV
                self.logger.log_p2p_round_metrics(
                    client_id=client.client_id,
                    round_num=round_num,
                    test_accuracy=test_accuracy,
                    test_loss=test_loss,
                    gradient_norm=grad_norm,
                    gradient_change=gradient_change,
                    class_metrics=class_metrics,
                    cluster_id=cluster_id
                )
            
            print(f"Client {client.client_id}: Loss={metrics['final_loss']:.4f}, Acc={metrics['final_accuracy']:.2f}%")
        
        # Phase 2: Gossip aggregation
        print("Phase 2: Gossip aggregation...")
        
        # Determine active edges for this round
        active_edges = get_active_edges(self.graph, round_num, self.seed)
        
        # Create mixing matrix based on active edges and mixing method
        W = get_active_mixing_matrix(
            self.graph,
            self.num_clients,
            active_edges,
            method=self.mixing_method
        )
        
        # Share models with neighbors (simplified - no detailed propagation logging)
        communication_start = time.time()
        for client in self.clients:
            neighbors = list(self.graph.neighbors(client.client_id))
            active_neighbors = [n for n in neighbors if active_edges.get((client.client_id, n), False)]
            
            # Collect neighbor models
            neighbor_states = {}
            for neighbor_id in active_neighbors:
                neighbor_states[neighbor_id] = client_states[neighbor_id]
            
            # Store neighbor models
            client.store_neighbor_models(neighbor_states)
        
        communication_time = time.time() - communication_start
        print(f"Communication took {communication_time:.2f}s")
        
        # Perform gossip aggregation
        for client in self.clients:
            # Extract weights for this client
            weights = {}
            for neighbor_id in range(self.num_clients):
                if W[client.client_id, neighbor_id] > 0:
                    weights[neighbor_id] = W[client.client_id, neighbor_id]
            
            client.gossip_aggregate(weights)
        
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

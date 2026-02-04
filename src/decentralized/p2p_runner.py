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
            gradient_norms_list.append(metrics['gradient_norms'][-1])
            
            # Log per-client metrics with comprehensive logger
            if self.logger:
                for epoch_idx, (loss, acc, grad_norm) in enumerate(
                    zip(metrics['losses'], metrics['accuracies'], metrics['gradient_norms'])
                ):
                    self.logger.log_per_client_metrics(
                        client_id=client.client_id,
                        round_num=round_num,
                        epoch=epoch_idx,
                        train_loss=loss,
                        train_accuracy=acc
                    )
                
                # Evaluate and log per-class metrics
                test_metrics = client.evaluate(compute_per_class_metrics=True)
                if 'class_metrics' in test_metrics:
                    self.logger.log_per_class_metrics(
                        client_id=client.client_id,
                        round_num=round_num,
                        class_metrics=test_metrics['class_metrics']
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
        
        # Share models with neighbors and log comprehensive propagation metrics
        communication_start = time.time()
        for client in self.clients:
            neighbors = list(self.graph.neighbors(client.client_id))
            active_neighbors = [n for n in neighbors if active_edges.get((client.client_id, n), False)]
            
            # Collect neighbor models
            neighbor_states = {}
            for neighbor_id in active_neighbors:
                neighbor_states[neighbor_id] = client_states[neighbor_id]
                
                # Log comprehensive propagation metrics
                if self.logger:
                    # Simulate propagation delay (ms)
                    prop_delay = np.random.exponential(15.0)  
                    
                    # Compute hop count (shortest path in graph)
                    try:
                        hop_count = nx.shortest_path_length(self.graph, client.client_id, neighbor_id)
                    except:
                        hop_count = 1
                    
                    # Check if inter-cluster communication
                    source_cluster = self.cluster_assignments.get(client.client_id, 0)
                    dest_cluster = self.cluster_assignments.get(neighbor_id, 0)
                    inter_cluster = (source_cluster != dest_cluster)
                    
                    # Estimate model size in bytes (rough estimate)
                    bytes_transferred = sum(p.numel() * 4 for p in client.model.parameters())  # 4 bytes per float32
                    
                    self.logger.log_propagation_metrics(
                        round_num=round_num,
                        client_id=client.client_id,
                        source_client_id=client.client_id,
                        destination_client_id=neighbor_id,
                        propagation_delay=prop_delay,
                        hop_count=hop_count,
                        cluster_id=source_cluster,
                        inter_cluster_communication=inter_cluster,
                        model_version=round_num,
                        bytes_transferred=bytes_transferred
                    )
            
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
        
        # Log round summary
        if self.logger:
            self.logger.log_round_summary(
                round_num,
                eval_accuracies,
                eval_losses,
                gradient_norms_list
            )
        
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
        # Log data distribution before training
        if self.logger:
            print("\nComputing initial data distribution...")
            for client in self.clients:
                # Get class distribution from client's training data
                class_counts = {}
                for _, labels in client.train_loader:
                    for label in labels:
                        label_item = label.item()
                        class_counts[label_item] = class_counts.get(label_item, 0) + 1
                
                total_samples = sum(class_counts.values())
                class_dist = {k: v / total_samples for k, v in class_counts.items()}
                
                # Compute heterogeneity score (KL divergence from uniform)
                num_classes = len(class_dist)
                uniform_prob = 1.0 / num_classes
                kl_divergence = sum(p * np.log(p / uniform_prob) for p in class_dist.values() if p > 0)
                
                self.logger.log_data_distribution(
                    client_id=client.client_id,
                    round_num=0,
                    class_distribution=class_dist,
                    total_samples=total_samples,
                    heterogeneity_score=kl_divergence
                )
        
        for round_num in range(1, num_rounds + 1):
            self.train_round(round_num, local_epochs)
        
        print("\n=== Training Complete ===")

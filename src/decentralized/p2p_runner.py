"""P2P runner for decentralized federated learning."""

import torch
import torch.nn as nn
import copy
import time
import os
from typing import List, Dict, Tuple
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
        gossip_steps: int = 1,
        gossip_schedule: List[tuple] = None,
        delay_d: int = 0,
        client_parallelism: int = 8,
        save_full_gradients: bool = False,
        save_pre_gossip_weights: bool = False,
        save_client_final_weights: bool = True,
    ):
        """Initialize P2P runner.
        
        Args:
            clients: List of P2P clients
            graph: Network topology graph
            logger: Comprehensive metrics logger
            seed: Random seed for reproducibility
            mixing_method: Mixing method for gossip aggregation
            gossip_steps: Default number of gossip steps per round (used when no schedule)
            gossip_schedule: Optional list of (from_round, steps) tuples defining a
                gossip drop schedule, e.g. [(0,5),(100,3),(200,1)] means 5 steps
                until round 100, 3 until round 200, then 1 for remaining rounds.
                Overrides gossip_steps when provided.
            delay_d: Delayed aggregation depth d. If d>0, use
                w_i(t+1)=alpha_ii*w_i(t)+sum_{j!=i} a_ij*w_j(t-d).
        """
        self.clients = clients
        self.graph = graph
        self.logger = logger
        self.seed = seed
        self.mixing_method = mixing_method
        self.gossip_steps = gossip_steps
        self.delay_d = max(0, int(delay_d))
        self.client_parallelism = max(1, int(client_parallelism))
        self.save_full_gradients = bool(save_full_gradients)
        self.save_pre_gossip_weights = bool(save_pre_gossip_weights)
        self.save_client_final_weights = bool(save_client_final_weights)
        # Sort schedule by from_round ascending
        self.gossip_schedule = sorted(gossip_schedule, key=lambda x: x[0]) if gossip_schedule else None
        self.num_clients = len(clients)
        
        # Compute cluster assignments (for two-cluster topology)
        self.cluster_assignments = self._compute_clusters()
        self.metric_pairs = self._build_weight_diff_metric_pairs()
        
        if self.gossip_schedule:
            schedule_str = ', '.join(f'{s} steps from round {r}' for r, s in self.gossip_schedule)
            print(f"P2P Runner initialized with mixing method: {mixing_method}, gossip schedule: [{schedule_str}]")
        else:
            print(f"P2P Runner initialized with mixing method: {mixing_method}, gossip_steps: {gossip_steps}")
        if self.delay_d > 0:
            print(f"Delayed aggregation enabled: d={self.delay_d}")
            if self.logger is not None:
                delay_buffer_dir = os.path.join(self.logger.get_log_dir(), 'delay_buffer')
                os.environ['DELAY_BUFFER_DIR'] = delay_buffer_dir
                print(f"Delay buffer directory: {delay_buffer_dir}")
            delay_dbg = os.getenv('DELAY_DEBUG', '').strip().lower() in {'1', 'true', 'yes', 'on'}
            if delay_dbg and self.logger is not None:
                delay_log = os.path.join(self.logger.get_log_dir(), 'delay_debug.log')
                os.environ['DELAY_DEBUG_FILE'] = delay_log
                print(f"Delay debug log file: {delay_log}")

        if self.metric_pairs:
            print("Post-gossip weight-diff metrics configured:")
            for name, (i, j) in self.metric_pairs.items():
                print(f"  - {name}: ({i}, {j})")
    
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

    def _cluster_nodes(self, cluster_id: int) -> List[int]:
        return sorted([n for n, c in self.cluster_assignments.items() if c == cluster_id])

    def _find_central_nodes(self) -> Tuple[int, int]:
        """Find central bridge nodes that connect the two clusters.

        Preference: endpoints of an inter-cluster edge.
        Fallback: highest-degree node in each cluster.
        """
        bridge_edges = []
        for u, v in self.graph.edges():
            if self.cluster_assignments[u] != self.cluster_assignments[v]:
                bridge_edges.append((u, v))

        if bridge_edges:
            # Choose deterministically using sorted edge list by node ids
            u, v = sorted([tuple(sorted(e)) for e in bridge_edges])[0]
            if self.cluster_assignments[u] == 0:
                return u, v
            return v, u

        c0_nodes = self._cluster_nodes(0)
        c1_nodes = self._cluster_nodes(1)
        c0 = max(c0_nodes, key=lambda n: self.graph.degree(n))
        c1 = max(c1_nodes, key=lambda n: self.graph.degree(n))
        return c0, c1

    @staticmethod
    def _pick_distinct(nodes: List[int], exclude: List[int], k: int) -> List[int]:
        chosen = [n for n in nodes if n not in set(exclude)]
        if len(chosen) >= k:
            return chosen[:k]
        # Fallback if the cluster is small.
        for n in nodes:
            if n not in chosen:
                chosen.append(n)
            if len(chosen) == k:
                break
        return chosen

    def _build_weight_diff_metric_pairs(self) -> Dict[str, Tuple[int, int]]:
        """Build the 8 node pairs used for post-gossip weight-diff metrics."""
        c1, c2 = self._find_central_nodes()  # c1 in cluster 0, c2 in cluster 1
        cl0 = self._cluster_nodes(0)
        cl1 = self._cluster_nodes(1)

        u1 = self._pick_distinct(cl0, [c1], 1)[0]
        u2 = self._pick_distinct(cl1, [c2], 1)[0]

        a1, b1 = self._pick_distinct(cl0, [], 2)
        a2, b2 = self._pick_distinct(cl1, [], 2)

        x1 = self._pick_distinct(cl1, [c2], 1)[0]
        x2 = self._pick_distinct(cl0, [c1], 1)[0]

        p1 = self._pick_distinct(cl0, [c1], 1)[0]
        p2 = self._pick_distinct(cl1, [c2], 1)[0]

        return {
            'm1_central_bridge': (c1, c2),
            'm2_central1_inner1': (c1, u1),
            'm3_central2_inner2': (c2, u2),
            'm4_intra_cluster1_pair': (a1, b1),
            'm5_intra_cluster2_pair': (a2, b2),
            'm6_central1_to_cluster2': (c1, x1),
            'm7_central2_to_cluster1': (c2, x2),
            'm8_cross_cluster_pair': (p1, p2),
        }

    def _compute_pair_weight_diffs(self) -> Dict[str, float]:
        """Compute L2 differences for configured node pairs using post-gossip weights."""
        state_vectors: Dict[int, torch.Tensor] = {}
        for client in self.clients:
            state = client.get_state()
            state_vectors[client.client_id] = torch.cat([v.flatten().float() for v in state.values()])

        values: Dict[str, float] = {}
        for metric_name, (i, j) in self.metric_pairs.items():
            values[metric_name] = (state_vectors[i] - state_vectors[j]).norm(2).item()
        return values
    
    def train_round(
        self,
        round_num: int,
        local_epochs: int = 1,
        total_epochs: int = 0,
        use_lr_schedule: bool = False,
        warmup_epochs: int = 5,
        warmup_start_lr: float = 0.001,
    ) -> Dict[str, any]:
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
        
        # Determine the concurrency level for local client training.
        num_workers = min(self.client_parallelism, len(self.clients))
        
        def _train_p2p_client(client):
            """Train a single P2P client and return results."""
            global_epoch_start = (round_num - 1) * local_epochs
            metrics = client.train(
                epochs=local_epochs,
                round_num=round_num,
                total_epochs=total_epochs,
                global_epoch_start=global_epoch_start,
                use_lr_schedule=use_lr_schedule,
                warmup_epochs=warmup_epochs,
                warmup_start_lr=warmup_start_lr,
            )
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
                'current_lr': metrics.get('current_lr'),
                'lr_history': metrics.get('lr_history', []),
                'grad_norm': grad_norm,
                'gradient_change': gradient_change,
                'test_metrics': test_metrics,
                'last_grad_vec': metrics.get('last_grad_vec'),  # raw gradient vector
                'num_samples': metrics.get('num_samples')
            }
        
        if num_workers > 1:
            print(
                f"Training {len(self.clients)} clients in parallel with "
                f"{num_workers} worker(s)..."
            )
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(_train_p2p_client, client) for client in self.clients]
                results = {}
                for future in as_completed(futures):
                    result = future.result()
                    results[result['client_id']] = result
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
        
        # Save full per-client gradient vectors only when explicitly requested.
        if self.logger:
            if self.save_full_gradients:
                all_grad_dir = os.path.join(self.logger.get_log_dir(), 'all_gradients')
                os.makedirs(all_grad_dir, exist_ok=True)
                for client in self.clients:
                    grad_vec = results[client.client_id].get('last_grad_vec')
                    if grad_vec is not None:
                        npy_path = os.path.join(all_grad_dir, f'client_{client.client_id}_round_{round_num}.npy')
                        np.save(npy_path, grad_vec)
        
        # Save pre-gossip weights and test metrics (before Phase 2)
        if self.logger:
            pre_gossip_states = {c.client_id: results[c.client_id]['state'] for c in self.clients}
            if self.save_pre_gossip_weights:
                self.logger.save_pre_gossip_weights(pre_gossip_states, round_num)
            num_samples_map = {c.client_id: results[c.client_id].get('num_samples') or 1
                               for c in self.clients}
            total_samples = sum(num_samples_map.values())

            def _evaluate_weighted_average_model(state_map):
                avg_state = {}
                for key in state_map[self.clients[0].client_id].keys():
                    avg_state[key] = sum(
                        state_map[cid][key].float() * (num_samples_map[cid] / total_samples)
                        for cid in num_samples_map
                    )

                tmp_model = copy.deepcopy(self.clients[0].model)
                tmp_model.load_state_dict(avg_state)
                tmp_model.eval()
                criterion = nn.CrossEntropyLoss()
                total_loss, correct, total = 0.0, 0, 0
                num_classes = 10
                class_correct = [0] * num_classes
                class_total = [0] * num_classes
                all_targets, all_preds = [], []
                with torch.no_grad():
                    for data, target in self.clients[0].test_loader:
                        data, target = data.to(self.clients[0].device), target.to(self.clients[0].device)
                        output = tmp_model(data)
                        total_loss += criterion(output, target).item()
                        pred = output.argmax(dim=1)
                        correct += pred.eq(target).sum().item()
                        total += target.size(0)
                        all_targets.extend(target.cpu().numpy())
                        all_preds.extend(pred.cpu().numpy())
                        for i in range(len(target)):
                            lbl = target[i].item()
                            class_correct[lbl] += (pred[i] == target[i]).item()
                            class_total[lbl] += 1

                avg_acc = 100.0 * correct / total
                avg_loss = total_loss / len(self.clients[0].test_loader)
                class_metrics = {}
                for cls in range(num_classes):
                    if class_total[cls] > 0:
                        from sklearn.metrics import precision_score, recall_score, f1_score
                        import numpy as _np
                        t = _np.array(all_targets)
                        p = _np.array(all_preds)
                        class_metrics[cls] = {
                            'accuracy': 100.0 * class_correct[cls] / class_total[cls],
                            'precision': precision_score(t == cls, p == cls, zero_division=0),
                            'recall': recall_score(t == cls, p == cls, zero_division=0),
                            'f1_score': f1_score(t == cls, p == cls, zero_division=0),
                        }
                    else:
                        class_metrics[cls] = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}

                del tmp_model
                return avg_acc, avg_loss, class_metrics

            for client in self.clients:
                result = results[client.client_id]
                if result['test_metrics']:
                    cluster_id = self.cluster_assignments.get(client.client_id)
                    self.logger.log_pre_gossip_metrics(
                        client_id=client.client_id,
                        round_num=round_num,
                        test_accuracy=result['test_metrics']['accuracy'],
                        test_loss=result['test_metrics']['loss'],
                        class_metrics=result['test_metrics'].get('class_metrics', {}),
                        cluster_id=cluster_id,
                        train_accuracy=result['final_accuracy'],
                        train_loss=result['final_loss'],
                        num_samples=result.get('num_samples'),
                        current_lr=result.get('current_lr')
                    )
                    self.logger.log_lr_schedule_metrics(
                        mode='decentralized',
                        stage='pre_gossip',
                        client_id=client.client_id,
                        round_num=round_num,
                        lr_history=result.get('lr_history', [])
                    )
            # Log scalar weighted gradient sum norm per round to reduce disk usage.
            grad_weighted_sum = None
            for cid, n_samples in num_samples_map.items():
                grad_vec = results[cid].get('last_grad_vec')
                if grad_vec is None:
                    grad_weighted_sum = None
                    break
                coeff = float(n_samples) / float(total_samples)
                contribution = coeff * grad_vec
                grad_weighted_sum = contribution if grad_weighted_sum is None else grad_weighted_sum + contribution

            if grad_weighted_sum is not None:
                weighted_grad_sum_norm_l2 = float(np.linalg.norm(grad_weighted_sum, ord=2))
                self.logger.log_weighted_gradient_sum_norm(
                    round_num=round_num,
                    weighted_grad_sum_norm_l2=weighted_grad_sum_norm_l2,
                )

            pre_avg_acc, pre_avg_loss, pre_class_metrics = _evaluate_weighted_average_model(pre_gossip_states)
            self.logger.log_global_aggregated_metrics(
                round_num=round_num,
                test_accuracy=pre_avg_acc,
                test_loss=pre_avg_loss,
                class_metrics=pre_class_metrics,
                total_samples=total_samples,
                gossip_step=-1,
            )

        # Resolve effective gossip steps for this round (schedule or fixed)
        if self.gossip_schedule:
            effective_gossip_steps = self.gossip_schedule[0][1]  # default: first entry
            for from_round, steps in self.gossip_schedule:
                if round_num >= from_round:
                    effective_gossip_steps = steps
        else:
            effective_gossip_steps = self.gossip_steps

        # Phase 2: Gossip aggregation (repeated effective_gossip_steps times)
        print(f"Phase 2: Gossip aggregation ({effective_gossip_steps} step(s))...")
        
        communication_start = time.time()
        
        # Accumulate total weight diff across all gossip steps
        weight_diffs = {c.client_id: 0.0 for c in self.clients}
        pair_weight_diffs = {}
        eval_losses = []
        eval_accuracies = []
        avg_loss = 0.0
        avg_accuracy = 0.0
        std_accuracy = 0.0
        
        for gossip_step in range(effective_gossip_steps):
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

                if self.delay_d > 0:
                    step_weight_diff = client.gossip_aggregate_with_delay(
                        weights,
                        self.delay_d,
                        round_num=round_num,
                        gossip_step=gossip_step,
                    )
                else:
                    step_weight_diff = client.gossip_aggregate(weights)
                weight_diffs[client.client_id] += step_weight_diff

            if self.logger:
                pair_weight_diffs = self._compute_pair_weight_diffs()
                self.logger.log_p2p_pair_weight_diffs(
                    round_num=round_num,
                    gossip_step=gossip_step,
                    metric_values=pair_weight_diffs,
                    metric_pairs=self.metric_pairs,
                )

                current_states = {c.client_id: c.get_state() for c in self.clients}
                avg_acc, avg_loss, class_metrics = _evaluate_weighted_average_model(current_states)
                self.logger.log_global_aggregated_metrics(
                    round_num=round_num,
                    test_accuracy=avg_acc,
                    test_loss=avg_loss,
                    class_metrics=class_metrics,
                    total_samples=total_samples,
                    gossip_step=gossip_step,
                )

                step_eval_losses = []
                step_eval_accuracies = []
                for client in self.clients:
                    post_gossip_metrics = client.evaluate(compute_per_class_metrics=True)
                    step_eval_losses.append(post_gossip_metrics['loss'])
                    step_eval_accuracies.append(post_gossip_metrics['accuracy'])

                    result = results[client.client_id]
                    cluster_id = self.cluster_assignments.get(client.client_id)
                    self.logger.log_p2p_round_metrics(
                        client_id=client.client_id,
                        round_num=round_num,
                        test_accuracy=post_gossip_metrics['accuracy'],
                        test_loss=post_gossip_metrics['loss'],
                        class_metrics=post_gossip_metrics.get('class_metrics', {}),
                        cluster_id=cluster_id,
                        train_accuracy=result['final_accuracy'],
                        train_loss=result['final_loss'],
                        num_samples=result.get('num_samples'),
                        current_lr=result.get('current_lr'),
                        gossip_step=gossip_step,
                    )

                eval_losses = step_eval_losses
                eval_accuracies = step_eval_accuracies
                avg_loss = np.mean(eval_losses)
                avg_accuracy = np.mean(eval_accuracies)
                std_accuracy = np.std(eval_accuracies)

                print(f"  Gossip step {gossip_step + 1}/{effective_gossip_steps}: Average - Loss: {avg_loss:.4f}, Acc: {avg_accuracy:.2f}% (±{std_accuracy:.2f}%)")

        communication_time = time.time() - communication_start
        print(f"Communication took {communication_time:.2f}s")

        if not self.logger:
            print("Phase 3: Evaluation...")
            for client in self.clients:
                post_gossip_metrics = client.evaluate(compute_per_class_metrics=False)
                eval_losses.append(post_gossip_metrics['loss'])
                eval_accuracies.append(post_gossip_metrics['accuracy'])

            avg_loss = np.mean(eval_losses)
            avg_accuracy = np.mean(eval_accuracies)
            std_accuracy = np.std(eval_accuracies)

            print(f"Average - Loss: {avg_loss:.4f}, Acc: {avg_accuracy:.2f}% (±{std_accuracy:.2f}%)")
        elif eval_losses:
            print(f"Average - Loss: {avg_loss:.4f}, Acc: {avg_accuracy:.2f}% (±{std_accuracy:.2f}%)")
        
        return {
            'round': round_num,
            'client_losses': client_losses,
            'client_accuracies': client_accuracies,
            'eval_losses': eval_losses,
            'eval_accuracies': eval_accuracies,
            'avg_loss': avg_loss,
            'avg_accuracy': avg_accuracy,
            'gradient_norms': gradient_norms_list,
            'pair_weight_diffs': pair_weight_diffs,
        }
    
    def train(
        self,
        num_rounds: int,
        local_epochs: int = 1,
        use_lr_schedule: bool = False,
        warmup_epochs: int = 5,
        warmup_start_lr: float = 0.001,
    ):
        """Train for multiple rounds.
        
        Args:
            num_rounds: Number of federated rounds
            local_epochs: Local epochs per round
        """
        total_epochs = num_rounds * local_epochs

        for round_num in range(1, num_rounds + 1):
            self.train_round(
                round_num,
                local_epochs,
                total_epochs=total_epochs,
                use_lr_schedule=use_lr_schedule,
                warmup_epochs=warmup_epochs,
                warmup_start_lr=warmup_start_lr,
            )
        
        print("\n=== Training Complete ===")

        # Release delayed-buffer files after training to avoid disk buildup.
        if self.delay_d > 0:
            for client in self.clients:
                client.close()
        
        # Save final client weights for comparison
        if self.logger and self.save_client_final_weights:
            self.logger.save_client_final_weights(self.clients)
            self.logger.plot_p2p_pair_weight_diffs()

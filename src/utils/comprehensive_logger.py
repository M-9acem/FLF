"""Comprehensive metrics logging system for federated learning experiments with all requested metrics."""

import json
import csv
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
import torch


class ComprehensiveLogger:
    """Ultra-comprehensive logger tracking all FL metrics as requested."""
    
    def __init__(self, log_dir: str, experiment_name: str = "experiment"):
        """Initialize the comprehensive metrics logger.
        
        Args:
            log_dir: Base directory for logs
            experiment_name: Name of the experiment
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.exp_dir = Path(log_dir) / experiment_name / timestamp
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.plots_dir = self.exp_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Initialize all log files
        self._init_log_files()
        
        # In-memory storage
        self.all_metrics = {
            'per_client_metrics': [],
            'per_class_metrics': [],
            'global_metrics': [],
            'gradient_metrics': [],
            'layer_gradient_metrics': [],
            'global_gradient_metrics': [],
            'propagation_metrics': [],
            'convergence_metrics': [],
            'data_distribution': [],
            'communication_efficiency': [],
            'round_summaries': []
        }
        
    def _init_log_files(self):
        """Initialize all CSV files with headers."""
        # 1. Per-Client, Per-Round, Per-Epoch Metrics
        self.per_client_file = self.exp_dir / "per_client_metrics.csv"
        with open(self.per_client_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'client_id', 'round', 'epoch', 'timestamp',
                'train_loss', 'train_accuracy', 
                'val_loss', 'val_accuracy',
                'test_loss', 'test_accuracy'
            ])
        
        # 2. Per-Client, Per-Class Metrics
        self.per_class_file = self.exp_dir / "per_class_metrics.csv"
        with open(self.per_class_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'client_id', 'round', 'class_id', 'class_name',
                'class_accuracy', 'class_precision', 'class_recall', 'class_f1_score',
                'class_samples', 'class_correct_predictions'
            ])
        
        # 3. Global Model Metrics
        self.global_file = self.exp_dir / "global_metrics.csv"
        with open(self.global_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'round', 'timestamp',
                'global_train_loss', 'global_train_accuracy',
                'global_test_loss', 'global_test_accuracy',
                'num_participating_clients', 'aggregation_time'
            ])
        
        # 4. Per-Client Gradient Metrics
        self.gradient_file = self.exp_dir / "gradient_metrics.csv"
        with open(self.gradient_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'client_id', 'round', 'epoch', 'timestamp',
                'gradient_norm_l2', 'gradient_norm_linf',
                'gradient_mean', 'gradient_std',
                'gradient_max', 'gradient_min'
            ])
        
        # 5. Per-Layer Gradient Metrics
        self.layer_gradient_file = self.exp_dir / "layer_gradient_metrics.csv"
        with open(self.layer_gradient_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'client_id', 'round', 'epoch', 'layer_name',
                'layer_gradient_norm', 'layer_gradient_mean', 'layer_gradient_std'
            ])
        
        # 6. Global Gradient Metrics
        self.global_gradient_file = self.exp_dir / "global_gradient_metrics.csv"
        with open(self.global_gradient_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'round', 'global_gradient_norm',
                'gradient_variance_across_clients', 'gradient_diversity'
            ])
        
        # 7. Model Propagation Metrics (P2P)
        self.propagation_file = self.exp_dir / "propagation_metrics.csv"
        with open(self.propagation_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'round', 'client_id', 'timestamp',
                'source_client_id', 'destination_client_id',
                'propagation_delay', 'hop_count',
                'cluster_id', 'inter_cluster_communication',
                'model_version', 'bytes_transferred'
            ])
        
        # 8. Convergence Metrics
        self.convergence_file = self.exp_dir / "convergence_metrics.csv"
        with open(self.convergence_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'round', 'client_accuracy_mean', 'client_accuracy_std',
                'client_accuracy_min', 'client_accuracy_max',
                'client_loss_mean', 'client_loss_std',
                'accuracy_convergence_rate', 'loss_convergence_rate',
                'stragglers_count'
            ])
        
        # 9. Data Distribution Metrics
        self.data_dist_file = self.exp_dir / "data_distribution.csv"
        with open(self.data_dist_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'client_id', 'total_samples', 'class_distribution_json',
                'data_heterogeneity_score'
            ])
        
        # 10. Communication Efficiency Metrics
        self.comm_efficiency_file = self.exp_dir / "communication_efficiency.csv"
        with open(self.comm_efficiency_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'round', 'client_id', 'upload_time', 'download_time',
                'communication_rounds_total', 'bandwidth_used_mb'
            ])
        
        # 11. Round Summary
        self.round_summary_file = self.exp_dir / "round_summary.csv"
        with open(self.round_summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'round', 'timestamp', 'num_clients_participated',
                'accuracy_mean', 'accuracy_median', 'accuracy_q25', 'accuracy_q75',
                'loss_mean', 'loss_std', 'loss_min', 'loss_max',
                'gradient_mean_norm', 'gradient_variance',
                'communication_time_total', 'communication_time_avg',
                'stragglers_identified'
            ])
    
    # ===================================================================
    # LOGGING METHODS
    # ===================================================================
    
    def log_per_client_metrics(
        self,
        client_id: int,
        round_num: int,
        epoch: int,
        train_loss: float,
        train_accuracy: float,
        val_loss: Optional[float] = None,
        val_accuracy: Optional[float] = None,
        test_loss: Optional[float] = None,
        test_accuracy: Optional[float] = None
    ):
        """Log per-client, per-round, per-epoch metrics."""
        timestamp = datetime.now().isoformat()
        
        with open(self.per_client_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                client_id, round_num, epoch, timestamp,
                train_loss, train_accuracy,
                val_loss or '', val_accuracy or '',
                test_loss or '', test_accuracy or ''
            ])
        
        self.all_metrics['per_client_metrics'].append({
            'client_id': client_id, 'round': round_num, 'epoch': epoch,
            'timestamp': timestamp, 'train_loss': train_loss, 
            'train_accuracy': train_accuracy, 'test_loss': test_loss,
            'test_accuracy': test_accuracy
        })
    
    def log_per_class_metrics(
        self,
        client_id: int,
        round_num: int,
        class_metrics: Dict[int, Dict[str, float]],
        class_names: Optional[Dict[int, str]] = None
    ):
        """Log per-client, per-class metrics.
        
        Args:
            client_id: Client identifier
            round_num: Round number
            class_metrics: Dict mapping class_id to metrics dict containing:
                {accuracy, precision, recall, f1_score, samples, correct_predictions}
            class_names: Optional mapping of class_id to class name
        """
        with open(self.per_class_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for class_id, metrics in class_metrics.items():
                class_name = class_names.get(class_id, f"class_{class_id}") if class_names else f"class_{class_id}"
                writer.writerow([
                    client_id, round_num, class_id, class_name,
                    metrics.get('accuracy', 0.0),
                    metrics.get('precision', 0.0),
                    metrics.get('recall', 0.0),
                    metrics.get('f1_score', 0.0),
                    metrics.get('samples', 0),
                    metrics.get('correct_predictions', 0)
                ])
        
        self.all_metrics['per_class_metrics'].extend([
            {'client_id': client_id, 'round': round_num, 'class_id': cid, **m}
            for cid, m in class_metrics.items()
        ])
    
    def log_global_metrics(
        self,
        round_num: int,
        global_train_loss: float,
        global_train_accuracy: float,
        global_test_loss: float,
        global_test_accuracy: float,
        num_participating_clients: int,
        aggregation_time: float
    ):
        """Log global model metrics (FedAvg)."""
        timestamp = datetime.now().isoformat()
        
        with open(self.global_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                round_num, timestamp,
                global_train_loss, global_train_accuracy,
                global_test_loss, global_test_accuracy,
                num_participating_clients, aggregation_time
            ])
        
        self.all_metrics['global_metrics'].append({
            'round': round_num, 'timestamp': timestamp,
            'global_train_loss': global_train_loss,
            'global_train_accuracy': global_train_accuracy,
            'global_test_loss': global_test_loss,
            'global_test_accuracy': global_test_accuracy,
            'num_participating_clients': num_participating_clients,
            'aggregation_time': aggregation_time
        })
    
    def log_gradient_metrics(
        self,
        client_id: int,
        round_num: int,
        epoch: int,
        gradients: Dict[str, torch.Tensor]
    ):
        """Log per-client gradient metrics.
        
        Args:
            client_id: Client identifier
            round_num: Round number
            epoch: Epoch number
            gradients: Dict of parameter names to gradient tensors
        """
        timestamp = datetime.now().isoformat()
        
        # Compute global gradient statistics
        all_grads = torch.cat([g.flatten() for g in gradients.values()])
        grad_norm_l2 = torch.norm(all_grads, p=2).item()
        grad_norm_linf = torch.norm(all_grads, p=float('inf')).item()
        grad_mean = all_grads.mean().item()
        grad_std = all_grads.std().item()
        grad_max = all_grads.max().item()
        grad_min = all_grads.min().item()
        
        # Log global gradient metrics
        with open(self.gradient_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                client_id, round_num, epoch, timestamp,
                grad_norm_l2, grad_norm_linf,
                grad_mean, grad_std,
                grad_max, grad_min
            ])
        
        # Log per-layer gradient metrics
        with open(self.layer_gradient_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for layer_name, grad in gradients.items():
                layer_norm = torch.norm(grad, p=2).item()
                layer_mean = grad.mean().item()
                layer_std = grad.std().item()
                writer.writerow([
                    client_id, round_num, epoch, layer_name,
                    layer_norm, layer_mean, layer_std
                ])
        
        self.all_metrics['gradient_metrics'].append({
            'client_id': client_id, 'round': round_num, 'epoch': epoch,
            'gradient_norm_l2': grad_norm_l2, 'gradient_norm_linf': grad_norm_linf,
            'gradient_mean': grad_mean, 'gradient_std': grad_std
        })
    
    def log_global_gradient_metrics(
        self,
        round_num: int,
        global_gradient_norm: float,
        gradient_variance_across_clients: float,
        gradient_diversity: float
    ):
        """Log global gradient metrics."""
        with open(self.global_gradient_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                round_num, global_gradient_norm,
                gradient_variance_across_clients, gradient_diversity
            ])
        
        self.all_metrics['global_gradient_metrics'].append({
            'round': round_num,
            'global_gradient_norm': global_gradient_norm,
            'gradient_variance_across_clients': gradient_variance_across_clients,
            'gradient_diversity': gradient_diversity
        })
    
    def log_propagation_metrics(
        self,
        round_num: int,
        client_id: int,
        source_client_id: int,
        destination_client_id: int,
        propagation_delay: float,
        hop_count: int,
        cluster_id: int,
        inter_cluster_communication: bool,
        model_version: int,
        bytes_transferred: int
    ):
        """Log model propagation metrics (P2P)."""
        timestamp = datetime.now().isoformat()
        
        with open(self.propagation_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                round_num, client_id, timestamp,
                source_client_id, destination_client_id,
                propagation_delay, hop_count,
                cluster_id, inter_cluster_communication,
                model_version, bytes_transferred
            ])
        
        self.all_metrics['propagation_metrics'].append({
            'round': round_num, 'client_id': client_id,
            'source_client_id': source_client_id,
            'destination_client_id': destination_client_id,
            'propagation_delay': propagation_delay,
            'hop_count': hop_count,
            'cluster_id': cluster_id,
            'inter_cluster_communication': inter_cluster_communication,
            'model_version': model_version,
            'bytes_transferred': bytes_transferred
        })
    
    def log_convergence_metrics(
        self,
        round_num: int,
        client_accuracies: List[float],
        client_losses: List[float],
        prev_round_accuracies: Optional[List[float]] = None,
        prev_round_losses: Optional[List[float]] = None,
        straggler_threshold: float = 0.1
    ):
        """Log convergence metrics."""
        acc_mean = np.mean(client_accuracies)
        acc_std = np.std(client_accuracies)
        acc_min = np.min(client_accuracies)
        acc_max = np.max(client_accuracies)
        
        loss_mean = np.mean(client_losses)
        loss_std = np.std(client_losses)
        
        # Convergence rates
        acc_conv_rate = 0.0
        loss_conv_rate = 0.0
        if prev_round_accuracies and prev_round_losses:
            acc_conv_rate = np.std(prev_round_accuracies) - acc_std
            loss_conv_rate = np.std(prev_round_losses) - loss_std
        
        # Stragglers: clients significantly below mean
        stragglers_count = sum(1 for acc in client_accuracies if acc < acc_mean - straggler_threshold)
        
        with open(self.convergence_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                round_num, acc_mean, acc_std, acc_min, acc_max,
                loss_mean, loss_std,
                acc_conv_rate, loss_conv_rate,
                stragglers_count
            ])
        
        self.all_metrics['convergence_metrics'].append({
            'round': round_num,
            'client_accuracy_mean': acc_mean,
            'client_accuracy_std': acc_std,
            'accuracy_convergence_rate': acc_conv_rate,
            'stragglers_count': stragglers_count
        })
    
    def log_data_distribution(
        self,
        client_id: int,
        total_samples: int,
        class_distribution: Dict[int, int],
        data_heterogeneity_score: float
    ):
        """Log data distribution metrics (once at start)."""
        class_dist_json = json.dumps(class_distribution)
        
        with open(self.data_dist_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                client_id, total_samples, class_dist_json,
                data_heterogeneity_score
            ])
        
        self.all_metrics['data_distribution'].append({
            'client_id': client_id,
            'total_samples': total_samples,
            'class_distribution': class_distribution,
            'data_heterogeneity_score': data_heterogeneity_score
        })
    
    def log_communication_efficiency(
        self,
        round_num: int,
        client_id: int,
        upload_time: float,
        download_time: float,
        communication_rounds_total: int,
        bandwidth_used_mb: float
    ):
        """Log communication efficiency metrics."""
        with open(self.comm_efficiency_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                round_num, client_id, upload_time, download_time,
                communication_rounds_total, bandwidth_used_mb
            ])
        
        self.all_metrics['communication_efficiency'].append({
            'round': round_num, 'client_id': client_id,
            'upload_time': upload_time, 'download_time': download_time,
            'bandwidth_used_mb': bandwidth_used_mb
        })
    
    def log_round_summary(
        self,
        round_num: int,
        num_clients_participated: int,
        client_accuracies: List[float],
        client_losses: List[float],
        gradient_norms: Optional[List[float]] = None,
        communication_times: Optional[List[float]] = None,
        stragglers_identified: Optional[List[int]] = None
    ):
        """Log comprehensive round summary."""
        timestamp = datetime.now().isoformat()
        
        accuracies = np.array(client_accuracies)
        losses = np.array(client_losses)
        
        acc_mean = np.mean(accuracies)
        acc_median = np.median(accuracies)
        acc_q25 = np.percentile(accuracies, 25)
        acc_q75 = np.percentile(accuracies, 75)
        
        loss_mean = np.mean(losses)
        loss_std = np.std(losses)
        loss_min = np.min(losses)
        loss_max = np.max(losses)
        
        grad_mean_norm = np.mean(gradient_norms) if gradient_norms else 0.0
        grad_variance = np.var(gradient_norms) if gradient_norms else 0.0
        
        comm_time_total = np.sum(communication_times) if communication_times else 0.0
        comm_time_avg = np.mean(communication_times) if communication_times else 0.0
        
        stragglers_str = json.dumps(stragglers_identified) if stragglers_identified else '[]'
        
        with open(self.round_summary_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                round_num, timestamp, num_clients_participated,
                acc_mean, acc_median, acc_q25, acc_q75,
                loss_mean, loss_std, loss_min, loss_max,
                grad_mean_norm, grad_variance,
                comm_time_total, comm_time_avg,
                stragglers_str
            ])
        
        self.all_metrics['round_summaries'].append({
            'round': round_num, 'timestamp': timestamp,
            'accuracy_mean': acc_mean, 'accuracy_median': acc_median,
            'loss_mean': loss_mean, 'loss_std': loss_std
        })
    
    # ===================================================================
    # UTILITY METHODS
    # ===================================================================
    
    def save_config(self, config: Dict[str, Any]):
        """Save experiment configuration."""
        config_file = self.exp_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def save_final_report(
        self,
        best_global_accuracy: float,
        best_round: int,
        final_per_client_accuracies: Dict[int, float],
        total_training_time: float,
        total_communication_overhead: float,
        convergence_round: Optional[int] = None,
        final_per_class_accuracy: Optional[Dict[int, float]] = None,
        avg_propagation_delay: Optional[float] = None,
        avg_hop_count: Optional[float] = None
    ):
        """Save final report with summary statistics."""
        report = {
            'best_global_accuracy': best_global_accuracy,
            'round_when_best_achieved': best_round,
            'final_per_client_accuracies_sorted': sorted(
                final_per_client_accuracies.items(),
                key=lambda x: x[1],
                reverse=True
            ),
            'total_training_time_seconds': total_training_time,
            'total_communication_overhead_seconds': total_communication_overhead,
            'convergence_round': convergence_round,
            'final_per_class_accuracy': final_per_class_accuracy,
            'model_propagation_statistics': {
                'average_delay_ms': avg_propagation_delay,
                'average_hop_count': avg_hop_count
            } if avg_propagation_delay else None
        }
        
        report_file = self.exp_dir / "final_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
    
    def get_log_dir(self) -> str:
        """Get the experiment log directory path."""
        return str(self.exp_dir)
    
    def get_plots_dir(self) -> str:
        """Get the plots directory path."""
        return str(self.plots_dir)

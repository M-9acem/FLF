"""Comprehensive metrics logging system for federated learning experiments."""

import json
import csv
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np


class MetricsLogger:
    """Comprehensive logger for tracking federated learning metrics."""
    
    def __init__(self, log_dir: str, experiment_name: str = "experiment"):
        """Initialize the metrics logger.
        
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
        
        # Initialize log files
        self.client_metrics_file = self.exp_dir / "client_metrics.csv"
        self.global_metrics_file = self.exp_dir / "global_metrics.csv"
        self.gradient_tracking_file = self.exp_dir / "gradient_tracking.csv"
        self.communication_logs_file = self.exp_dir / "communication_logs.csv"
        self.round_summary_file = self.exp_dir / "round_summary.csv"
        
        # Initialize CSV writers
        self._init_csv_files()
        
        # In-memory storage for quick aggregation
        self.client_metrics: List[Dict[str, Any]] = []
        self.global_metrics: List[Dict[str, Any]] = []
        self.gradient_metrics: List[Dict[str, Any]] = []
        self.communication_logs: List[Dict[str, Any]] = []
        self.round_summaries: List[Dict[str, Any]] = []
        
    def _init_csv_files(self):
        """Initialize CSV files with headers."""
        # Client metrics
        with open(self.client_metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'round', 'epoch', 'client_id', 
                'metric_type', 'value', 'class_id'
            ])
        
        # Global metrics
        with open(self.global_metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'round', 'metric_type', 'value'
            ])
        
        # Gradient tracking
        with open(self.gradient_tracking_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'round', 'epoch', 'client_id', 
                'gradient_norm', 'gradient_variance'
            ])
        
        # Communication logs
        with open(self.communication_logs_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'round', 'source_client', 'target_client',
                'communication_time', 'message_type'
            ])
        
        # Round summary
        with open(self.round_summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'round', 'avg_accuracy', 'std_accuracy',
                'avg_loss', 'std_loss', 'min_accuracy', 'max_accuracy',
                'gradient_norm_mean', 'gradient_norm_std'
            ])
    
    def log_client_metric(
        self, 
        round_num: int, 
        epoch: int, 
        client_id: int,
        metric_type: str,
        value: float,
        class_id: Optional[int] = None
    ):
        """Log a per-client metric.
        
        Args:
            round_num: Current round number
            epoch: Current epoch within the round
            client_id: ID of the client
            metric_type: Type of metric (e.g., 'accuracy', 'loss')
            value: Metric value
            class_id: Optional class ID for per-class metrics
        """
        timestamp = datetime.now().isoformat()
        entry = {
            'timestamp': timestamp,
            'round': round_num,
            'epoch': epoch,
            'client_id': client_id,
            'metric_type': metric_type,
            'value': value,
            'class_id': class_id if class_id is not None else ''
        }
        
        self.client_metrics.append(entry)
        
        # Write to CSV
        with open(self.client_metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, round_num, epoch, client_id,
                metric_type, value, class_id if class_id is not None else ''
            ])
    
    def log_global_metric(
        self,
        round_num: int,
        metric_type: str,
        value: float
    ):
        """Log a global model metric.
        
        Args:
            round_num: Current round number
            metric_type: Type of metric (e.g., 'global_accuracy', 'global_loss')
            value: Metric value
        """
        timestamp = datetime.now().isoformat()
        entry = {
            'timestamp': timestamp,
            'round': round_num,
            'metric_type': metric_type,
            'value': value
        }
        
        self.global_metrics.append(entry)
        
        # Write to CSV
        with open(self.global_metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, round_num, metric_type, value])
    
    def log_gradient(
        self,
        round_num: int,
        epoch: int,
        client_id: int,
        gradient_norm: float,
        gradient_variance: Optional[float] = None
    ):
        """Log gradient metrics.
        
        Args:
            round_num: Current round number
            epoch: Current epoch within the round
            client_id: ID of the client
            gradient_norm: L2 norm of the gradient
            gradient_variance: Variance of gradient components
        """
        timestamp = datetime.now().isoformat()
        entry = {
            'timestamp': timestamp,
            'round': round_num,
            'epoch': epoch,
            'client_id': client_id,
            'gradient_norm': gradient_norm,
            'gradient_variance': gradient_variance if gradient_variance is not None else ''
        }
        
        self.gradient_metrics.append(entry)
        
        # Write to CSV
        with open(self.gradient_tracking_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, round_num, epoch, client_id,
                gradient_norm, gradient_variance if gradient_variance is not None else ''
            ])
    
    def log_communication(
        self,
        round_num: int,
        source_client: int,
        target_client: int,
        communication_time: float,
        message_type: str = "model_update"
    ):
        """Log communication events.
        
        Args:
            round_num: Current round number
            source_client: ID of the sending client
            target_client: ID of the receiving client
            communication_time: Time taken for communication (seconds)
            message_type: Type of message exchanged
        """
        timestamp = datetime.now().isoformat()
        entry = {
            'timestamp': timestamp,
            'round': round_num,
            'source_client': source_client,
            'target_client': target_client,
            'communication_time': communication_time,
            'message_type': message_type
        }
        
        self.communication_logs.append(entry)
        
        # Write to CSV
        with open(self.communication_logs_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, round_num, source_client, target_client,
                communication_time, message_type
            ])
    
    def log_round_summary(
        self,
        round_num: int,
        client_accuracies: List[float],
        client_losses: List[float],
        gradient_norms: Optional[List[float]] = None
    ):
        """Log aggregated statistics for a round.
        
        Args:
            round_num: Current round number
            client_accuracies: List of accuracy values for all clients
            client_losses: List of loss values for all clients
            gradient_norms: Optional list of gradient norms for all clients
        """
        timestamp = datetime.now().isoformat()
        
        avg_accuracy = np.mean(client_accuracies)
        std_accuracy = np.std(client_accuracies)
        avg_loss = np.mean(client_losses)
        std_loss = np.std(client_losses)
        min_accuracy = np.min(client_accuracies)
        max_accuracy = np.max(client_accuracies)
        
        gradient_norm_mean = np.mean(gradient_norms) if gradient_norms else ''
        gradient_norm_std = np.std(gradient_norms) if gradient_norms else ''
        
        entry = {
            'timestamp': timestamp,
            'round': round_num,
            'avg_accuracy': avg_accuracy,
            'std_accuracy': std_accuracy,
            'avg_loss': avg_loss,
            'std_loss': std_loss,
            'min_accuracy': min_accuracy,
            'max_accuracy': max_accuracy,
            'gradient_norm_mean': gradient_norm_mean,
            'gradient_norm_std': gradient_norm_std
        }
        
        self.round_summaries.append(entry)
        
        # Write to CSV
        with open(self.round_summary_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, round_num, avg_accuracy, std_accuracy,
                avg_loss, std_loss, min_accuracy, max_accuracy,
                gradient_norm_mean, gradient_norm_std
            ])
    
    def save_config(self, config: Dict[str, Any]):
        """Save experiment configuration.
        
        Args:
            config: Configuration dictionary
        """
        config_file = self.exp_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def get_log_dir(self) -> str:
        """Get the experiment log directory path."""
        return str(self.exp_dir)
    
    def get_plots_dir(self) -> str:
        """Get the plots directory path."""
        return str(self.plots_dir)

# Federated Learning Framework

A comprehensive Python framework for federated learning that supports both **centralized** (FedAvg) and **decentralized** (peer-to-peer) training modes with extensive metrics tracking and visualization.

## Features

### Training Modes
- **Centralized (FedAvg)**: Traditional server-client federated averaging
- **Decentralized (P2P)**: Peer-to-peer gossip-based learning with two-cluster topology

### Mixing Methods for Decentralized Training

The mixing matrix determines how clients weight their neighbors' models during gossip aggregation:

1. **Metropolis-Hastings** (default)
   - Weight: `w_ij = 1 / (1 + max(degree_i, degree_j))`
   - Guarantees doubly stochastic matrix
   - Most widely used, proven convergence properties
   - Best for general-purpose decentralized learning

2. **Max-Degree**
   - Weight: `w_ij = 1 / max_degree` for all edges
   - Simple uniform weights based on graph's maximum degree
   - Good for regular or nearly-regular graphs
   - Faster computation than Metropolis-Hastings

3. **Jaccard Similarity**
   - Weights based on neighborhood overlap: `w_ij = 1 - |N(i) ∩ N(j)| / |N(i) ∪ N(j)|`
   - Adapts to local graph structure
   - Better for highly clustered or community-structured networks
   - Emphasizes structural similarity

4. **MATCHA** (Optimal)
   - Uses convex optimization to find optimal activation probabilities and mixing weights
   - Maximizes convergence rate theoretically
   - Requires `cvxpy` library and more computation
   - Best when communication is expensive and convergence speed is critical
   - May fall back to Metropolis-Hastings if graph is disconnected or optimization fails

### Models
- **SimpleCNN** (default): For CIFAR-10, MNIST, Fashion-MNIST
- **LeNet5**: For MNIST, Fashion-MNIST
- **ResNet8**: For CIFAR-10
- **ResNet18**: For CIFAR-10
- **ResNet50**: For CIFAR-10

All models are implemented in `src/models/` and selectable via `--model`.

### Data Partitioning
- **IID**: Random uniform distribution (all clients get similar data)
- **Non-IID Dirichlet**: Controlled heterogeneity with `--alpha` parameter (lower alpha = more heterogeneity)
- **Non-IID Pathological**: Each client receives a fixed number of classes (`--classes_per_client`)

Partitioning logic is in `src/utils/data_loader.py`.

### Comprehensive Metrics Logging
- Per-client metrics (accuracy, loss, per-class performance)
- Global model metrics (centralized mode)
- Per-class metrics (all modes)
- Gradient tracking (norms, variance)
- Communication logs (decentralized mode)
- Round-level aggregate statistics
- Final client weights and statistics (for comparison)

### Visualization
- Jupyter notebook with publication-ready plots (`analysis_centralized_simple.ipynb`, `analysis_p2p_simple.ipynb`)
- Training progress charts (accuracy/loss vs. rounds)
- Per-class performance analysis
- Gradient convergence analysis
- Client convergence patterns
- Communication matrix (decentralized mode)
- Network topology visualization (HTML)
- `--intra_cluster_communication`: If set, all nodes in a cluster are interconnected (default, fully connected cluster). If not set, only the edge node in each cluster communicates with the rest (star topology, no intra-cluster communication except via the edge node).

## Project Structure

```
FederatedLearningFramework/
├── main.py                 # Main entry point with CLI
├── analysis.ipynb          # Analysis and visualization notebook
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── src/
│   ├── centralized/       # FedAvg implementation
│   │   ├── client.py     # FedAvg client
│   │   └── server.py     # FedAvg server
│   ├── decentralized/     # P2P implementation
│   │   ├── p2p_client.py # P2P client
│   │   ├── p2p_runner.py # P2P coordinator
│   │   └── topology.py   # Network topology creation
│   ├── models/            # Neural network models
│   │   ├── lenet5.py     # LeNet5 architecture
│   │   └── resnet.py     # ResNet18/50 architectures
│   └── utils/             # Utility modules
│       ├── logger.py      # Comprehensive metrics logger
│       └── data_loader.py # Dataset and partitioning utilities
├── config/                # Configuration files (optional)
├── data/                  # Downloaded datasets
└── logs/                  # Experiment logs and results
    └── experiment_name/
        └── YYYY-MM-DD_HH-MM-SS/
            ├── client_metrics.csv
            ├── global_metrics.csv
            ├── gradient_tracking.csv
            ├── communication_logs.csv
            ├── round_summary.csv
            ├── config.json
            └── plots/
```

## Installation

### 1. Clone or download this repository

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Basic Commands

#### Centralized FedAvg Training

```bash
python main.py --type centralized --rounds 10 --epochs 5 --num_clients 10
```

#### Decentralized P2P Training

```bash
python main.py --type decentralized --rounds 10 --epochs 5 --num_clients 10
```

### Command-Line Arguments

#### Required Arguments
- `--type`: Training mode (`centralized` or `decentralized`)
- `--rounds`: Number of federated rounds
- `--epochs`: Number of local training epochs per round
- `--num_clients`: Number of clients

#### Model & Dataset
- `--model`: Model architecture (`simple_cnn`, `lenet5`, `resnet18`, `resnet50`, default: `simple_cnn`)
- `--dataset`: Dataset (`mnist`, `cifar10`, `fashion_mnist`, default: `cifar10`)
- `--data_dir`: Directory for storing datasets (default: `./data`)

#### Data Partitioning
- `--partition`: Partitioning strategy (`iid`, `dirichlet`, `pathological`, default: `iid`)
- `--alpha`: Dirichlet concentration parameter (default: `0.5`, lower = more heterogeneous)
- `--classes_per_client`: Number of classes per client for pathological partitioning (default: `2`)

#### Optimizer Parameters
- `--lr`: Learning rate (default: `0.01`)
- `--momentum`: SGD momentum (default: `0.9`)
- `--weight_decay`: Weight decay for regularization (default: `0.0`)
- `--batch_size`: Batch size (default: `32`)

#### Decentralized-Specific Parameters

- `--intra_cluster_communication`: If set, all nodes in a cluster are interconnected (default, fully connected cluster). If not set, only the edge node in each cluster communicates with the rest (star topology, no intra-cluster communication except via the edge node).

- `--main_link_prob`: Probability of main bridge link activation (default: `1.0`)
- `--border_link_prob`: Probability of border links activation (default: `1.0`)
- `--intra_cluster_prob`: Probability of intra-cluster links (default: `0.8`)
- `--mixing_method`: Mixing matrix method for gossip aggregation (default: `metropolis_hastings`)
   - `metropolis_hastings`: Balanced weights based on node degrees (most common)
   - `max_degree`: Uniform weights based on maximum degree in graph
   - `jaccard`: Weights based on Jaccard similarity of neighborhoods
   - `matcha`: Optimal weights via convex optimization (requires cvxpy)

#### System Parameters
- `--no_cuda`: Disable CUDA (use CPU only)
- `--num_workers`: Number of data loading workers (default: `0`)
- `--seed`: Random seed for reproducibility (default: `42`)

#### Logging Parameters
- `--log_dir`: Base directory for logs (default: `./logs`)
- `--experiment_name`: Custom experiment name (default: auto-generated)

### Example Commands

#### 1. CIFAR-10 with ResNet18, IID data, Centralized

```bash
python main.py --type centralized --model resnet18 --dataset cifar10 --partition iid --num_clients 10 --rounds 20 --epochs 5 --lr 0.01
```

#### 2. MNIST with LeNet5, Non-IID Dirichlet, Decentralized

```bash
python main.py --type decentralized --model lenet5 --dataset mnist --partition dirichlet --alpha 0.1 --num_clients 10 --rounds 15 --epochs 3
```

#### 3. CIFAR-10 with Non-IID Pathological (2 classes per client)

```bash
python main.py --type centralized --dataset cifar10 --partition pathological --classes_per_client 2 --num_clients 5 --rounds 25 --epochs 5
```

#### 4. Decentralized with Custom Topology Parameters

```bash
python main.py --type decentralized --num_clients 20 --rounds 30 --epochs 5 --main_link_prob 0.8 --border_link_prob 0.5 --intra_cluster_prob 0.9
```

#### 6. Decentralized with Star Topology (no intra-cluster communication except via edge node)

```bash
python main.py --type decentralized --num_clients 10 --rounds 10 --epochs 3 --intra_cluster_communication False
```

#### 5. Decentralized with Different Mixing Methods

```bash
# Metropolis-Hastings (default, most common)
python main.py --type decentralized --mixing_method metropolis_hastings --num_clients 10 --rounds 15

# Max-degree mixing (uniform weights)
python main.py --type decentralized --mixing_method max_degree --num_clients 10 --rounds 15

# Jaccard similarity mixing
python main.py --type decentralized --mixing_method jaccard --num_clients 10 --rounds 15

# MATCHA optimal mixing (requires cvxpy)
python main.py --type decentralized --mixing_method matcha --num_clients 10 --rounds 15
```

## Analyzing Results

### Using the Jupyter Notebook

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Open `analysis.ipynb`

3. Run all cells to generate:
   - Global model performance charts
   - Per-client training progress
   - Per-class accuracy analysis
   - Gradient analysis and convergence
   - Communication patterns (for decentralized mode)

The notebook automatically loads the most recent experiment and saves all plots to the `logs/experiment_name/YYYY-MM-DD_HH-MM-SS/plots/` directory.

### Manual Analysis

All metrics are saved in CSV format and can be analyzed using pandas:

```python
import pandas as pd

# Load metrics
df_client = pd.read_csv('logs/experiment_name/YYYY-MM-DD_HH-MM-SS/client_metrics.csv')
df_summary = pd.read_csv('logs/experiment_name/YYYY-MM-DD_HH-MM-SS/round_summary.csv')

# Analyze
print(df_summary[['round', 'avg_accuracy', 'std_accuracy']])
```

## Output Files

Each experiment creates the following files:

- **client_metrics.csv**: Per-client, per-round, per-epoch metrics (accuracy, loss, class-specific)
- **client_final_weights.pt**: All client weights at end of training (for comparison)
- **client_final_stats.csv**: Per-client statistics at end of training
- **global_metrics.csv**: Global model metrics per round (centralized mode only)
- **gradient_tracking.csv**: Gradient norms and variance
- **communication_logs.csv**: Network communication events (decentralized mode only)
- **round_summary.csv**: Aggregated statistics per round (mean, std, min, max)
- **config.json**: Experiment configuration for reproducibility
- **plots/**: Generated visualization plots (PNG format)
- **network_topology.html**: Interactive network visualization (decentralized mode)
- All features listed above are implemented and tested in the codebase. See `src/` for full logic and `requirements.txt` for dependencies.

## Tips and Best Practices

### For Better Convergence
- Use higher learning rates (`--lr 0.05` to `0.1`) for faster convergence
- Increase local epochs (`--epochs 10`) for fewer communication rounds
- Use momentum (`--momentum 0.9`) for smoother optimization

### For Non-IID Scenarios
- Lower `--alpha` (e.g., `0.1` to `0.5`) for more heterogeneous data
- Use `--partition pathological --classes_per_client 2` for extreme heterogeneity
- Consider more rounds (`--rounds 50+`) for non-IID convergence

### For Decentralized Training
- Adjust topology parameters for different communication patterns
- Lower `--main_link_prob` to simulate unreliable bridge connections
- Monitor communication logs to understand network behavior
- Choose mixing method based on your needs:
  - **Metropolis-Hastings**: Best general-purpose choice, proven convergence
  - **Max-degree**: Simple uniform weights, good for regular graphs
  - **Jaccard**: Adapts to neighborhood similarity, better for clustered networks
  - **MATCHA**: Optimal convergence rate, but requires cvxpy and more computation

### For Resource Management
- Use `--no_cuda` to force CPU training (slower but uses less memory)
- Reduce `--batch_size` if running into memory issues
- Set `--num_workers 0` to avoid multiprocessing overhead on Windows

## Reproducibility

All experiments are seeded for reproducibility:
- Use `--seed` to set a specific random seed
- Configuration is saved in `config.json` for each experiment
- Reload the configuration to reproduce experiments exactly

## Troubleshooting

### CUDA Out of Memory
```bash
python main.py --type centralized --batch_size 16 --model lenet5
```
or
```bash
python main.py --type centralized --no_cuda
```

### Slow Training on CPU
- Use smaller models (LeNet5)
- Reduce number of clients
- Decrease batch size

### Import Errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Citation

If you use this framework in your research, please cite the FLP2P project and this framework.

## License

This project is provided for educational and research purposes.

## Contributing

Feel free to open issues or submit pull requests for improvements!

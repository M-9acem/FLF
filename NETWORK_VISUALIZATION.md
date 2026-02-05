# Network Topology Visualization

This framework now generates interactive HTML visualizations of the P2P network topology, similar to the FLP2P project.

## Features

- **Interactive HTML visualization** using PyVis
- **Cluster-based color coding**: Clients colored by their cluster membership
  - Cluster 0: Blue
  - Cluster 1: Red
- **Edge weights**: Displayed on edges when available
- **Network statistics**: Saved to `topology_info.txt`
- **Interactive controls**: Zoom, pan, drag nodes, physics simulation

## Installation

Install the required `pyvis` package:

```bash
pip install pyvis
# or
pip install -r requirements.txt
```

## Usage

### Automatic Generation

When running P2P experiments, the network topology is automatically saved:

```bash
python main.py --type decentralized --rounds 10 --epochs 5 --num_clients 10
```

This creates:
- `network_topology.html` - Interactive visualization
- `topology_info.txt` - Network statistics

### Test the Visualization

Run the test script to generate a sample visualization:

```bash
python test_visualization.py
```

This creates a `test_visualization/` directory with example files.

## Output Files

After running a P2P experiment, your logs directory will contain:

```
logs/
  └── [mixing_method]_comparison/
      └── [timestamp]/
          ├── p2p_metrics.csv
          ├── p2p_per_class_metrics.csv
          ├── network_topology.html       ← Interactive visualization
          ├── topology_info.txt           ← Network statistics
          └── config.json
```

## Viewing the Visualization

Simply open `network_topology.html` in any web browser:

```bash
# Windows
start logs/[experiment]/network_topology.html

# Linux/Mac
open logs/[experiment]/network_topology.html
```

The visualization is fully interactive:
- **Drag nodes** to rearrange the layout
- **Zoom** in/out with mouse wheel
- **Click nodes** to see details
- **Hover over edges** to see weights
- **Pan** by dragging the background

## Network Statistics

The `topology_info.txt` file contains:
- Number of nodes and edges
- Average degree
- Network density
- Diameter (if connected)
- Degree distribution per node
- Cluster assignments

## Customization

To customize the visualization, edit [src/utils/visualization.py](src/utils/visualization.py):

```python
from src.utils.visualization import plot_network_topology

plot_network_topology(
    graph=your_graph,
    output_path="custom_topology",
    title="My Custom Network",
    cluster_assignments=clusters,
    width="1600px",
    height="1000px"
)
```

## Cluster Color Scheme

- Cluster 0: Blue (#3498db)
- Cluster 1: Red (#e74c3c)
- Cluster 2: Green (#2ecc71)
- Cluster 3: Orange (#f39c12)
- No cluster: Gray (#95a5a6)

## Physics Simulation

The visualization uses the ForceAtlas2 algorithm for node positioning, which:
- Repels nodes from each other
- Attracts connected nodes
- Stabilizes after ~1000 iterations
- Can be paused/resumed in the browser

## Troubleshooting

If visualization doesn't generate:

1. **Check pyvis installation**:
   ```bash
   pip install pyvis
   ```

2. **Verify file permissions**: Ensure the logs directory is writable

3. **Check console output**: The script will print warnings if pyvis is missing

4. **Manual test**:
   ```bash
   python test_visualization.py
   ```

## Comparison with FLP2P

Similar to FLP2P's `plot_topology()` function, but with enhancements:
- ✅ Cluster-based coloring
- ✅ Better physics simulation
- ✅ Automatic statistics generation
- ✅ Integrated into experiment logging
- ✅ Configurable appearance

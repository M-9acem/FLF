# Test Network Topology Visualization
# This script demonstrates the network visualization feature

from pathlib import Path
from src.decentralized.topology import create_two_cluster_topology
from src.utils.visualization import plot_network_topology, save_topology_info

# Create a test network
num_clients = 10
graph = create_two_cluster_topology(
    num_clients=num_clients,
    main_link_prob=1.0,
    border_link_prob=0.8,
    intra_cluster_prob=0.6
)

# Define cluster assignments
cluster_assignments = {i: 0 if i < num_clients // 2 else 1 for i in range(num_clients)}

# Create output directory
output_dir = Path("test_visualization")
output_dir.mkdir(exist_ok=True)

print("Generating network topology visualization...")
print(f"Number of nodes: {graph.number_of_nodes()}")
print(f"Number of edges: {graph.number_of_edges()}")

# Save visualization
plot_network_topology(
    graph=graph,
    output_path=str(output_dir / "test_topology"),
    title="Test Two-Cluster Network",
    cluster_assignments=cluster_assignments
)

# Save topology info
save_topology_info(
    graph=graph,
    output_dir=output_dir,
    cluster_assignments=cluster_assignments
)

print(f"\nVisualization saved to: {output_dir / 'test_topology.html'}")
print(f"Open this file in your web browser to view the interactive network!")
print(f"Topology info saved to: {output_dir / 'topology_info.txt'}")

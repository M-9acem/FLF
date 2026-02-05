"""Visualization utilities for federated learning experiments."""

import networkx as nx
from pathlib import Path
from typing import Optional, Dict


def plot_network_topology(
    graph: nx.Graph, 
    output_path: str,
    title: str = "Network Topology",
    cluster_assignments: Optional[Dict[int, int]] = None,
    width: str = "1200px",
    height: str = "800px"
) -> None:
    """Plot network topology as interactive HTML using PyVis.
    
    Args:
        graph: NetworkX graph to visualize
        output_path: Path to save HTML file (with or without .html extension)
        title: Title for the visualization
        cluster_assignments: Optional dict mapping node_id to cluster_id for coloring
        width: Width of the visualization
        height: Height of the visualization
    """
    try:
        from pyvis.network import Network
    except ImportError:
        print("Warning: pyvis not installed. Install with: pip install pyvis")
        print("Skipping network visualization.")
        return
    
    # Remove self-loops for cleaner visualization
    graph_clean = graph.copy()
    self_loops = list(nx.selfloop_edges(graph_clean))
    graph_clean.remove_edges_from(self_loops)
    
    # Create PyVis network
    net = Network(
        notebook=False,
        width=width,
        height=height,
        bgcolor="#222222",
        font_color="white",
        directed=False
    )
    
    # Define cluster colors
    cluster_colors = {
        0: "#3498db",  # Blue
        1: "#e74c3c",  # Red
        2: "#2ecc71",  # Green
        3: "#f39c12",  # Orange
    }
    default_color = "#95a5a6"  # Gray
    
    # Add nodes with cluster-based coloring
    for node in graph_clean.nodes():
        color = default_color
        if cluster_assignments and node in cluster_assignments:
            cluster_id = cluster_assignments[node]
            color = cluster_colors.get(cluster_id, default_color)
        
        label = f"Client {node}"
        if cluster_assignments and node in cluster_assignments:
            label += f"\nCluster {cluster_assignments[node]}"
        
        net.add_node(
            node,
            label=label,
            title=f"Client {node}",
            color=color,
            size=25,
            font={"size": 14, "color": "white"}
        )
    
    # Add edges with weights if available
    for u, v, data in graph_clean.edges(data=True):
        weight = data.get('weight', 1.0)
        label = f"{weight:.3f}" if 'weight' in data else ""
        
        net.add_edge(
            u, v,
            title=f"Weight: {weight:.4f}",
            label=label,
            color="#7f8c8d",
            width=2
        )
    
    # Configure physics for better layout
    net.set_options("""
    {
      "nodes": {
        "borderWidth": 2,
        "borderWidthSelected": 4,
        "shape": "dot"
      },
      "edges": {
        "color": {
          "inherit": false
        },
        "smooth": {
          "enabled": true,
          "type": "continuous"
        }
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 200,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {
          "enabled": true,
          "iterations": 1000
        }
      }
    }
    """)
    
    # Ensure path has .html extension
    if not output_path.endswith('.html'):
        output_path = output_path + '.html'
    
    # Save the visualization
    net.save_graph(output_path)
    print(f"Network topology saved to: {output_path}")


def save_topology_info(
    graph: nx.Graph,
    output_dir: Path,
    cluster_assignments: Optional[Dict[int, int]] = None
) -> None:
    """Save network topology information to text file.
    
    Args:
        graph: NetworkX graph
        output_dir: Directory to save info file
        cluster_assignments: Optional cluster assignments
    """
    info_path = output_dir / "topology_info.txt"
    
    with open(info_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("NETWORK TOPOLOGY INFORMATION\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Number of Nodes: {graph.number_of_nodes()}\n")
        f.write(f"Number of Edges: {graph.number_of_edges()}\n")
        f.write(f"Average Degree: {sum(dict(graph.degree()).values()) / graph.number_of_nodes():.2f}\n")
        f.write(f"Density: {nx.density(graph):.4f}\n")
        
        if nx.is_connected(graph):
            f.write(f"Diameter: {nx.diameter(graph)}\n")
            f.write(f"Average Shortest Path Length: {nx.average_shortest_path_length(graph):.2f}\n")
        else:
            f.write("Graph is not connected\n")
            f.write(f"Number of Connected Components: {nx.number_connected_components(graph)}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("NODE DEGREES\n")
        f.write("=" * 80 + "\n")
        for node, degree in sorted(graph.degree(), key=lambda x: x[0]):
            cluster_info = ""
            if cluster_assignments and node in cluster_assignments:
                cluster_info = f" (Cluster {cluster_assignments[node]})"
            f.write(f"Node {node}{cluster_info}: {degree}\n")
        
        if cluster_assignments:
            f.write("\n" + "=" * 80 + "\n")
            f.write("CLUSTER INFORMATION\n")
            f.write("=" * 80 + "\n")
            clusters = {}
            for node, cluster_id in cluster_assignments.items():
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(node)
            
            for cluster_id in sorted(clusters.keys()):
                nodes = sorted(clusters[cluster_id])
                f.write(f"Cluster {cluster_id}: {len(nodes)} nodes - {nodes}\n")
    
    print(f"Topology information saved to: {info_path}")

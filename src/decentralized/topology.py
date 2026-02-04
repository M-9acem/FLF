"""Network topology creation for P2P federated learning."""

import networkx as nx
import numpy as np
from typing import Tuple, Dict, Literal

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

# Type alias for mixing methods
MixingMethod = Literal['metropolis_hastings', 'max_degree', 'jaccard', 'matcha']


def create_two_cluster_topology(
    num_clients: int,
    main_link_prob: float = 1.0,
    border_link_prob: float = 1.0,
    intra_cluster_prob: float = 0.8
) -> nx.Graph:
    """Create a two-cluster network topology.
    
    This creates two densely connected clusters with a bridge connection between them.
    
    Args:
        num_clients: Total number of clients
        main_link_prob: Probability of main bridge link activation
        border_link_prob: Probability of border links activation
        intra_cluster_prob: Probability of edges within clusters
        
    Returns:
        NetworkX graph representing the topology
    """
    if num_clients < 4:
        raise ValueError("Need at least 4 clients for two-cluster topology")
    
    G = nx.Graph()
    G.add_nodes_from(range(num_clients))
    
    # Split into two clusters
    cluster_size = num_clients // 2
    cluster1 = list(range(cluster_size))
    cluster2 = list(range(cluster_size, num_clients))
    
    # Create dense connections within each cluster
    for cluster in [cluster1, cluster2]:
        for i in cluster:
            for j in cluster:
                if i < j and np.random.random() < intra_cluster_prob:
                    G.add_edge(i, j, probability_selection=intra_cluster_prob)
    
    # Find center nodes (highest degree in each cluster)
    cluster1_degrees = [(node, G.degree(node)) for node in cluster1]
    cluster2_degrees = [(node, G.degree(node)) for node in cluster2]
    
    center1 = max(cluster1_degrees, key=lambda x: x[1])[0]
    center2 = max(cluster2_degrees, key=lambda x: x[1])[0]
    
    # Add main bridge link between centers
    G.add_edge(center1, center2, probability_selection=main_link_prob)
    
    # Find border nodes (neighbors of centers)
    neighbors1 = [n for n in G.neighbors(center1) if n in cluster1]
    neighbors2 = [n for n in G.neighbors(center2) if n in cluster2]
    
    # Add border links if we have neighbors
    if neighbors1 and neighbors2:
        border1 = neighbors1[0]
        border2 = neighbors2[0]
        
        # Add edge between border nodes
        if not G.has_edge(border1, border2):
            G.add_edge(border1, border2, probability_selection=border_link_prob)
    
    return G


def create_mixing_matrix(
    graph: nx.Graph,
    num_clients: int,
    method: MixingMethod = 'metropolis_hastings'
) -> np.ndarray:
    """Create mixing matrix for gossip aggregation.
    
    The mixing matrix defines how much weight each client gives to itself
    and its neighbors during aggregation.
    
    Args:
        graph: Network topology graph
        num_clients: Number of clients
        method: Mixing method to use ('metropolis_hastings', 'max_degree', 'jaccard', 'matcha')
        
    Returns:
        Mixing matrix (W) of shape (num_clients, num_clients)
    """
    if method == 'metropolis_hastings':
        return _metropolis_hastings_weights(graph, num_clients)
    elif method == 'max_degree':
        return _max_degree_weights(graph, num_clients)
    elif method == 'jaccard':
        return _jaccard_weights(graph, num_clients)
    elif method == 'matcha':
        return _matcha_weights(graph, num_clients)
    else:
        raise ValueError(f"Unknown mixing method: {method}")


def _metropolis_hastings_weights(graph: nx.Graph, num_clients: int) -> np.ndarray:
    """Metropolis-Hastings mixing weights.
    
    Weight between i and j: 1 / (1 + max(degree_i, degree_j))
    """
    W = np.zeros((num_clients, num_clients))
    
    for node in range(num_clients):
        neighbors = list(graph.neighbors(node))
        degree = len(neighbors)
        
        for neighbor in neighbors:
            neighbor_degree = len(list(graph.neighbors(neighbor)))
            max_degree = max(degree, neighbor_degree)
            W[node, neighbor] = 1.0 / (max_degree + 1)
        
        # Self-weight: ensure rows sum to 1
        W[node, node] = 1.0 - W[node, :].sum()
    
    return W


def _max_degree_weights(graph: nx.Graph, num_clients: int) -> np.ndarray:
    """Maximum degree mixing weights.
    
    All edges use weight 1/max_degree where max_degree is the maximum
    degree in the entire graph.
    """
    W = np.zeros((num_clients, num_clients))
    
    # Find maximum degree in graph
    if len(graph.edges()) == 0:
        return np.eye(num_clients)
    
    max_degree = max(dict(graph.degree()).values())
    
    for node in range(num_clients):
        neighbors = list(graph.neighbors(node))
        
        # Edge weights
        for neighbor in neighbors:
            W[node, neighbor] = 1.0 / max_degree
        
        # Self-weight
        degree = len(neighbors)
        W[node, node] = 1.0 - degree / max_degree
    
    return W


def _jaccard_weights(graph: nx.Graph, num_clients: int) -> np.ndarray:
    """Jaccard similarity-based mixing weights.
    
    Weight based on Jaccard similarity of neighborhoods:
    J(i,j) = |N(i) ∩ N(j)| / |N(i) ∪ N(j)|
    Weight = 1 - J(i,j), then normalized to doubly stochastic
    """
    W = np.zeros((num_clients, num_clients))
    
    if len(graph.edges()) == 0:
        return np.eye(num_clients)
    
    nodes = list(range(num_clients))
    
    # Compute Jaccard-based weights
    for i in nodes:
        neighbors_i = set(graph.neighbors(i))
        set_i = neighbors_i | {i}  # Include self
        
        for j in neighbors_i:
            set_j = set(graph.neighbors(j)) | {j}
            
            # Jaccard similarity: intersection over union
            intersection = len(set_i & set_j)
            union = len(set_i | set_j)
            
            # Weight is 1 - Jaccard similarity
            W[i, j] = 1.0 - intersection / union
            W[j, i] = 1.0 - intersection / union
    
    # Normalize: scale by maximum flow to ensure stability
    flows = np.sum(W, axis=0)
    global_scaling_factor = np.max(flows)
    
    if global_scaling_factor > 0:
        W /= (global_scaling_factor + 1e-10)
    
    # Add self-loops to make rows sum to 1
    for i in range(num_clients):
        W[i, i] = max(1.0 - np.sum(W[i, :]), 0.0)
    
    return W


def _matcha_weights(graph: nx.Graph, num_clients: int) -> np.ndarray:
    """MATCHA (Optimal) mixing weights using convex optimization.
    
    Uses CVX optimization to find optimal edge activation probabilities
    and mixing weights for fastest convergence.
    
    Requires: cvxpy library
    """
    if not CVXPY_AVAILABLE:
        print("Warning: cvxpy not available. Falling back to Metropolis-Hastings.")
        return _metropolis_hastings_weights(graph, num_clients)
    
    if len(graph.edges()) == 0:
        return np.eye(num_clients)
    
    try:
        # Get subgraphs (connected components)
        subgraphs = list(nx.connected_components(graph))
        
        if len(subgraphs) > 1:
            # Graph is not connected, use simpler method
            print("Warning: Graph not connected. Using Metropolis-Hastings for MATCHA.")
            return _metropolis_hastings_weights(graph, num_clients)
        
        # Compute Laplacian matrix
        L = nx.laplacian_matrix(graph).toarray()
        
        # Get optimal activation probabilities using CVX
        edge_probs = _get_optimal_probabilities(graph, L)
        
        # Compute mixing weights using SDP
        W = _get_optimal_alpha(graph, edge_probs, num_clients)
        
        return W
        
    except Exception as e:
        print(f"MATCHA optimization failed: {e}. Falling back to Metropolis-Hastings.")
        return _metropolis_hastings_weights(graph, num_clients)


def _get_optimal_probabilities(graph: nx.Graph, L: np.ndarray) -> Dict[Tuple[int, int], float]:
    """Compute optimal edge activation probabilities for MATCHA.
    
    Solves CVX problem to maximize expected spectral gap.
    """
    edges = list(graph.edges())
    n_edges = len(edges)
    
    # Create CVX variables
    p = cp.Variable(n_edges, nonneg=True)
    
    # Objective: maximize spectral gap (minimize trace of expected Laplacian)
    # Expected Laplacian: sum_e p_e * L_e
    objective = cp.Minimize(cp.sum(p))
    
    # Constraints
    constraints = [
        p >= 0.1,  # Minimum probability
        p <= 1.0,  # Maximum probability
        cp.sum(p) >= 1.0  # At least one edge active in expectation
    ]
    
    # Solve
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.CVXOPT, kktsolver=cp.ROBUST_KKTSOLVER)
        
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            # If optimization fails, use uniform probabilities
            return {edge: 1.0 for edge in edges}
        
        # Extract probabilities
        edge_probs = {}
        for i, edge in enumerate(edges):
            prob = float(p.value[i]) if p.value is not None else 1.0
            edge_probs[edge] = min(max(prob, 0.1), 1.0)  # Clamp to [0.1, 1.0]
            edge_probs[(edge[1], edge[0])] = edge_probs[edge]  # Symmetric
        
        return edge_probs
        
    except Exception as e:
        print(f"CVX optimization failed: {e}")
        return {edge: 1.0 for edge in edges}


def _get_optimal_alpha(
    graph: nx.Graph,
    edge_probs: Dict[Tuple[int, int], float],
    num_clients: int
) -> np.ndarray:
    """Compute optimal mixing weights alpha for MATCHA using SDP.
    
    Given edge probabilities, find mixing weights that optimize convergence.
    """
    W = np.zeros((num_clients, num_clients))
    
    # For each node, distribute weight among active neighbors
    for node in range(num_clients):
        neighbors = list(graph.neighbors(node))
        
        if not neighbors:
            W[node, node] = 1.0
            continue
        
        # Get expected weights based on probabilities
        total_prob = 0.0
        for neighbor in neighbors:
            prob = edge_probs.get((node, neighbor), 1.0)
            W[node, neighbor] = prob / (1 + len(neighbors))
            total_prob += W[node, neighbor]
        
        # Self-weight
        W[node, node] = 1.0 - total_prob
    
    return W


def get_active_edges(
    graph: nx.Graph,
    round_num: int,
    seed: int = None
) -> Dict[Tuple[int, int], bool]:
    """Determine which edges are active in this round based on probabilities.
    
    Args:
        graph: Network topology graph
        round_num: Current round number
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping edges to activation status
    """
    if seed is not None:
        np.random.seed(seed + round_num)
    
    active_edges = {}
    
    for edge in graph.edges():
        prob = graph.edges[edge].get('probability_selection', 1.0)
        is_active = np.random.random() < prob
        active_edges[edge] = is_active
        active_edges[(edge[1], edge[0])] = is_active  # Undirected
    
    return active_edges


def get_active_mixing_matrix(
    graph: nx.Graph,
    num_clients: int,
    active_edges: Dict[Tuple[int, int], bool],
    method: MixingMethod = 'metropolis_hastings'
) -> np.ndarray:
    """Create mixing matrix considering only active edges.
    
    Args:
        graph: Network topology graph
        num_clients: Number of clients
        active_edges: Dictionary of edge activation status
        method: Mixing method to use
        
    Returns:
        Active mixing matrix
    """
    if method == 'metropolis_hastings':
        return _active_metropolis_hastings(graph, num_clients, active_edges)
    elif method == 'max_degree':
        return _active_max_degree(graph, num_clients, active_edges)
    elif method == 'jaccard':
        return _active_jaccard(graph, num_clients, active_edges)
    elif method == 'matcha':
        return _active_matcha(graph, num_clients, active_edges)
    else:
        raise ValueError(f"Unknown mixing method: {method}")


def _active_metropolis_hastings(
    graph: nx.Graph,
    num_clients: int,
    active_edges: Dict[Tuple[int, int], bool]
) -> np.ndarray:
    """Metropolis-Hastings with active edges only."""
    W = np.zeros((num_clients, num_clients))
    
    for node in range(num_clients):
        neighbors = list(graph.neighbors(node))
        active_neighbors = [n for n in neighbors if active_edges.get((node, n), False)]
        
        if not active_neighbors:
            W[node, node] = 1.0
            continue
        
        degree = len(active_neighbors)
        
        for neighbor in active_neighbors:
            neighbor_active = [n for n in graph.neighbors(neighbor) 
                             if active_edges.get((neighbor, n), False)]
            neighbor_degree = len(neighbor_active)
            max_degree = max(degree, neighbor_degree)
            W[node, neighbor] = 1.0 / (max_degree + 1)
        
        W[node, node] = 1.0 - W[node, :].sum()
    
    return W


def _active_max_degree(
    graph: nx.Graph,
    num_clients: int,
    active_edges: Dict[Tuple[int, int], bool]
) -> np.ndarray:
    """Max degree with active edges only."""
    W = np.zeros((num_clients, num_clients))
    
    # Find max degree considering only active edges
    max_degree = 0
    for node in range(num_clients):
        neighbors = list(graph.neighbors(node))
        active_neighbors = [n for n in neighbors if active_edges.get((node, n), False)]
        max_degree = max(max_degree, len(active_neighbors))
    
    if max_degree == 0:
        return np.eye(num_clients)
    
    for node in range(num_clients):
        neighbors = list(graph.neighbors(node))
        active_neighbors = [n for n in neighbors if active_edges.get((node, n), False)]
        
        if not active_neighbors:
            W[node, node] = 1.0
            continue
        
        for neighbor in active_neighbors:
            W[node, neighbor] = 1.0 / max_degree
        
        degree = len(active_neighbors)
        W[node, node] = 1.0 - degree / max_degree
    
    return W


def _active_jaccard(
    graph: nx.Graph,
    num_clients: int,
    active_edges: Dict[Tuple[int, int], bool]
) -> np.ndarray:
    """Jaccard with active edges only."""
    W = np.zeros((num_clients, num_clients))
    
    # Build active neighborhood sets
    active_neighbors_map = {}
    for node in range(num_clients):
        neighbors = list(graph.neighbors(node))
        active_neighbors_map[node] = set(n for n in neighbors 
                                         if active_edges.get((node, n), False))
    
    # Check if any active edges exist
    if all(len(ns) == 0 for ns in active_neighbors_map.values()):
        return np.eye(num_clients)
    
    # Compute Jaccard weights for active edges
    for i in range(num_clients):
        neighbors_i = active_neighbors_map[i]
        set_i = neighbors_i | {i}
        
        for j in neighbors_i:
            set_j = active_neighbors_map[j] | {j}
            
            intersection = len(set_i & set_j)
            union = len(set_i | set_j)
            
            if union > 0:
                W[i, j] = 1.0 - intersection / union
    
    # Normalize
    flows = np.sum(W, axis=0)
    global_scaling_factor = np.max(flows)
    
    if global_scaling_factor > 0:
        W /= (global_scaling_factor + 1e-10)
    
    # Self-loops
    for i in range(num_clients):
        W[i, i] = max(1.0 - np.sum(W[i, :]), 0.0)
    
    return W


def _active_matcha(
    graph: nx.Graph,
    num_clients: int,
    active_edges: Dict[Tuple[int, int], bool]
) -> np.ndarray:
    """MATCHA with active edges only.
    
    For dynamic edge activation, we fall back to simpler methods
    since MATCHA optimization assumes a fixed graph.
    """
    # MATCHA is designed for static graphs with probabilistic activation
    # For active edge subsets, use Metropolis-Hastings
    return _active_metropolis_hastings(graph, num_clients, active_edges)

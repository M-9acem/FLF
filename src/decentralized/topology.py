"""Network topology creation for P2P federated learning."""

import importlib.util
import os
from pathlib import Path
import networkx as nx
import numpy as np
from typing import Tuple, Dict, Literal, Any

# Type alias for mixing methods
MixingMethod = Literal['metropolis_hastings', 'max_degree', 'jaccard', 'jaccard_dissimilarity', 'matcha']

# Module-level cache for MATCHA artifacts (avoid re-solving every round)
_matcha_cache: Dict[int, np.ndarray] = {}
_matcha_meta_cache: Dict[int, Dict[str, Any]] = {}


def _matcha_debug_enabled(graph: nx.Graph = None) -> bool:
    env = os.getenv('MATCHA_DEBUG', '').strip().lower()
    if env in {'1', 'true', 'yes', 'on'}:
        return True
    if graph is not None and bool(graph.graph.get('matcha_debug', False)):
        return True
    return False


def _matcha_log(msg: str, graph: nx.Graph = None) -> None:
    if _matcha_debug_enabled(graph):
        print(f"[MATCHA] {msg}")


def create_two_cluster_topology(
    num_clients: int,
    main_link_prob: float = 1.0,
    border_link_prob: float = 1.0,
    intra_cluster_prob: float = 0.8,
    intra_cluster_communication: bool = True
) -> nx.Graph:
    """Create a symmetrical two-cluster network topology.
    
    This creates two identically structured clusters with a bridge connection between them.
    The clusters are perfectly symmetrical - they have the same internal structure.
    
    Args:
        num_clients: Total number of clients (should be even for perfect symmetry)
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
    
    # Split into two clusters evenly
    cluster_size = num_clients // 2
    cluster1 = list(range(cluster_size))
    cluster2 = list(range(cluster_size, num_clients))
    
    print(f"Creating symmetrical clusters: Cluster 0 ({len(cluster1)} clients), Cluster 1 ({len(cluster2)} clients)")
    
    if intra_cluster_communication:
        # All nodes in a cluster are interconnected (current behavior)
        cluster1_edges = []
        for i in range(len(cluster1)):
            for j in range(i + 1, len(cluster1)):
                if np.random.random() < intra_cluster_prob:
                    cluster1_edges.append((i, j))
        # Add cluster 1 edges (using actual node IDs)
        for i, j in cluster1_edges:
            G.add_edge(cluster1[i], cluster1[j], probability_selection=intra_cluster_prob)
        # Mirror edges to cluster 2 (maintaining symmetry)
        for i, j in cluster1_edges:
            G.add_edge(cluster2[i], cluster2[j], probability_selection=intra_cluster_prob)
    else:
        # Only edge node in each cluster connects to others in the cluster (star topology)
        # Choose the first node in each cluster as the edge node
        edge1 = cluster1[0]
        edge2 = cluster2[0]
        for node in cluster1:
            if node != edge1:
                G.add_edge(edge1, node, probability_selection=intra_cluster_prob)
        for node in cluster2:
            if node != edge2:
                G.add_edge(edge2, node, probability_selection=intra_cluster_prob)
    
    # Find center nodes (highest degree in each cluster)
    # Due to symmetry, centers will be at the same relative position
    cluster1_degrees = [(node, G.degree(node)) for node in cluster1]
    cluster2_degrees = [(node, G.degree(node)) for node in cluster2]
    
    center1 = max(cluster1_degrees, key=lambda x: x[1])[0]
    center2 = max(cluster2_degrees, key=lambda x: x[1])[0]
    
    print(f"  Cluster 0 center: Client {center1} (degree {G.degree(center1)})")
    print(f"  Cluster 1 center: Client {center2} (degree {G.degree(center2)})")
    
    # Add ONLY ONE bridge link between the two clusters (connecting centers)
    G.add_edge(center1, center2, probability_selection=main_link_prob)
    print(f"  Bridge link: {center1}-{center2} (single connection between clusters)")
    
    # Verify symmetry
    cluster1_edge_count = len([(u, v) for u, v in G.edges() if u in cluster1 and v in cluster1])
    cluster2_edge_count = len([(u, v) for u, v in G.edges() if u in cluster2 and v in cluster2])
    inter_cluster_edges = len([(u, v) for u, v in G.edges() if (u in cluster1 and v in cluster2) or (u in cluster2 and v in cluster1)])
    print(f"  Internal edges: Cluster 0 = {cluster1_edge_count}, Cluster 1 = {cluster2_edge_count}")
    print(f"  Inter-cluster edges: {inter_cluster_edges}")
    
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
    elif method == 'jaccard_dissimilarity':
        return _jaccard_dissimilarity_weights(graph, num_clients)
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
    """
    Jaccard dissimilarity mixing weights (standalone variant).

    For each edge (i,j), using closed neighborhoods N[i] = {i} ∪ neighbors(i):
        J(i,j)  = |N[i] ∩ N[j]| / |N[i] ∪ N[j]|   (Jaccard similarity)
        y_ij    = 1 - J(i,j)                          (dissimilarity → weight)

    Normalization:
        - Per-node row normalization if row sum > 1
        - Symmetrize: w_ij = min(y_ij, y_ji)
        - Self-loop:  w_ii = 1 - sum of off-diagonal row weights
    """
    if len(graph.edges()) == 0:
        return np.eye(num_clients)

    # Step 1: compute raw directed weights
    Y = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        set_i = set(graph.neighbors(i)) | {i}
        for j in graph.neighbors(i):
            set_j = set(graph.neighbors(j)) | {j}
            intersection = len(set_i & set_j)
            union        = len(set_i | set_j)
            Y[i, j] = 1.0 - intersection / union  # in [0, 1]

    # Step 2: per-node row normalization (only rescales if sum > 1)
    for i in range(num_clients):
        row_sum = Y[i, :].sum()
        if row_sum > 1.0:
            Y[i, :] /= row_sum

    # Step 3: symmetrize via min, then self-loops
    W = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in graph.neighbors(i):
            W[i, j] = min(Y[i, j], Y[j, i])
        W[i, i] = max(1.0 - W[i, :].sum(), 0.0)

    # Sanity check (remove in production)
    assert np.allclose(W, W.T, atol=1e-6), "W is not symmetric"
    assert np.all(W >= 0),                 "W has negative entries"
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-6), "Rows don't sum to 1"

    return W

def _jaccard_dissimilarity_weights(graph: nx.Graph, num_clients: int) -> np.ndarray:
    """
    Neighborhood Algorithm mixing weights (Avrachenkov et al., 2011).
    Uses closed neighborhoods N[i] = {i} ∪ neighbors(i).

    For each edge (i, j):
        y_ij = 1 - |N[i] ∩ N[j]| / (1 + 2*min(|N[i]|, |N[j]|) - |N[i] ∩ N[j]|)

    Then per Algorithm 1:
      - Per-node row normalization if row sum > 1
      - Symmetrize: w_ij = min(y_ij, y_ji)
      - Self-loop: w_ii = 1 - sum of off-diagonal weights
    """
    if len(graph.edges()) == 0:
        return np.eye(num_clients)

    # Step 2: compute raw y_ij for each directed edge
    Y = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        set_i = set(graph.neighbors(i)) | {i}
        for j in graph.neighbors(i):
            set_j = set(graph.neighbors(j)) | {j}
            intersection = len(set_i & set_j)
            denom = 1 + 2 * min(len(set_i), len(set_j)) - intersection
            Y[i, j] = 1.0 - intersection / denom

    # Step 3: per-node row normalization (only if row sum > 1)
    for i in range(num_clients):
        row_sum = np.sum(Y[i, :])
        if row_sum > 1.0:
            Y[i, :] /= row_sum

    # Step 5: symmetrize with min, then self-loops
    W = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in graph.neighbors(i):
            W[i, j] = min(Y[i, j], Y[j, i])
        W[i, i] = max(1.0 - np.sum(W[i, :]), 0.0)

    return W

def _matcha_weights(graph: nx.Graph, num_clients: int) -> np.ndarray:
    """MATCHA expected mixing matrix using cloned MATCHA MatchaProcessor."""
    if len(graph.edges()) == 0:
        return np.eye(num_clients)

    cache_key = id(graph)
    if cache_key in _matcha_cache and cache_key in _matcha_meta_cache:
        _matcha_log(f"using cached expected W for graph_id={cache_key}", graph)
        return _matcha_cache[cache_key]

    processor = _get_matcha_processor(graph, num_clients)
    probs = np.array(processor.probabilities, dtype=float)
    L_matrices = [np.asarray(L, dtype=float) for L in processor.L_matrices]
    alpha = float(processor.neighbor_weight)

    # Expected MATCHA matrix: E[W_t] = I - alpha * E[L_t]
    mean_L = np.zeros((num_clients, num_clients), dtype=float)
    for i in range(len(L_matrices)):
        mean_L += probs[i] * L_matrices[i]

    W = np.eye(num_clients) - alpha * mean_L
    W = np.maximum(W, 0.0)
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    W = W / row_sums

    _matcha_cache[cache_key] = W
    _matcha_meta_cache[cache_key] = {
        'alpha': alpha,
        'processor': processor,
    }
    _matcha_log(
        (
            f"computed expected W: graph_id={cache_key}, subgraphs={len(L_matrices)}, "
            f"alpha={alpha:.6f}, probs={np.round(probs, 6).tolist()}, "
            f"row_sums={np.round(W.sum(axis=1), 6).tolist()}"
        ),
        graph,
    )
    return W


def _calculate_matcha_iterations(graph: nx.Graph) -> int:
    """Calculate total communication iterations from gossip schedule and rounds."""
    rounds = int(graph.graph.get('rounds', 10))
    gossip_schedule_str = graph.graph.get('gossip_schedule', None)
    gossip_steps = int(graph.graph.get('gossip_steps', 1))
    
    if gossip_schedule_str:
        # Parse schedule "5:0,3:1,1:2" into list of (steps, from_round) tuples
        schedule = []
        for item in gossip_schedule_str.split(','):
            steps, from_round = map(int, item.split(':'))
            schedule.append((steps, from_round))
        
        # Sort by from_round for correct order
        schedule.sort(key=lambda x: x[1])
        
        # Calculate total iterations across all rounds
        total_iterations = 0
        for round_num in range(rounds):
            # Find which schedule entry applies to this round
            applicable_steps = gossip_steps  # default fallback
            for steps, from_round in schedule:
                if from_round <= round_num:
                    applicable_steps = steps
            total_iterations += applicable_steps
        return total_iterations
    else:
        return rounds * gossip_steps


def _get_matcha_processor(graph: nx.Graph, num_clients: int):
    """Create/retrieve the original MATCHA MatchaProcessor from cloned repo."""
    cache_key = id(graph)
    cached = _matcha_meta_cache.get(cache_key, {}).get('processor')
    if cached is not None:
        _matcha_log(f"using cached processor for graph_id={cache_key}", graph)
        return cached

    matcha_file = Path(__file__).resolve().parent / 'MATCHA' / 'graph_manager.py'
    spec = importlib.util.spec_from_file_location('matcha_graph_manager', matcha_file)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load MATCHA from {matcha_file}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    MatchaProcessor = mod.MatchaProcessor

    comm_budget = float(graph.graph.get('matcha_comm_budget', 1.0))
    iterations = _calculate_matcha_iterations(graph)  # Based on gossip schedule and rounds
    processor = MatchaProcessor(
        base_graph=nx.Graph(graph),
        commBudget=comm_budget,
        rank=0,
        size=num_clients,
        iterations=iterations,
        issubgraph=False,
    )
    _matcha_log(
        (
            f"processor created: graph_id={cache_key}, clients={num_clients}, "
            f"comm_budget={comm_budget}, iterations={iterations}, "
            f"subgraphs={len(processor.subGraphs)}, "
            f"probs={np.round(np.array(processor.probabilities, dtype=float), 6).tolist()}, "
            f"alpha={float(processor.neighbor_weight):.6f}"
        ),
        graph,
    )
    _matcha_meta_cache.setdefault(cache_key, {})['processor'] = processor
    return processor


def get_active_edges(
    graph: nx.Graph,
    round_num: int,
    seed: int = None,
    method: MixingMethod = 'metropolis_hastings',
    gossip_step: int = 0
) -> Dict[Tuple[int, int], bool]:
    """Determine which edges are active in this round based on probabilities.
    
    Args:
        graph: Network topology graph
        round_num: Current round number
        seed: Random seed for reproducibility
        method: Mixing method; MATCHA uses subgraph-level activation
        gossip_step: Gossip step index within the current round
        
    Returns:
        Dictionary mapping edges to activation status
    """
    if seed is not None:
        np.random.seed(seed + round_num * 1000 + gossip_step)

    if method == 'matcha':
        return _get_matcha_active_edges(graph, round_num, gossip_step, seed)
    
    active_edges = {}
    
    for edge in graph.edges():
        prob = graph.edges[edge].get('probability_selection', 1.0)
        is_active = np.random.random() < prob
        active_edges[edge] = is_active
        active_edges[(edge[1], edge[0])] = is_active  # Undirected
    
    return active_edges


def _get_matcha_active_edges(
    graph: nx.Graph,
    round_num: int,
    gossip_step: int,
    seed: int = None,
) -> Dict[Tuple[int, int], bool]:
    """Sample active edges according to MATCHA subgraph probabilities.

    All edges inside an activated subgraph are activated together.
    """
    num_clients = graph.number_of_nodes()
    processor = _get_matcha_processor(graph, num_clients)

    active_edges: Dict[Tuple[int, int], bool] = {}
    subgraphs = processor.subGraphs
    probs = np.array(processor.probabilities, dtype=float)

    for i, edges in enumerate(subgraphs):
        is_on = np.random.random() < float(probs[i])
        for (u, v) in edges:
            active_edges[(u, v)] = is_on
            active_edges[(v, u)] = is_on

    # If an edge is somehow outside decomposition, keep it always active.
    for u, v in graph.edges():
        if (u, v) not in active_edges:
            active_edges[(u, v)] = True
            active_edges[(v, u)] = True

    active_undirected = sum(
        1 for (u, v), on in active_edges.items()
        if on and u < v
    )
    total_undirected = graph.number_of_edges()
    probs_preview = np.round(probs.astype(float), 4).tolist()
    _matcha_log(
        (
            f"round={round_num}, gossip_step={gossip_step}, "
            f"active_edges={active_undirected}/{total_undirected}, "
            f"probabilities={probs_preview}"
        ),
        graph,
    )

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
    elif method == 'jaccard_dissimilarity':
        return _active_jaccard_dissimilarity(graph, num_clients, active_edges)
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
    """Jaccard dissimilarity with active edges only.

    Mirrors the same Algorithm 1 steps as _jaccard_weights:
      1. Compute raw y_ij = 1 - |N[i] ∩ N[j]| / |N[i] ∪ N[j]| (closed neighborhoods)
      2. Per-node row normalisation if row sum > 1
      3. Symmetrize: w_ij = min(y_ij, y_ji)
      4. Self-loop: w_ii = 1 - sum of off-diagonal weights
    """
    # Build active closed neighborhood sets
    active_neighbors_map = {
        node: set(
            n for n in graph.neighbors(node)
            if active_edges.get((node, n), False)
        )
        for node in range(num_clients)
    }

    if all(len(ns) == 0 for ns in active_neighbors_map.values()):
        return np.eye(num_clients)

    # Step 1: raw directed weights
    Y = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        set_i = active_neighbors_map[i] | {i}
        for j in active_neighbors_map[i]:
            set_j = active_neighbors_map[j] | {j}
            intersection = len(set_i & set_j)
            union        = len(set_i | set_j)
            Y[i, j] = 1.0 - intersection / union

    # Step 2: per-node row normalisation (only if sum > 1)
    for i in range(num_clients):
        row_sum = Y[i, :].sum()
        if row_sum > 1.0:
            Y[i, :] /= row_sum

    # Step 3 & 4: symmetrize via min, then self-loops
    W = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in active_neighbors_map[i]:
            W[i, j] = min(Y[i, j], Y[j, i])
        W[i, i] = max(1.0 - W[i, :].sum(), 0.0)

    return W


def _active_jaccard_dissimilarity(
    graph: nx.Graph,
    num_clients: int,
    active_edges: Dict[Tuple[int, int], bool]
) -> np.ndarray:
    """Neighborhood Algorithm (Avrachenkov et al., 2011) with active edges only.

    Uses closed neighborhoods restricted to currently active edges:
        N_active[i] = {i} ∪ {j : (i,j) active}
    Mirrors the same Algorithm 1 steps as _jaccard_dissimilarity_weights:
      1. Compute raw y_ij for each active directed edge
      2. Per-node row normalisation if row sum > 1
      3. Symmetrize: w_ij = min(y_ij, y_ji)
      4. Self-loop: w_ii = 1 - sum of off-diagonal weights
    """
    # Build active closed neighborhood sets
    active_neighbors_map = {
        node: set(
            n for n in graph.neighbors(node)
            if active_edges.get((node, n), False)
        )
        for node in range(num_clients)
    }

    if all(len(ns) == 0 for ns in active_neighbors_map.values()):
        return np.eye(num_clients)

    # Step 1: raw y_ij for each active directed edge
    Y = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        set_i = active_neighbors_map[i] | {i}
        for j in active_neighbors_map[i]:
            set_j = active_neighbors_map[j] | {j}
            intersection = len(set_i & set_j)
            denom = 1 + 2 * min(len(set_i), len(set_j)) - intersection
            Y[i, j] = 1.0 - intersection / denom

    # Step 2: per-node row normalisation (only if row sum > 1)
    for i in range(num_clients):
        row_sum = np.sum(Y[i, :])
        if row_sum > 1.0:
            Y[i, :] /= row_sum

    # Step 3 & 4: symmetrize with min, then self-loops
    W = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in active_neighbors_map[i]:
            W[i, j] = min(Y[i, j], Y[j, i])
        W[i, i] = max(1.0 - np.sum(W[i, :]), 0.0)

    return W


def _active_matcha(
    graph: nx.Graph,
    num_clients: int,
    active_edges: Dict[Tuple[int, int], bool]
) -> np.ndarray:
    """MATCHA with active edges only.
    
    Computes the full nonnegative MATCHA mixing matrix once (cached), then for
    each round keeps only currently active edge weights and redistributes the
    removed weight to self-loops so rows still sum to 1.
    """
    global _matcha_cache
    
    # Compute full MATCHA weights once and cache by graph identity
    cache_key = id(graph)
    if cache_key not in _matcha_cache:
        _matcha_cache[cache_key] = _matcha_weights(graph, num_clients)
    
    W_full = _matcha_cache[cache_key]

    # Build active mixing matrix from the cached, row-normalized MATCHA weights.
    # This preserves the nonnegative expected weights from _matcha_weights while
    # moving any removed inactive-edge mass back to the self-loop.
    W = np.zeros((num_clients, num_clients))
    
    for node in range(num_clients):
        neighbors = list(graph.neighbors(node))
        active_neighbors = [n for n in neighbors
                            if active_edges.get((node, n), False)]
        
        for neighbor in active_neighbors:
            W[node, neighbor] = W_full[node, neighbor]
        
        # Self-weight absorbs weight from inactive edges
        W[node, node] = max(1.0 - np.sum(W[node, :]), 0.0)

    _matcha_log(
        (
            f"active W built: row_sums={np.round(W.sum(axis=1), 6).tolist()}, "
            f"diag={np.round(np.diag(W), 6).tolist()}"
        ),
        graph,
    )
    
    return W

import torch
from torch import Tensor
from torch_geometric.utils import to_dense_adj
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from numba import jit


def get_adj(edge_index: Tensor, set_diag: bool = True, symmetric_normalize: bool = True, device='cpu') -> Tensor:
    """
    Generate a dense adjacency matrix from edge indices with optional diagonal setting and symmetric normalization.

    Parameters:
    edge_index (torch.Tensor): Edge indices of the graph.
    set_diag (bool): If True, set the diagonal to 1. Defaults to True.
    symmetric_normalize (bool): If True, apply symmetric normalization. Defaults to True.

    Returns:
    torch.Tensor: Dense adjacency matrix.
    """
    adj = to_dense_adj(edge_index).squeeze().to(device)

    if set_diag:
        adj += torch.eye(adj.size(0), device=device)

    if symmetric_normalize:
        D = torch.diag(adj.sum(dim=1)).to(device)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D.diagonal())).to(device)
        adj = torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)
        adj[torch.isnan(adj)] = 0.0

    return adj

def setup_logging(log_file):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_format = '%(asctime)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format,
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])


def log_message(message):
    logging.info(message)
    # if args.display:
    # print(message)

def compute_diameter(original_graph=None, rewired_graph=None):  # Time complexity: O(n^2 + nm)
    """
    Compute and return the diameters of the original and rewired graphs.
    """

    if original_graph is None and rewired_graph is None:
        raise ValueError("At least one of original_graph or rewired_graph must be provided.")

    def get_largest_component_diameter(graph):
        """Compute the diameter of the largest connected component in a graph."""
        # Find connected components
        components = list(nx.connected_components(graph))
        if not components:
            return float('inf')  # No components, return infinity

        # Find the largest component
        largest_component = max(components, key=len)

        # Create a subgraph of the largest component
        subgraph = graph.subgraph(largest_component)

        # Return the diameter of the largest component
        return nx.diameter(subgraph)

    # Convert the original graph to NetworkX and compute its diameter
    if original_graph is not None:
        original_graph = to_networkx(original_graph, to_undirected=True)
        if nx.is_connected(original_graph):
            original_diameter = nx.diameter(original_graph)
        else:
            original_diameter = get_largest_component_diameter(original_graph)
        return original_diameter

    # Get the rewired data and convert to NetworkX
    if rewired_graph is not None:
        rewired_graph = to_networkx(rewired_graph, to_undirected=True)
        if nx.is_connected(rewired_graph):
            rewired_diameter = nx.diameter(rewired_graph)
        else:
            rewired_diameter = get_largest_component_diameter(rewired_graph)
        return rewired_diameter


def gnm_random_graph_v2(m, seed=None, directed=False, n=None, graph=None):
    """
    Returns a G(n,m) random graph with O(m) time complexity.
    If an existing graph is provided, it will add m new random edges
    without duplicating existing edges.

    Args:
        m: Number of new edges to add.
        seed: Random seed for reproducibility.
        directed: Whether the graph is directed or not.
        n: (Optional) Number of nodes. If not provided, inferred from existing_edges.
        graph: (Optional) PyG Data object.

    Returns:
        A PyG Data object with the updated edges.
    """

    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    existing_edges = [tuple(edge) for edge in graph.edge_index.T.tolist()] if graph is not None else []

    # Initialize edge set
    edge_set = set(existing_edges) if graph else set()

    # Infer the number of nodes from the existing edges, if not provided
    if graph:
        nodes_from_edges = set([node for edge in existing_edges for node in edge])
        inferred_n = max(nodes_from_edges) + 1
        n = max(n, inferred_n) if n is not None else inferred_n
    else:
        assert n is not None, "Number of nodes must be provided if no existing edges are given."

    # Maximum number of possible edges (account for undirected graph if needed)
    max_edges = n * (n - 1) if directed else n * (n - 1) // 2

    # Total edges that can be added (considering existing ones)
    if directed:
        remaining_edges = max_edges - len(edge_set)
    else:
        remaining_edges = max_edges - len(edge_set) // 2
    m = min(m, remaining_edges)  # We can't add more than the remaining available edges

    # Add nodes
    nlist = list(range(n))
    existing_edge_count = len(existing_edges) // (2 if not directed else 1)
    edge_set_count = len(edge_set) // (2 if not directed else 1)

    while edge_set_count < m + existing_edge_count:
        # Generate random edge u, v
        u, v = random.sample(nlist, 2)  # Randomly select two distinct nodes without replacement
        # Create edge respecting direction if required
        edge = (u, v)
        # Add edge to set only if it's not already in the existing edges
        if edge not in edge_set:
            edge_set.add(edge)
            if not directed:
                edge_set.add((v, u))
            edge_set_count += 1

    edge_list = list(edge_set)

    edge_type = torch.zeros(len(edge_list), dtype=torch.long)
    for i, edge in enumerate(edge_list):
        if edge in existing_edges:
            edge_type[i] = 0
        else:
            edge_type[i] = 1

    # Convert edge list to tensor of shape (2, num_edges)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Return as PyG Data object
    # return Data(x=graph.x, edge_index=edge_index, edge_type=edge_type, y=graph.y, num_nodes=n)
    return edge_index, edge_type, graph.x, graph.y, n # remember to fix

def calculate_target_probability(diameter: int):
    max_prob = 0.99
    # min_prob = 0.01
    beta = 0.25

    target_probability = max_prob * (1 - math.exp(-beta * (diameter - 1)))

    return target_probability

def calculate_num_random_edges(dataset, diameter: int, n: int):
    binomial_d = math.comb(diameter, 2)
    binomial_n = math.comb(n, 2)

    existing_edges = (dataset.edge_index.shape[1] - dataset.num_nodes) // 2
    target_probability = calculate_target_probability(diameter)

    P = binomial_d / (binomial_n - existing_edges)
    k = np.log(1 - target_probability) / np.log(1 - P) # TODO Debug why we get division by zero

    return math.ceil(k)


# @jit(nopython=True)
def dirichlet_energy(X, edge_index):
    # computes Dirichlet energy of a vector field X with respect to a graph with a given edge index
    n = X.shape[0]
    m = len(edge_index[0])
    l = X.shape[1]
    degrees = np.zeros(n)
    for I in range(m):
        u = edge_index[0][I]
        degrees[u] += 1
    y = np.linalg.norm(X.flatten()) ** 2
    for I in range(m):
        for i in range(l):
            u = edge_index[0][I]
            v = edge_index[1][I]
            y -= X[u][i] * X[v][i] / (degrees[u] * degrees[v]) ** 0.5
    return y


def dirichlet_normalized(X, edge_index):
    energy = dirichlet_energy(X, edge_index)
    norm_squared = sum(sum(X ** 2))
    return energy / norm_squared


def generate_qq_plots(decay_rates, path):
    plt.figure()
    stats.probplot(decay_rates, dist="norm", plot=plt)
    plt.title(f'Q-Q Plot)')
    plt.savefig(path)
    plt.close()

import torch
from torch import Tensor
from torch_geometric.utils import to_dense_adj
import networkx as nx
from torch_geometric.utils import to_networkx
from typing import List
import random
import math
import tqdm
from args import get_args
from typing import Any, Union
import os.path as osp
from torch_geometric.datasets import TUDataset, LRGBDataset
from torch_geometric.utils import degree
from torch_geometric.utils import add_self_loops
import torch_geometric.transforms as T
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

class CustomTransform(object):
    def __call__(self, data):
        if data.x is None:  # Only set features if they don't exist
            data.x = torch.ones((data.num_nodes, 1), dtype=torch.float32)
        return data


class AddSelfLoopsTransform(object):
    def __call__(self, data):
        data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
        return data

def get_dataset(args: Any) -> Union[TUDataset, LRGBDataset]:
    """
    Load the specified dataset.

    Parameters:
    dataset_name (str): Name of the dataset.

    Returns:
    torch_geometric.datasets.Dataset: Loaded dataset.
    """
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', args.dataset)

    if args.dataset in ['MUTAG', 'PROTEINS', 'DD', 'ENZYMES']:
        return TUDataset(path, name=args.dataset, transform=AddSelfLoopsTransform())
    elif args.dataset in ['IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'COLLAB']:
        return TUDataset(path, name=args.dataset, transform=T.Compose([CustomTransform(), AddSelfLoopsTransform()]))
    elif args.dataset in ['Peptides-func', 'PCQM-Contact', 'Peptides-struct']:
        return LRGBDataset(path, name=args.dataset, transform=AddSelfLoopsTransform())
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

def dataset_statistics():
    args = get_args()
    dataset = get_dataset(args)

    num_nodes_list = []
    degree_list = []
    diameter_list = []
    num_edges_list = []
    density_list = []
    counter_of_connected = 0
    for data in tqdm(dataset, desc='Processing', total=len(dataset)):
        num_nodes = data.num_nodes
        num_edges = data.edge_index.shape[1] // 2
        max_deg = degree(data.edge_index[0], num_nodes, dtype=torch.long).max()
        diameter = compute_diameter(data)
        num_nodes_list.append(num_nodes)
        num_edges_list.append(num_edges)
        degree_list.append(max_deg)

        # --- density for this graph ---------------------------------------------
        if num_nodes > 1:
            density = 2.0 * num_edges / (num_nodes * (num_nodes - 1))
        else:                              # isolated single-node graph
            density = 0.0
        density_list.append(density)

        if diameter != float('inf'):
            counter_of_connected += 1
            diameter_list.append(diameter)

    mean_deg = torch.tensor(degree_list).float().mean()
    max_deg = torch.tensor(degree_list).max()
    min_deg = torch.tensor(degree_list).min()
    print(f'Mean Degree: {mean_deg}')
    print(f'Max Degree: {max_deg}')
    print(f'Min Degree: {min_deg}')

    mean_num_nodes = torch.tensor(num_nodes_list).float().mean().round().long().item()
    max_num_nodes = torch.tensor(num_nodes_list).float().max().round().long().item()
    min_num_nodes = torch.tensor(num_nodes_list).float().min().round().long().item()
    print(f'Mean number of nodes: {mean_num_nodes}')
    print(f'Max number of nodes: {max_num_nodes}')
    print(f'Min number of nodes: {min_num_nodes}')
    print(f'Number of graphs: {len(dataset)}')

    mean_num_edges = torch.tensor(num_edges_list).float().mean().round().long().item()
    max_num_edges = torch.tensor(num_edges_list).float().max().round().long().item()
    min_num_edges = torch.tensor(num_edges_list).float().min().round().long().item()

    print(f'Mean number of edges: {mean_num_edges}')
    print(f'Max number of edges: {max_num_edges}')
    print(f'Min number of edges: {min_num_edges}')

    mean_diameter = torch.tensor(diameter_list).float().mean().item()
    max_diameter = torch.tensor(diameter_list).float().max().long().item()
    min_diameter = torch.tensor(diameter_list).float().min().long().item()

    print(f'Mean diameter: {mean_diameter}')
    print(f'Max diameter: {max_diameter}')
    print(f'Min diameter: {min_diameter}')

    # === NEW: dataset-level density summary ====================================
    mean_density = torch.tensor(density_list).float().mean().item()
    max_density  = torch.tensor(density_list).max().item()
    min_density  = torch.tensor(density_list).min().item()

    print(f'Mean density: {mean_density:.4f}')
    print(f'Max  density: {max_density:.4f}')
    print(f'Min  density: {min_density:.4f}')

    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of disconnected graphs: {len(dataset) - counter_of_connected}')

def num_connected_components(data):
    g = to_networkx(data, to_undirected=True)
    return nx.number_connected_components(g)

def average_connected_components():
    args = get_args()
    dataset = get_dataset(args)

    total_components = 0
    for data in tqdm(dataset, desc='Counting components'):
        total_components += num_connected_components(data)

    avg_components = total_components / len(dataset)
    print(f"Average number of connected components: {avg_components}")

# def plot_distributions(args):
#     """
#     Plot the distribution of Y_pre, Y_mean, Y_std, and Y_skew with LaTeX-rendered labels.
#     """
#
#     device = torch.device('cpu')
#     datasets = load_datasets(args)
#     sns.set_theme(style="white")
#     plt.rcParams['text.usetex'] = True
#     plt.rcParams['font.family'] = 'serif'
#     plt.rcParams['font.serif'] = ['Computer Modern Roman']
#     plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
#
#     plt.figure(figsize=(12, 8))
#
#     for name, dataset in datasets.items():
#         Y_pre_list, Y_ave_list, Y_std_list, Y_max_list = compute_metrics_for_dataset(dataset, name, args,
#                                                                                       device=device)
#         metrics = {
#             r"$\mathbf{Y}_{\textbf{pre}}$": Y_pre_list,
#             r"$\mathbf{Y}_{\textbf{mean}}$": Y_ave_list,
#             r"$\mathbf{Y}_{\textbf{std}}$": Y_std_list,
#             r"$\mathbf{Y}_{\textbf{max}}$": Y_max_list
#         }
#         for i, (label, values) in enumerate(metrics.items(), 1):
#             plt.subplot(2, 2, i)
#             sns.histplot(values, kde=True, bins=30)
#             plt.xlabel(label, fontsize=14)
#             plt.ylabel(r"\text{Density}", fontsize=14)
#             plt.xticks(fontsize=16)
#             plt.yticks(fontsize=16)
#
#         plt.tight_layout()
#
#         path = osp.join(osp.dirname(osp.realpath(__file__)), 'Y_distributions')
#         os.makedirs(path, exist_ok=True)
#         plt.savefig(f'{path}/{name}.pdf', dpi=300, bbox_inches='tight')
#         plt.show()
#         plt.clf()

def plot_histogram(decay_rates: List[float], filename: str) -> None:
    """
    Plot and save a histogram of the decay rates.

    Parameters:
    decay_rates (List[float]): List of decay slopes.
    filename (str): Filename to save the plot.
    """
    plt.figure()
    plt.hist(decay_rates, bins=20, edgecolor='black')
    plt.title('Distribution of Decay Slopes')
    plt.xlabel('Slope (-Decay Rate)')
    plt.ylabel('Frequency')
    plt.savefig(filename)
    print(f"Histogram saved as {filename}")

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

    # if graph is not None:
    #     edge_index = graph.edge_index
    #     if not directed:
    #         edge_index = to_undirected(edge_index)
    #     existing_edges = {tuple(e) for e in edge_index.t().tolist()}

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
    # return edge_index, edge_type, graph.x, graph.y, n # remember to fix
    x = graph.x if graph is not None else None
    y = graph.y if graph is not None else None
    return edge_index, edge_type, x, y, n

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



def _make_dummy_graph():
    # 4 nodes, undirected triangle (0,1,2) plus isolated 3
    edge_index = torch.tensor([[0,1,1,2,2,0],
                               [1,0,2,1,0,2]], dtype=torch.long)
    x = torch.randn(4, 3)
    y = torch.tensor([0])
    return Data(x=x, edge_index=edge_index, y=y)

# -------------------- TESTS ----------------------------
def test_determinism():
    e1, t1, *_ = gnm_random_graph_v2(m=5, seed=42, directed=True, n=10)
    e2, t2, *_ = gnm_random_graph_v2(m=5, seed=42, directed=True, n=10)
    assert torch.equal(e1, e2) and torch.equal(t1, t2), "Same seed → different graphs!"

def test_variability():
    e1, *_ = gnm_random_graph_v2(m=5, seed=1, directed=True, n=10)
    e2, *_ = gnm_random_graph_v2(m=5, seed=2, directed=True, n=10)
    assert not torch.equal(e1, e2), "Different seeds → identical graphs!"

def test_graph_without_prior_edges():
    m, n = 6, 8
    ei, et, *_ = gnm_random_graph_v2(m=m, seed=0, directed=False, n=n)
    pair_cnt = ei.size(1) // 2   # each undirected edge appears twice
    assert pair_cnt == m, f"Expected {m} undirected edges, got {pair_cnt}"
    assert ei.max() < n, "Node index ≥ n"
    # uniqueness check (unordered pairs)
    pairs = {tuple(sorted(ei[:, i].tolist())) for i in range(0, ei.size(1), 2)}
    assert len(pairs) == m, "Duplicate edges generated"
    # self-loop check
    assert not any(ei[0] == ei[1]), "Self-loop present"

def test_respects_existing_edges():
    g = _make_dummy_graph()
    existing = {tuple(g.edge_index[:, i].tolist()) for i in range(g.edge_index.size(1))}
    m = 4
    ei, et, *_ = gnm_random_graph_v2(m=m, seed=0, directed=False, graph=g)
    new_flags = (et == 1).nonzero().flatten()
    assert len(new_flags) == m * 2, \
        "edge_type should flag exactly the new undirected edges (both directions)"
    # none of the "new" edges should be already in existing
    for idx in new_flags:
        assert tuple(ei[:, idx].tolist()) not in existing, "Duplicate added into graph"

def test_overflow_m_is_capped():
    n = 5
    max_undirected = n * (n - 1) // 2
    # we already ask for 3× more than possible
    ei, _, *_ = gnm_random_graph_v2(m=3 * max_undirected, seed=0, directed=False, n=n)
    assert ei.size(1) // 2 == max_undirected, "Edge count exceeds theoretical maximum"

def test_assert_raised_without_n_or_graph():
    try:
        gnm_random_graph_v2(m=3)     # missing n **and** graph
    except AssertionError:
        pass
    else:
        assert False, "Expected AssertionError when n and graph are both None"

# -------------------- MAIN -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)  # stability for dummy graph generation
    tests = [obj for name, obj in globals().items() if name.startswith("test_")]
    for test in tests:
        test()            # run
        print(f"{test.__name__}: ✅")

import os
import os.path as osp
from typing import List, Union, Tuple, Any
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch_geometric.datasets import TUDataset, LRGBDataset
from tqdm import tqdm
from utils import get_adj
from scipy.stats import linregress
from args import get_args
from toy_datasets import generate_ring_transfer_graph, visualize_graph
from toy_datasets import generate_tree_transfer_graph
from toy_datasets import generate_lollipop_transfer_graph
from utils import compute_diameter
from torch_geometric.utils import add_self_loops
import torch_geometric.transforms as T
from statsmodels.stats.multitest import multipletests

matplotlib.use('Agg')  # Use the Agg backend for plotting


class Custom_transform(object):
    def __call__(self, data):
        data.x = torch.ones(data.num_nodes, 1)
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
        return TUDataset(path, name=args.dataset, transform=T.Compose([Custom_transform(), AddSelfLoopsTransform()]))
    elif args.dataset in ['Peptides-func', 'PCQM-Contact', 'Peptides-struct']:
        return LRGBDataset(path, name=args.dataset, transform=AddSelfLoopsTransform())
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")


def ball_of_radius(adj: torch.Tensor, v: int, radius: int, device: str) -> float:
    """
    Computes the volume of the ball of a given radius centered at node v.

    Parameters:
    adj (torch.Tensor): Adjacency matrix (NxN).
    v (int): Center node.
    radius (int): The radius of the ball to compute.
    device (str): Device to run the computation on ('cpu' or 'cuda').

    Returns:
    float: The volume of the ball, defined as the sum of degrees of all nodes in the ball.
    """
    adj = adj.to(device)
    degrees = adj.sum(dim=1)  # Degree of each node
    num_nodes = adj.shape[0]
    # total_volume = 0.0
    # last_volume = 0.0

    if radius == 0:
        # If the radius is 0, the ball contains only the node itself
        return degrees[v].item()

    # Distance tensor initialized to infinity
    distances = torch.full((num_nodes,), float('inf'), device=device)
    distances[v] = 0  # Distance to itself is 0

    # Start BFS from node v
    frontier = torch.tensor([v], device=device)  # Start with node v
    # total_volume += degrees[v].item()

    for dist in range(1, radius + 1):
        if frontier.numel() == 0:
            # total_volume += last_volume
            # continue
            break
        # Get all neighbors of nodes in the current frontier
        neighbors = adj[frontier].nonzero(as_tuple=True)[1].unique()

        # Update distances of nodes not yet visited (distance = inf)
        unvisited = distances[neighbors] == float('inf')
        distances[neighbors[unvisited]] = dist

        # Set up the next frontier
        frontier = neighbors[unvisited]

        # Nodes within the radius
    ball = (distances <= radius).nonzero(as_tuple=True)[0]
        # last_volume = degrees[ball].sum().item()

        # total_volume += degrees[ball].sum().item()

    # Return the volume (sum of degrees in the ball)
    return degrees[ball].sum().item()


def decay_rate(
    adj: torch.Tensor,
    diameter: int,
    alpha: float = 0.05,
    device: str = 'cpu',
    calculate_metrics: bool = False,
) -> Tuple[List[float], float, float, float]:
    """
    Compute the decay rates for all pairs of nodes in an adjacency matrix by fitting a linear model
    on the log of the matrix powers. Returns the decay rates along with the mean corrected p-value,
    mean R^2 value, and mean uncorrected p-value.

    Parameters:
        adj (torch.Tensor): Adjacency matrix.
        diameter (int): Used to define the range [diameter, 2*diameter] for computing matrix powers.
        alpha (float): Significance level for multiple test correction.
        device (str): Device for computation.

    Returns:
        Tuple[List[float], float, float, float]:
            - List of decay rates.
            - Mean of corrected p-values.
            - Mean of R^2 values.
            - Mean of uncorrected p-values.
    """
    adj = adj.to(device)

    A_powers = torch.stack([torch.matrix_power(adj, k) for k in range(diameter, 2 * diameter)], dim=2)
    log_A_powers = torch.log(A_powers)

    k_values = torch.arange(diameter, 2 * diameter, device=device, dtype=torch.float32)

    slopes, r2_values, p_values = [], [], []

    v_indices, u_indices = torch.triu_indices(adj.shape[0], adj.shape[1], offset=1) # 0 or 1? 1 means no diagonal
    sum_M_ku_l = torch.sum(A_powers, dim=0)

    for v, u in zip(v_indices.cpu().numpy(), u_indices.cpu().numpy()):
            log_A_vu_k = log_A_powers[v, u, :] - torch.log(sum_M_ku_l[u, :])

            slope, _, r_value, p_value, _ = linregress(k_values.cpu().numpy(), log_A_vu_k.cpu().numpy())
            slopes.append(slope)
            r2_values.append(r_value ** 2)
            p_values.append(p_value)

    slopes_matrix = torch.full((adj.shape[0], adj.shape[1]), torch.nan, device=device)
    slopes_matrix[v_indices, u_indices] = torch.tensor(slopes, device=device, dtype=torch.float32)
    slopes_matrix[u_indices, v_indices] = torch.tensor(slopes, device=device, dtype=torch.float32)

    slopes_matrix = torch.round(slopes_matrix, decimals=2)
    # slopes_matrix = -1 * slopes_matrix  # Convert slopes to decay rates

    decay_rates = slopes_matrix[~torch.isnan(slopes_matrix)].tolist()
    decay_rates_with_nan = slopes_matrix.flatten().tolist()

    if calculate_metrics:
        _, corrected_p_values, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
        mean_corrected_p = torch.mean(torch.tensor(corrected_p_values)).item()
        mean_r2 = torch.mean(torch.tensor(r2_values)).item()
        mean_p = torch.mean(torch.tensor(p_values)).item()

        return decay_rates, mean_corrected_p, mean_r2, mean_p
    else:
        return decay_rates, decay_rates_with_nan

def plot_histogram(slopes: List[float], filename: str) -> None:
    """
    Plot and save a histogram of the slopes.

    Parameters:
    slopes (List[float]): List of decay slopes.
    filename (str): Filename to save the plot.
    """
    plt.figure()
    plt.hist(slopes, bins=20, edgecolor='black')
    plt.title('Distribution of Decay Slopes')
    plt.xlabel('Slope (-Decay Rate)')
    plt.ylabel('Frequency')
    plt.savefig(filename)
    print(f"Histogram saved as {filename}")


# def main() -> None:
#     args = get_args()
#     try:
#         dataset = get_dataset(args)
#     except ValueError as e:
#         print(e)
#         return
#
#     all_slopes = []
#
#     print(f"Processing dataset with {len(dataset)} graphs")
#     for data in tqdm(dataset, desc='Processing', total=len(dataset)):
#         adj = get_adj(data.edge_index, set_diag=False, symmetric_normalize=False)
#         slopes = slope(adj, 5)
#         all_slopes.extend(slopes)
#
#     histograms_dir = osp.join(osp.dirname(osp.realpath(__file__)), 'histograms')
#     if not osp.exists(histograms_dir):
#         os.makedirs(histograms_dir)
#
#     path = osp.join(histograms_dir, f'{args.dataset}_histogram.png')
#     plot_histogram(all_slopes, path)

import numpy as np
import scipy.sparse as sp

def is_connected(data):
    n = data.num_nodes
    # Convert edge_index to numpy arrays
    row, col = data.edge_index.cpu().numpy()
    # Build a sparse COO matrix; assume unweighted edges
    A = sp.coo_matrix((np.ones_like(row), (row, col)), shape=(n, n))
    # Ensure symmetry (i.e., treat the graph as undirected)
    A = A + A.T
    # Compute connected components
    n_components, _ = sp.csgraph.connected_components(A)
    return n_components


def main():
    # datasets = [
    #     ("ring", generate_ring_transfer_graph(10, [0, 1], False, True)),
    #     # ("cross ring", generate_ring_transfer_graph(10, [0, 1], True)),
    #     # ("lolipop", generate_lollipop_transfer_graph(8, [0, 1]))
    #     # ("Tree", generate_tree_transfer_graph(2, [0, 1], 2))
    # ]

    real_datasets = [
        # ("proteins", list(TUDataset(root="data", name="PROTEINS", transform=AddSelfLoopsTransform()))),
        # ("enzymes", list(TUDataset(root="data", name="ENZYMES", transform=AddSelfLoopsTransform()))),
        ("imdb_binary", list(TUDataset(root="data", name="IMDB-BINARY", transform=AddSelfLoopsTransform()))),
        # ("reddit", list(TUDataset(root="data", name="REDDIT-BINARY", transform=AddSelfLoopsTransform()))),
        # ("peptides_func", list(LRGBDataset(root="data", name="Peptides-func", transform=AddSelfLoopsTransform()))),
        # ("peptides_struct", list(LRGBDataset(root="data", name="Peptides-struct", transform=AddSelfLoopsTransform())))
    ]
    for name, dataset in real_datasets:
        for idx, data in enumerate(dataset):
            if is_connected(data) != 1:
                print(f"{name} graph {idx} is not connected, skipped.")
                continue

            visualize_graph(data, data.num_nodes, name, "test")
            adj = get_adj(data.edge_index, set_diag=False, symmetric_normalize=False)

            original_diameter = compute_diameter(data)
            decay_rates = decay_rate(adj, diameter=original_diameter)
            # print(f'{name} slopes:\n {decay_rates}')
            avg_decay_rates = torch.mean(torch.tensor(decay_rates)).item()
            print(f'{name} avg_decay_rates: {avg_decay_rates}')
            # print(f'{name} mean_corrected_p_values: {mean_corrected_p_values}')
            # print(f'{name} mean_r2: {mean_r2}')
            # print(f'{name} mean_p_values: {mean_p_values}')



if __name__ == "__main__":
    main()

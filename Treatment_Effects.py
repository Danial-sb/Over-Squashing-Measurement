from copy import deepcopy
from typing import List
from typing import Tuple
from typing import Dict
from typing import Any
from scipy import stats
from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor, Planetoid
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import Compose, LargestConnectedComponents
import torch
from measurement import is_connected
import math
from scipy.stats import ttest_rel, chi2
from toy_datasets import generate_ring_transfer_graph, generate_lollipop_transfer_graph, visualize_graph
from tqdm import tqdm
import logging
from measurement import get_dataset
from toy_datasets import generate_tree_transfer_graph
import os
import random
from args import get_args
from torch_geometric.datasets import TUDataset, LRGBDataset
from Rewire import AddSelfLoopsTransform, CustomTransform
import os.path as osp
from utils import get_adj
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.utils import degree
from measurement import decay_rate
from utils import calculate_num_random_edges, gnm_random_graph_v2, compute_diameter
from utils import log_message, setup_logging
from preprocessing import digl, sdrf, fosr, borf


# def gnm_random_graph(n, m, seed=None, directed=False):
#     """
#     Returns a G(n,m) random graph with O(m) time complexity.
#     """
#
#     # Set random seed if provided
#     if seed is not None:
#         random.seed(seed)
#         torch.manual_seed(seed)
#
#     # Initialize an empty set for edges
#     edge_set = set()
#
#     # Maximum number of possible edges (account for undirected graph if needed)
#     max_edges = n * (n - 1) if directed else n * (n - 1) // 2
#
#     if m >= max_edges:
#         # If the number of edges exceeds the possible max edges, return a complete graph
#         if directed:
#             edge_list = [(i, j) for i in range(n) for j in range(n) if i != j]
#         else:
#             edge_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
#     else:
#         # Add nodes
#         nlist = list(range(n))
#
#         while len(edge_set) < m:
#             # Generate random edge u, v
#             u, v = random.sample(nlist, 2)  # Randomly select two distinct nodes without replacement
#             # Add edge to set, respecting direction if required
#             edge_set.add((u, v) if directed else (min(u, v), max(u, v)))
#
#         edge_list = list(edge_set)
#
#     # Convert edge list to tensor of shape (2, num_edges)
#     edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
#
#     # Return as PyG Data object
#     return Data(edge_index=edge_index, num_nodes=n)


# def test_and_plot_toy_datasets(seed):
#     datasets = [
#         ("Ring", generate_ring_transfer_graph(10, [0, 1], False, True)),
#         ("CrossedRing", generate_ring_transfer_graph(10, [0, 1], True, True)),
#         ("Lolipop", generate_lollipop_transfer_graph(8, [0, 1], True)),
#         ("Tree", generate_tree_transfer_graph(2, [0, 1], 2))
#     ]
#     # power = 5
#     # alpha = 0.05
#
#     histograms_dir = osp.join(osp.dirname(osp.realpath(__file__)), 'histograms')
#     if not osp.exists(histograms_dir):
#         os.makedirs(histograms_dir)
#
#     for name, dataset in datasets:
#         # visualize_graph(dataset, dataset.num_nodes, f'{name}_new')
#         # print(f'Dirichlet Energy of Original {name}: {dirichlet_normalized(dataset.x, dataset.edge_index)}')
#         adj = get_adj(dataset.edge_index, set_diag=False, symmetric_normalize=False)
#         original_diameter = compute_diameter(dataset)
#         print(f'{name} diameter [Original]: {original_diameter}')
#
#         decay_rates_original, mean_corrected, mean_r2, mean_p_values = decay_rate(adj, diameter=original_diameter)
#         print(f'{name} decay_rates [Original]:\n {decay_rates_original}')
#
#         # if len(decay_rates_original) > 7:  # Normality test needs at least 8 samples
#         #     stat, p_original = stats.normaltest(decay_rates_original)
#         #     is_original_normal = p_original > alpha
#         #     print(f'{name} decay_rates [Original] normal test p-value: {p_original}')
#         #     print("The decay rates follow a normal distribution." if is_original_normal else
#         #           "The decay rates do not follow a normal distribution.")
#         # else:
#         #     print(f"Not enough valid data for normality test on decay rates [Original]. Only"
#         #           f"{len(decay_rates_original)} valid values.")
#
#         # path = osp.join(histograms_dir, f'{name}_QQ.png')
#         # generate_qq_plots(decay_rates_original, path)
#         # plot_histogram(slopes, path)
#
#         print("=" * 20)
#         k = calculate_num_random_edges(dataset, original_diameter, dataset.num_nodes)
#         wired_dataset = gnm_random_graph_v2(k, seed=seed, directed=False, graph=dataset)
#         # visualize_graph(wired_dataset, dataset.num_nodes, f'{name}_wired_new')
#
#         # print(f'Dirichlet Energy of Wired {name}: {dirichlet_normalized(wired_dataset.x, wired_dataset.edge_index)}')
#         adj = get_adj(wired_dataset.edge_index, set_diag=False, symmetric_normalize=False)
#         wired_diameter = compute_diameter(wired_dataset)
#         print(f'{name} diameter [Wired]: {wired_diameter}')
#         decay_rates_wired, decay_rates_wired_with_nan, A_powers_wired = decay_rate(adj, diameter=original_diameter)
#         print(f'{name} decay_rates [Wired]:\n {decay_rates_wired}')
#
#         # if len(decay_rates_wired) > 7:
#         #     stat, p_wired = stats.normaltest(decay_rates_wired)
#         #     is_wired_normal = p_wired > alpha
#         #     print(f'{name} decay_rates [Wired] normal test p-value: {p_wired}')
#         #     print("The decay rates follow a normal distribution." if is_wired_normal else
#         #           "The decay rates do not follow a normal distribution.")
#         # else:
#         #     print(f"Not enough valid data for normality test on decay rates [Wired]. Only"
#         #           f" {len(decay_rates_wired)} valid values.")
#
#         # path = osp.join(histograms_dir, f'{name}_Wired_QQ.png')
#         # generate_qq_plots(decay_rates_wired, path)
#         # path = osp.join(histograms_dir, f'{name}_wired_histogram_k={power}.png')
#         # plot_histogram(slopes, path)
#
#         print("=" * 20)
#         # if len(decay_rates_original) > 0 and len(decay_rates_wired) > 0:
#         #     # Choose statistical test based on normality of both distributions
#         #     if is_original_normal and is_wired_normal:
#         #         print("Both distributions are normal. Performing paired t-test.")
#         #         stat, p = stats.ttest_rel(decay_rates_original_with_nan, decay_rates_wired_with_nan, nan_policy='omit')
#         #     else:
#         #         print("Distributions are not normal. Performing Wilcoxon signed-rank test.")
#         #         stat, p = stats.wilcoxon(decay_rates_original_with_nan, decay_rates_wired_with_nan, nan_policy='omit')
#         #
#         #     print(f'Test statistic: {stat}, p-value: {p:.5f}')
#         #     if p > alpha:
#         #         print("Fail to reject H0: No significant difference between the two decay rates.")
#         #     else:
#         #         print("Reject H0: There is a significant difference between the two decay rates.")
#         # else:
#         #     print(f"Not enough valid decay rates for comparison. Original: {len(decay_rates_original)}, Wired:"
#         #           f"{len(decay_rates_wired)}")
#
#         # Convert lists to torch tensors
#         decay_rates_original = torch.tensor(decay_rates_original_with_nan)
#         decay_rates_wired = torch.tensor(decay_rates_wired_with_nan)
#         #
#         # Filter out pairs where either original or wired values are NaN
#         mask = ~torch.isnan(decay_rates_original) & ~torch.isnan(decay_rates_wired)
#         filtered_original = decay_rates_original[mask]
#         filtered_wired = decay_rates_wired[mask]
#         #
#         if len(filtered_original) < 5 or len(filtered_wired) < 5:
#             print("Too few valid data points after filtering NaNs for further analysis.")
#             continue
#
#         # # Number of values to consider (e.g., bottom 10%)
#         # num_extreme_values = int(0.1 * len(filtered_original))
#         #
#         # # Find the indices of the most negative decay rates in the original distribution
#         # sorted_indices = torch.argsort(filtered_original)
#         # most_negative_indices = sorted_indices[:num_extreme_values]
#         #
#         # # Extract the corresponding values from both filtered distributions
#         # most_negative_original = filtered_original[most_negative_indices]
#         # most_negative_wired = filtered_wired[most_negative_indices]
#
#         negative_mask = filtered_original < 0.0
#         most_negative_original = filtered_original[negative_mask]
#         most_negative_wired = filtered_wired[negative_mask]
#
#         # Calculate how these values have changed
#         changes_in_negative = most_negative_wired - most_negative_original
#
#         changes_in_total = filtered_wired - filtered_original
#
#         num_positive_changes = torch.sum(changes_in_negative > 0).item()
#         num_negative_changes = torch.sum(changes_in_negative < 0).item()
#         num_zero_changes = torch.sum(changes_in_negative == 0).item()
#         total_changes = len(changes_in_negative)
#
#         num_positive_changes_total = torch.sum(changes_in_total > 0).item()
#         num_negative_changes_total = torch.sum(changes_in_total < 0).item()
#         num_zero_changes_total = torch.sum(changes_in_total == 0).item()
#         total_changes_total = len(changes_in_total)
#
#         positive_percentage = (num_positive_changes / total_changes) * 100
#         negative_percentage = (num_negative_changes / total_changes) * 100
#         zero_percentage = (num_zero_changes / total_changes) * 100
#
#         positive_percentage_total = (num_positive_changes_total / total_changes_total) * 100
#         negative_percentage_total = (num_negative_changes_total / total_changes_total) * 100
#         zero_percentage_total = (num_zero_changes_total / total_changes_total) * 100
#
#         print("=" * 20)
#         print(f'Percentage of positive shifts: {positive_percentage:.2f}%')
#         print(f'Percentage of negative shifts: {negative_percentage:.2f}%')
#         print(f'Percentage of no change: {zero_percentage:.2f}%')
#
#         print(f'Percentage of positive shifts total: {positive_percentage_total:.2f}%')
#         print(f'Percentage of negative shifts total: {negative_percentage_total:.2f}%')
#         print(f'Percentage of no change total: {zero_percentage_total:.2f}%')
#
#         # Calculate the maximum and average positive and negative shifts
#         if num_positive_changes > 0:
#             max_positive_shift = torch.max(changes_in_negative[changes_in_negative > 0]).item()
#             avg_positive_shift = torch.mean(changes_in_negative[changes_in_negative > 0]).item()
#             print(f'Maximum positive shift: {max_positive_shift}')
#             print(f'Average positive shift: {avg_positive_shift}')
#         else:
#             print("No positive shifts found.")
#
#         if num_negative_changes > 0:
#             max_negative_shift = torch.min(changes_in_negative[changes_in_negative < 0]).item()
#             avg_negative_shift = torch.mean(changes_in_negative[changes_in_negative < 0]).item()
#             print(f'Maximum negative shift: {max_negative_shift}')
#             print(f'Average negative shift: {avg_negative_shift}')
#         else:
#             print("No negative shifts found.")
#
#         if num_positive_changes_total > 0:
#             max_positive_shift_total = torch.max(changes_in_total[changes_in_total > 0]).item()
#             avg_positive_shift_total = torch.mean(changes_in_total[changes_in_total > 0]).item()
#             print(f'Maximum positive shift total: {max_positive_shift_total}')
#             print(f'Average positive shift total: {avg_positive_shift_total}')
#         else:
#             print("No positive shifts found.")
#
#         if num_negative_changes_total > 0:
#             max_negative_shift_total = torch.min(changes_in_total[changes_in_total < 0]).item()
#             avg_negative_shift_total = torch.mean(changes_in_total[changes_in_total < 0]).item()
#             print(f'Maximum negative shift total: {max_negative_shift_total}')
#             print(f'Average negative shift total: {avg_negative_shift_total}')
#         else:
#             print("No negative shifts found.")
#
#         print("=" * 40)


# def test_and_plot_real_datasets(seed):
#     args = get_args()
#     try:
#         dataset = get_dataset(args)
#     except ValueError as e:
#         print(e)
#         return
#     # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     device = torch.device('cpu')
#
#     rewired_dataset = Rewire(root=osp.join(osp.dirname(osp.realpath(__file__)), 'Rewired_dataset'),
#                              name=args.dataset, dataset=dataset, seed=seed)
#
#     all_decays_original = []
#     all_decays_original_nan = []
#
#     print(f"Processing dataset with {len(dataset)} graphs")
#     for data in tqdm(dataset, desc='Processing', total=len(dataset)):
#         adj = get_adj(data.edge_index, set_diag=False, symmetric_normalize=False, device=device)
#         original_diameter = compute_diameter(original_graph=data, rewired_graph=None)
#         decay_rates_original, decay_rates_original_nan = slope(adj, diameter=original_diameter, device=device)
#         all_decays_original.extend(decay_rates_original)
#         all_decays_original_nan.extend(decay_rates_original_nan)
#
#     histograms_dir = osp.join(osp.dirname(osp.realpath(__file__)), 'histograms')
#     if not osp.exists(histograms_dir):
#         os.makedirs(histograms_dir)
#
#     # path = osp.join(histograms_dir, f'{args.dataset}_histogram_k={power}.png')
#     # plot_histogram(all_decays_original_nan, path)
#
#     all_decays_wired = []
#     all_decays_wired_nan = []
#
#     print(f"Processing rewired dataset with {len(rewired_dataset)} graphs")
#     for rewired_data in tqdm(rewired_dataset, desc='Processing', total=len(rewired_dataset)):
#         adj = get_adj(rewired_data.edge_index, set_diag=False, symmetric_normalize=False, device=device)
#         rewired_diameter = compute_diameter(original_graph=None, rewired_graph=rewired_data)
#         decay_rates_wired, decay_rates_wired_nan = slope(adj, diameter=rewired_diameter, device=device)
#         all_decays_wired.extend(decay_rates_wired)
#         all_decays_wired_nan.extend(decay_rates_wired_nan)
#
#     # check normality of the original decay rates
#     # stat, p_value_original = stats.normaltest(all_decays_original)
#     # if p_value_original > 0.05:
#     #     print("The decay rates of the original dataset follow a normal distribution.")
#     # else:
#     #     print("The decay rates of the original dataset do not follow a normal distribution.")
#
#     # check normality of the wired decay rates
#     # stat, p_value_wired = stats.normaltest(all_decays_wired)
#     # if p_value_wired > 0.05:
#     #     print("The decay rates of the wired dataset follow a normal distribution.")
#     # else:
#     #     print("The decay rates of the wired do not follow a normal distribution.")
#
#     # if they were not normal, use Wilcoxon signed-rank test
#     # if p_value_original < 0.05 or p_value_wired < 0.05:
#     #     stat, p = stats.wilcoxon(all_decays_original, all_decays_wired)
#     #     print(f'Wilcoxon test statistic: {stat}, p-value: {p:.5f}')
#     #     if p > 0.05:
#     #         print("Fail to reject H0: No significant difference between the two decay rates.")
#     #     else:
#     #         print("Reject H0: There is a significant difference between the two decay rates.")
#     # else:
#     #     # if they were normal, use paired t-test
#     #     stat, p = stats.ttest_rel(all_decays_original, all_decays_wired)
#     #     print(f'T-test statistic: {stat}, p-value: {p:.5f}')
#     #     if p > 0.05:
#     #         print("Fail to reject H0: No significant difference between the two decay rates.")
#     #     else:
#     #         print("Reject H0: There is a significant difference between the two decay rates.")
#
#     decay_rates_original = torch.tensor(all_decays_original_nan) #Todo: Do for each graph seperatly.
#     decay_rates_wired = torch.tensor(all_decays_wired_nan)
#
#     # Filter out pairs where either original or wired values are NaN
#     mask = ~torch.isnan(decay_rates_original) & ~torch.isnan(decay_rates_wired)
#     filtered_original = decay_rates_original[mask]
#     filtered_wired = decay_rates_wired[mask]
#
#     # Number of values to consider (e.g., bottom 10%)
#     num_extreme_values = int(0.1 * len(filtered_original))
#
#     # Find the indices of the most negative decay rates in the original distribution
#     sorted_indices = torch.argsort(filtered_original)
#     most_negative_indices = sorted_indices[:num_extreme_values]
#
#     # Extract the corresponding values from both filtered distributions
#     most_negative_original = filtered_original[most_negative_indices]
#     most_negative_wired = filtered_wired[most_negative_indices]
#
#     # Calculate how these values have changed
#     changes_in_negative = most_negative_wired - most_negative_original
#
#     num_positive_changes = torch.sum(changes_in_negative > 0).item()
#     num_negative_changes = torch.sum(changes_in_negative < 0).item()
#     num_zero_changes = torch.sum(changes_in_negative == 0).item()
#     total_changes = len(changes_in_negative)
#
#     positive_percentage = (num_positive_changes / total_changes) * 100
#     negative_percentage = (num_negative_changes / total_changes) * 100
#     zero_percentage = (num_zero_changes / total_changes) * 100
#
#     print(f'Percentage of positive shifts: {positive_percentage:.2f}%')
#     print(f'Percentage of negative shifts: {negative_percentage:.2f}%')
#     print(f'Percentage of no change: {zero_percentage:.2f}%')
#
#     # Calculate the maximum and average positive and negative shifts
#     if num_positive_changes > 0:
#         max_positive_shift = torch.max(changes_in_negative[changes_in_negative > 0]).item()
#         avg_positive_shift = torch.mean(changes_in_negative[changes_in_negative > 0]).item()
#         print(f'Maximum positive shift: {max_positive_shift}')
#         print(f'Average positive shift: {avg_positive_shift}')
#     else:
#         print("No positive shifts found.")
#
#     if num_negative_changes > 0:
#         max_negative_shift = torch.min(changes_in_negative[changes_in_negative < 0]).item()
#         avg_negative_shift = torch.mean(changes_in_negative[changes_in_negative < 0]).item()
#         print(f'Maximum negative shift: {max_negative_shift}')
#         print(f'Average negative shift: {avg_negative_shift}')
#     else:
#         print("No negative shifts found.")
#
#     # path = osp.join(histograms_dir, f'{args.dataset}_rewired_histogram_k={power}.png')
#     # plot_histogram(all_decays_wired, path)

# def test_and_plot_real_datasets():
#     args = get_args()
#     try:
#         dataset = get_dataset(args)
#     except ValueError as e:
#         log_message(f"Error while loading dataset: {e}")
#         return
#
#     device = torch.device('cpu')
#
#     try:
#         root_dir = osp.join(osp.dirname(osp.realpath(__file__)), 'Rewired_dataset')
#     except NameError:
#         log_message("Error: '__file__' is not defined in this environment.")
#         return
#
#     rewired_dataset = Rewire(root=root_dir, name=args.dataset, dataset=dataset, args=args)
#
#     # Lists to store the positive shift values for all graphs
#     max_positive_shifts = []
#     avg_positive_shifts = []
#     max_negative_shifts = []
#     avg_negative_shifts = []
#     max_positive_shifts_total = []
#     avg_positive_shifts_total = []
#     max_negative_shifts_total = []
#     avg_negative_shifts_total = []
#
#     log_message(f"Processing dataset with {len(dataset)} graphs")
#     for i, (data, wired_data) in enumerate(zip(dataset, rewired_dataset)):
#         # print(f"Processing graph {i + 1}/{len(dataset)}")
#         # if i >= 500:
#         #     log_message("Processed 500 graphs. Stopping.")
#         #     break
#         # visualize_graph(data, data.num_nodes, f'Original_COLLAB_{i + 1}')
#
#         try:
#             # dirichlet_energy_original = dirichlet_normalized(data.x, data.edge_index)
#             # dirichlet_energies_original.append(dirichlet_energy_original)
#             adj = get_adj(data.edge_index, set_diag=False, symmetric_normalize=False, device=device)
#             original_diameter = compute_diameter(original_graph=data, rewired_graph=None)
#             decay_rates_original, decay_rates_original_nan, A_power = decay_rate(adj, diameter=original_diameter,
#                                                                                  device=device)
#
#             # dirichlet_energy_wired = dirichlet_normalized(wired_data.x, wired_data.edge_index)
#             # dirichlet_energies_wired.append(dirichlet_energy_wired)
#             adj = get_adj(wired_data.edge_index, set_diag=False, symmetric_normalize=False, device=device)
#             rewired_diameter = compute_diameter(original_graph=None, rewired_graph=wired_data)
#             decay_rates_wired, decay_rates_wired_nan, A_powers_wired = decay_rate(adj, diameter=rewired_diameter,
#                                                                                   device=device)
#
#             decay_rates_original_nan = torch.tensor(decay_rates_original_nan)
#             decay_rates_wired_nan = torch.tensor(decay_rates_wired_nan)
#
#             mask = ~torch.isnan(decay_rates_original_nan) & ~torch.isnan(decay_rates_wired_nan)
#             filtered_original = decay_rates_original_nan[mask]
#             filtered_wired = decay_rates_wired_nan[mask]
#
#             if len(filtered_original) == 0:
#                 log_message(f"Graph {i + 1} has no valid decay rates after filtering.")
#                 log_message("=" * 40)
#                 continue
#
#             # if len(filtered_original) <= 10:
#             #     num_extreme_values = len(filtered_original)
#             # else:
#             #     num_extreme_values = int(0.1 * len(filtered_original))
#             #
#             # # Find the indices of the most negative decay rates in the original distribution
#             # sorted_indices = torch.argsort(filtered_original)
#             # most_negative_indices = sorted_indices[filtered_original[sorted_indices] < 0][:num_extreme_values]
#             #
#             # # Extract the corresponding values from both filtered distributions
#             # most_negative_original = filtered_original[most_negative_indices]
#             # most_negative_wired = filtered_wired[most_negative_indices]
#
#             negative_mask = filtered_original < 0.0
#             most_negative_original = filtered_original[negative_mask]
#             most_negative_wired = filtered_wired[negative_mask]
#
#             # Calculate how these values have changed
#             changes_in_negative = most_negative_wired - most_negative_original
#             changes_in_total = filtered_wired - filtered_original
#
#             num_positive_changes = torch.sum(changes_in_negative > 0).item()
#             num_negative_changes = torch.sum(changes_in_negative < 0).item()
#             num_zero_changes = torch.sum(changes_in_negative == 0).item()
#             total_changes = len(changes_in_negative)
#
#             num_positive_changes_total = torch.sum(changes_in_total > 0).item()
#             num_negative_changes_total = torch.sum(changes_in_total < 0).item()
#             num_zero_changes_total = torch.sum(changes_in_total == 0).item()
#             total_changes_total = len(changes_in_total)
#
#             if total_changes > 0:
#                 positive_percentage = (num_positive_changes / total_changes) * 100
#                 negative_percentage = (num_negative_changes / total_changes) * 100
#                 zero_percentage = (num_zero_changes / total_changes) * 100
#                 log_message(f'Graph {i + 1}:')
#                 log_message(f'Percentage of positive shifts: {positive_percentage:.2f}%')
#                 log_message(f'Percentage of negative shifts: {negative_percentage:.2f}%')
#                 log_message(f'Percentage of no change: {zero_percentage:.2f}%')
#             else:
#                 log_message(f"Graph {i + 1}: No valid decay rate changes found.")
#
#             if total_changes_total > 0:
#                 positive_percentage_total = (num_positive_changes_total / total_changes_total) * 100
#                 negative_percentage_total = (num_negative_changes_total / total_changes_total) * 100
#                 zero_percentage_total = (num_zero_changes_total / total_changes_total) * 100
#                 # log_message(f'Graph {i + 1}:')
#                 log_message(f'Percentage of positive shifts total: {positive_percentage_total:.2f}%')
#                 log_message(f'Percentage of negative shifts total: {negative_percentage_total:.2f}%')
#                 log_message(f'Percentage of no change total: {zero_percentage_total:.2f}%')
#             else:
#                 log_message(f"Graph {i + 1}: No valid decay rate changes found.")
#
#             # Calculate the maximum and average positive and negative shifts
#             if num_positive_changes > 0:
#                 max_positive_shift = torch.max(changes_in_negative[changes_in_negative > 0]).item()
#                 avg_positive_shift = torch.mean(changes_in_negative[changes_in_negative > 0]).item()
#
#                 # Store the values for calculating overall averages
#                 max_positive_shifts.append(max_positive_shift)
#                 avg_positive_shifts.append(avg_positive_shift)
#
#                 log_message(f'Maximum positive shift: {max_positive_shift}')
#                 log_message(f'Average positive shift: {avg_positive_shift}')
#             else:
#                 log_message("No positive shifts found.")
#
#             if num_negative_changes > 0:
#                 max_negative_shift = torch.min(changes_in_negative[changes_in_negative < 0]).item()
#                 avg_negative_shift = torch.mean(changes_in_negative[changes_in_negative < 0]).item()
#
#                 # Store the values for calculating overall averages
#                 max_negative_shifts.append(max_negative_shift)
#                 avg_negative_shifts.append(avg_negative_shift)
#
#                 log_message(f'Maximum negative shift: {max_negative_shift}')
#                 log_message(f'Average negative shift: {avg_negative_shift}')
#             else:
#                 log_message("No negative shifts found.")
#
#             if num_positive_changes_total > 0:
#                 max_positive_shift_total = torch.max(changes_in_total[changes_in_total > 0]).item()
#                 avg_positive_shift_total = torch.mean(changes_in_total[changes_in_total > 0]).item()
#
#                 # Store the values for calculating overall averages
#                 max_positive_shifts_total.append(max_positive_shift_total)
#                 avg_positive_shifts_total.append(avg_positive_shift_total)
#
#                 log_message(f'Maximum positive shift total: {max_positive_shift_total}')
#                 log_message(f'Average positive shift total: {avg_positive_shift_total}')
#             else:
#                 log_message("No positive shifts found.")
#
#             if num_negative_changes_total > 0:
#                 max_negative_shift_total = torch.min(changes_in_total[changes_in_total < 0]).item()
#                 avg_negative_shift_total = torch.mean(changes_in_total[changes_in_total < 0]).item()
#
#                 # Store the values for calculating overall averages
#                 max_negative_shifts_total.append(max_negative_shift_total)
#                 avg_negative_shifts_total.append(avg_negative_shift_total)
#
#                 log_message(f'Maximum negative shift total: {max_negative_shift_total}')
#                 log_message(f'Average negative shift total: {avg_negative_shift_total}')
#             else:
#                 log_message("No negative shifts found.")
#
#             log_message("=" * 40)
#
#         except Exception as e:
#             log_message(f"Error while processing graph {i + 1}: {e}")
#             continue
#
#     if max_positive_shifts:
#         overall_avg_max_positive_shift = torch.mean(torch.tensor(max_positive_shifts)).item()
#         overall_avg_avg_positive_shift = torch.mean(torch.tensor(avg_positive_shifts)).item()
#         overall_std_max_positive_shift = torch.std(torch.tensor(max_positive_shifts)).item()
#         overall_std_avg_positive_shift = torch.std(torch.tensor(avg_positive_shifts)).item()
#         log_message(f'Overall average of max positive shifts: {overall_avg_max_positive_shift}')
#         log_message(f'Overall standard deviation of max positive shifts: {overall_std_max_positive_shift}')
#         log_message(f'Overall average of avg positive shifts: {overall_avg_avg_positive_shift}')
#         log_message(f'Overall standard deviation of avg positive shifts: {overall_std_avg_positive_shift}')
#     else:
#         log_message("No positive shifts found in any graphs.")
#
#     if max_negative_shifts:
#         overall_avg_max_negative_shift = torch.mean(torch.tensor(max_negative_shifts)).item()
#         overall_avg_avg_negative_shift = torch.mean(torch.tensor(avg_negative_shifts)).item()
#         overall_std_max_negative_shift = torch.std(torch.tensor(max_negative_shifts)).item()
#         overall_std_avg_negative_shift = torch.std(torch.tensor(avg_negative_shifts)).item()
#         log_message(f'Overall average of max negative shifts: {overall_avg_max_negative_shift}')
#         log_message(f'Overall standard deviation of max negative shifts: {overall_std_max_negative_shift}')
#         log_message(f'Overall average of avg negative shifts: {overall_avg_avg_negative_shift}')
#         log_message(f'Overall standard deviation of avg negative shifts: {overall_std_avg_negative_shift}')
#     else:
#         log_message("No negative shifts found in any graphs.")
#
#     if max_positive_shifts_total:
#         overall_avg_max_positive_shift_total = torch.mean(torch.tensor(max_positive_shifts_total)).item()
#         overall_avg_avg_positive_shift_total = torch.mean(torch.tensor(avg_positive_shifts_total)).item()
#         overall_std_max_positive_shift_total = torch.std(torch.tensor(max_positive_shifts_total)).item()
#         overall_std_avg_positive_shift_total = torch.std(torch.tensor(avg_positive_shifts_total)).item()
#         log_message(f'Overall average of max positive shifts total: {overall_avg_max_positive_shift_total}')
#         log_message(f'Overall standard deviation of max positive shifts total: {overall_std_max_positive_shift_total}')
#         log_message(f'Overall average of avg positive shifts total: {overall_avg_avg_positive_shift_total}')
#         log_message(f'Overall standard deviation of avg positive shifts total: {overall_std_avg_positive_shift_total}')
#     else:
#         log_message("No positive shifts found in any graphs.")
#
#     if max_negative_shifts_total:
#         overall_avg_max_negative_shift_total = torch.mean(torch.tensor(max_negative_shifts_total)).item()
#         overall_avg_avg_negative_shift_total = torch.mean(torch.tensor(avg_negative_shifts_total)).item()
#         overall_std_max_negative_shift_total = torch.std(torch.tensor(max_negative_shifts_total)).item()
#         overall_std_avg_negative_shift_total = torch.std(torch.tensor(avg_negative_shifts_total)).item()
#         log_message(f'Overall average of max negative shifts total: {overall_avg_max_negative_shift_total}')
#         log_message(f'Overall standard deviation of max negative shifts total: {overall_std_max_negative_shift_total}')
#         log_message(f'Overall average of avg negative shifts total: {overall_avg_avg_negative_shift_total}')
#         log_message(f'Overall standard deviation of avg negative shifts total: {overall_std_avg_negative_shift_total}')
#     else:
#         log_message("No negative shifts found in any graphs.")
#
#     # average_dirichlet_energy_original = torch.mean(torch.tensor(dirichlet_energies_original)).item()
#     # std_dirichlet_energy_original = torch.std(torch.tensor(dirichlet_energies_original)).item()
#     # average_dirichlet_energy_wired = torch.mean(torch.tensor(dirichlet_energies_wired)).item()
#     # std_dirichlet_energy_wired = torch.std(torch.tensor(dirichlet_energies_wired)).item()
#     # log_message(f'Average Dirichlet Energy of Original Dataset: {average_dirichlet_energy_original} ±'
#     #             f'{std_dirichlet_energy_original}')
#     # log_message(f'Average Dirichlet Energy of Wired Dataset: {average_dirichlet_energy_wired} ±'
#     #             f'{std_dirichlet_energy_wired}')
#
#     log_message("All graphs processed.")

def compute_over_squashing_metrics(decay_rates: List[float]) -> Tuple[float, float, float, float]:
    """
    Compute over-squashing metrics:
      1. Over-Squashing Prevalence (Y_pre): fraction of node pairs with positive decay rates.
      2. Average Decay Rate (Y_avg): average decay rate over all node pairs.
      3. Variance of Decay Rates (Y_var): variability of the decay rates.
      4. Skewness of Decay Rates (Y_skew): asymmetry of the decay rate distribution.

    Parameters:
        decay_rates (List[float]): List of decay rates k_{vu} for each node pair.

    Returns:
        Tuple[float, float, float, float]: (Y_pre, Y_avg, Y_var, Y_skew)
    """
    if not decay_rates:
        return 0.0, 0.0, 0.0, 0.0

    N = len(decay_rates)

    # 1. Prevalence: fraction of pairs with a positive decay rate.
    num_positive = sum(1 for rate in decay_rates if rate > 0)
    Y_pre = num_positive / N

    # 2. Average decay rate.
    Y_avg = sum(decay_rates) / N

    # 3. Variance of decay rates.
    Y_var = sum((rate - Y_avg) ** 2 for rate in decay_rates) / N

    # 4. Skewness of decay rates.
    std = math.sqrt(Y_var) if Y_var > 0 else 0.0
    if std > 0:
        Y_skew = sum(((rate - Y_avg) / std) ** 3 for rate in decay_rates) / N
    else:
        Y_skew = 0.0

    return Y_pre, Y_avg, Y_var, Y_skew


def graph_level_average_toy() -> None:
    """
    Process toy datasets (each with one graph) to compute and log over-squashing metrics
    for both the original and rewired graphs.
    """
    args = get_args()
    # Define toy datasets
    datasets = [
        ("Ring", generate_ring_transfer_graph(10, [0, 1], False, True)),
        ("CrossedRing", generate_ring_transfer_graph(10, [0, 1], True, True)),
        ("Lolipop", generate_lollipop_transfer_graph(8, [0, 1], True)),
        ("Tree", generate_tree_transfer_graph(2, [0, 1], 2))
    ]

    # Create logging directory
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'decay_rate_toy_logs')
    os.makedirs(path, exist_ok=True)

    for name, dataset in datasets:
        try:
            # Setup logging for current dataset
            log_file = f'{path}/{name}_{args.rewiring}.log'
            setup_logging(log_file)
            log_message(f"Processing dataset: {name} using rewiring method: {args.rewiring}")

            # --------------------
            # ORIGINAL GRAPH METRICS
            # --------------------
            visualize_graph(dataset, dataset.num_nodes, f'{name}_original', args.rewiring)
            adj = get_adj(dataset.edge_index, set_diag=False, symmetric_normalize=False)
            original_diameter = compute_diameter(dataset)
            log_message(f"{name} Original Diameter: {original_diameter}")

            # Compute decay rates and over-squashing metrics for the original graph
            decay_rates_original, *_ = decay_rate(adj, diameter=original_diameter)
            Y_pre_orig, Y_avg_orig, Y_var_orig, Y_skew_orig = compute_over_squashing_metrics(decay_rates_original)
            log_message(f"{name} Original Over-Squashing Prevalence: {Y_pre_orig:.3f}")
            log_message(f"{name} Original Average Decay Rate: {Y_avg_orig:.3f}")
            log_message(f"{name} Original Variance of Decay Rates: {Y_var_orig:.3f}")
            log_message(f"{name} Original Skewness of Decay Rates: {Y_skew_orig:.3f}")

            # --------------------
            # REWIRED GRAPH METRICS
            # --------------------
            log_message(f"TESTING rewiring for: {name} ({args.rewiring})")
            # Deepcopy to avoid modifying the original graph
            rewired_dataset = deepcopy(dataset)
            if args.rewiring == "fosr":
                edge_index, edge_type, _ = fosr.edge_rewire(rewired_dataset.edge_index.numpy(),
                                                            num_iterations=args.num_iterations)
                rewired_dataset.edge_index = torch.tensor(edge_index)
                rewired_dataset.edge_type = torch.tensor(edge_type)
            elif args.rewiring == "sdrf":
                rewired_dataset.edge_index, rewired_dataset.edge_type = sdrf.sdrf(
                    rewired_dataset, loops=args.num_iterations, remove_edges=False, is_undirected=True)
            elif args.rewiring == "digl":
                rewired_dataset.edge_index = digl.rewire(rewired_dataset, alpha=args.alpha, eps=args.eps)
                m = rewired_dataset.edge_index.shape[1]
                rewired_dataset.edge_type = torch.tensor(np.zeros(m, dtype=np.int64))
            elif args.rewiring == "borf":
                log_message(f"[INFO] BORF hyper-parameter: num_iterations = {args.num_iterations}")
                log_message(f"[INFO] BORF hyper-parameter: batch_add = {args.borf_batch_add}")
                log_message(f"[INFO] BORF hyper-parameter: batch_remove = {args.borf_batch_remove}")
                rewired_dataset.edge_index, rewired_dataset.edge_type = borf.borf3(
                    rewired_dataset, loops=args.num_iterations, remove_edges=False, is_undirected=True,
                    batch_add=args.borf_batch_add, batch_remove=args.borf_batch_remove,
                    dataset_name=name, graph_index=0)
            else:
                log_message("No rewiring method specified. Skipping rewiring.")

            # Compute adjacency and diameter for the rewired graph
            log_message(f"Processing Rewired Graph: {name}")
            visualize_graph(rewired_dataset, rewired_dataset.num_nodes, f'{name}_rewired', args.rewiring)
            rewired_adj = get_adj(rewired_dataset.edge_index, set_diag=False, symmetric_normalize=False)
            rewired_diameter = compute_diameter(rewired_dataset)
            log_message(f"{name} Rewired Diameter: {rewired_diameter}")

            # Compute decay rates and over-squashing metrics for the rewired graph
            decay_rates_rewired, *_ = decay_rate(rewired_adj, diameter=rewired_diameter)
            Y_pre_rew, Y_avg_rew, Y_var_rew, Y_skew_rew = compute_over_squashing_metrics(decay_rates_rewired)
            log_message(f"{name} Rewired Over-Squashing Prevalence: {Y_pre_rew:.3f}")
            log_message(f"{name} Rewired Average Decay Rate: {Y_avg_rew:.3f}")
            log_message(f"{name} Rewired Variance of Decay Rates: {Y_var_rew:.3f}")
            log_message(f"{name} Rewired Skewness of Decay Rates: {Y_skew_rew:.3f}")

            # --------------------
            # COMPUTE INDIVIDUAL TREATMENT EFFECTS (ITEs)
            # --------------------
            ITE_pre = Y_pre_rew - Y_pre_orig
            ITE_avg = Y_avg_rew - Y_avg_orig
            ITE_var = Y_var_rew - Y_var_orig
            ITE_skew = Y_skew_rew - Y_skew_orig

            log_message(f"{name} ITE Prevalence: {ITE_pre:.3f}")
            log_message(f"{name} ITE Average: {ITE_avg:.3f}")
            log_message(f"{name} ITE Variance: {ITE_var:.3f}")
            log_message(f"{name} ITE Skewness: {ITE_skew:.3f}")
            log_message('=' * 40)

        except Exception as e:
            logging.error(f"Error processing {name}: {str(e)}")
            continue


def load_datasets():
    combined_transform = Compose([AddSelfLoopsTransform(), CustomTransform()])

    datasets = {
        # "mutag": list(TUDataset(root="data", name="MUTAG", transform=AddSelfLoopsTransform())),
        # "enzymes": list(TUDataset(root="data", name="ENZYMES", transform=AddSelfLoopsTransform())),
        # "proteins": list(TUDataset(root="data", name="PROTEINS", transform=AddSelfLoopsTransform())),
        # "imdb_binary": list(TUDataset(root="data", name="IMDB-BINARY", transform=combined_transform)),
        # "collab": list(TUDataset(root="data", name="COLLAB", transform=combined_transform)),
        "reddit_binary": list(TUDataset(root="data", name="REDDIT-BINARY", transform=combined_transform)),
        # "peptides-func": list(LRGBDataset(root="data", name="Peptides-func", transform=AddSelfLoopsTransform())),
        # "peptides-struct": list(LRGBDataset(root="data", name="Peptides-struct", transform=AddSelfLoopsTransform()))
    }

    return datasets

def apply_rewiring(rewired_dataset: List[Any], rewiring_method: str, args: Any, name: str) -> List[Any]:
    """
    Apply rewiring to a dataset based on the specified method.

    Parameters:
        rewired_dataset (List[Any]): List of graph data objects.
        rewiring_method (str): The rewiring method to use.
        args (Any): Arguments containing hyperparameters.
        name (str): Name of the dataset.

    Returns:
        List[Any]: The rewired dataset.
    """
    for idx, data in tqdm(enumerate(rewired_dataset), total=len(rewired_dataset), desc="Rewiring"):
        try:
            if rewiring_method == "fosr":
                edge_index, edge_type, _ = fosr.edge_rewire(data.edge_index.numpy(), num_iterations=args.num_iterations)
            elif rewiring_method == "sdrf":
                edge_index, edge_type = sdrf.sdrf(data, loops=args.num_iterations, remove_edges=False,
                                                  is_undirected=True)
            elif rewiring_method == "digl":
                edge_index = digl.rewire(data, alpha=args.alpha, eps=args.eps)
                edge_type = torch.zeros(edge_index.shape[1], dtype=torch.int64)
            elif rewiring_method == "borf":
                # log_message(f"BORF hyper-parameter : num_iterations = {args.num_iterations}")
                # log_message(f"BORF hyper-parameter : batch_add = {args.borf_batch_add}")
                # log_message(f"BORF hyper-parameter : batch_remove = {args.borf_batch_remove}")
                edge_index, edge_type = borf.borf3(data, loops=args.num_iterations, remove_edges=False, # check later
                                                   is_undirected=True,
                                                   batch_add=args.borf_batch_add, batch_remove=args.borf_batch_remove,
                                                   dataset_name=name, graph_index=idx)
            else:
                log_message("No rewiring method specified. Skipping rewiring.")
                return rewired_dataset
        except Exception as e:
            log_message(f"Rewiring failed for graph index {idx} with error: {e}")
            continue  # Optionally, skip this graph or handle differently

        data.edge_index = torch.as_tensor(edge_index).clone().detach()
        data.edge_type = torch.as_tensor(edge_type).clone().detach()

    return rewired_dataset


def perform_t_tests(pre_list: List[float],
                    ave_list: List[float],
                    var_list: List[float],
                    skew_list: List[float],
                    alpha: float = 0.05) -> Dict[str, Tuple[float, float]]:
    """
    Perform one-sample t-tests on the lists of individual treatment effects (ITEs) for each metric.

    Parameters:
        pre_list, ave_list, var_list, skew_list (List[float]): Lists of ITEs.
        alpha (float): Overall significance level.

    Returns:
        Dict[str, Tuple[float, float]]: Dictionary with metric names as keys and (t_stat, p_value) as values.
    """
    results = {}
    tests = {
        'prevalence': pre_list,
        'average': ave_list,
        'variance': var_list,
        'skewness': skew_list
    }
    # Bonferroni correction for multiple comparisons.
    corrected_alpha = alpha / len(tests)

    for metric, values in tests.items():
        t_stat, p_val = stats.ttest_1samp(values, 0)
        results[metric] = (t_stat, p_val, corrected_alpha)
    return results


def compute_metrics_for_dataset(dataset: List[Any], name: str, args: Any, device: torch.device):
    """
    Compute individual treatment effects (ITEs) for over-squashing metrics for each graph in the dataset.

    Parameters:
        dataset (List[Any]): List of graph data objects.
        name (str): Name of the dataset.
        args (Any): Arguments with hyperparameters.
        device (torch.device): Computation device.

    Returns:
        Tuple: (ITE metric lists, t-test results)
    """
    ITE_pre_list, ITE_ave_list, ITE_var_list, ITE_skew_list = [], [], [], []
    # Y_pre_list, Y_ave_list, Y_var_list, Y_skew_list = [], [], [], []
    for idx, data in enumerate(tqdm(dataset, desc="Processing decay rates")):
        # Compute original graph metrics.
        try:
            # if is_connected(data) != 1:
            #     log_message(f"{name} graph {idx} is not connected, skipped.")
            #     continue
            adj = get_adj(data.edge_index, set_diag=False, symmetric_normalize=False, device=device)
            diameter = compute_diameter(data)
            decay_vals, _ = decay_rate(adj, diameter=diameter)
            if len(decay_vals) == 0:
                print(f'No decay rate in graph {idx}')
                continue
            Y_pre, Y_avg, Y_var, Y_skew = compute_over_squashing_metrics(decay_vals)
            # Y_pre_list.append(Y_pre)
            # Y_ave_list.append(Y_avg)
            # Y_var_list.append(Y_var)
            # Y_skew_list.append(Y_skew)

            # Compute metrics for rewired graph.
            rewired_data = apply_rewiring([data], args.rewiring, args, name)[0]
            rewired_adj = get_adj(rewired_data.edge_index, set_diag=False, symmetric_normalize=False, device=device)
            rewired_diameter = compute_diameter(rewired_data)
            decay_vals_r, _ = decay_rate(rewired_adj, diameter=rewired_diameter)
            Y_pre_r, Y_avg_r, Y_var_r, Y_skew_r = compute_over_squashing_metrics(decay_vals_r)

            # Compute individual treatment effects (ITEs).
            ITE_pre_list.append(Y_pre_r - Y_pre)
            ITE_ave_list.append(Y_avg_r - Y_avg)
            ITE_var_list.append(Y_var_r - Y_var)
            ITE_skew_list.append(Y_skew_r - Y_skew)
        except Exception as e:
            log_message(f"Error processing graph index {idx}: {e}")
            continue

    t_test_results = perform_t_tests(ITE_pre_list, ITE_ave_list, ITE_var_list, ITE_skew_list)
    return (ITE_pre_list, ITE_ave_list, ITE_var_list, ITE_skew_list), t_test_results
    # return Y_pre_list, Y_ave_list, Y_var_list, Y_skew_list


def graph_level_average():
    args = get_args()
    datasets = load_datasets()
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'decay_rate_logs')
    os.makedirs(path, exist_ok=True)
    device = torch.device('cpu')  # Consider parameterizing the device.

    for name, dataset in datasets.items():
        setup_logging(f'{path}/{name}_{args.rewiring}_{args.layer_type}.log')
        ite_lists, t_test_results = compute_metrics_for_dataset(dataset, name, args, device)
        # Y_pre_list, Y_ave_list, Y_var_list, Y_skew_list = compute_metrics_for_dataset(dataset, name, args, device)
        ITE_pre_list, ITE_ave_list, ITE_var_list, ITE_skew_list = ite_lists

        ate_pre = torch.mean(torch.tensor(ITE_pre_list)).item()
        ate_ave = torch.mean(torch.tensor(ITE_ave_list)).item()
        ate_var = torch.mean(torch.tensor(ITE_var_list)).item()
        ate_skew = torch.mean(torch.tensor(ITE_skew_list)).item()

        for metric, (t_stat, p_val, corr_alpha) in t_test_results.items():
            ate = {'prevalence': ate_pre, 'average': ate_ave, 'variance': ate_var, 'skewness': ate_skew}[metric]
            significance = 'Significant' if p_val < corr_alpha else 'Not Significant'
            log_message(f"{metric.capitalize()} ATE: {ate} (t={t_stat:.2f}, p={p_val:.4f}, {significance})")
        # log_message("Original Graph Stats:")
        # log_message(f"Prevalence ATE: {ate_pre}")
        # log_message(f"Average ATE: {ate_ave}")
        # log_message(f"Variance ATE: {ate_var}")
        # log_message(f"Skewness ATE: {ate_skew}")


# def test_and_plot_real_datasets_list():
#     args = get_args()
#     datasets = {
#         # "mutag": list(TUDataset(root="data", name="MUTAG", transform=AddSelfLoopsTransform())),
#         # "enzymes": list(TUDataset(root="data", name="ENZYMES", transform=AddSelfLoopsTransform())),
#         # "proteins": list(TUDataset(root="data", name="PROTEINS", transform=AddSelfLoopsTransform())),
#         "imdb_binary": list(TUDataset(root="data", name="IMDB-BINARY",
#                                       transform=T.Compose([Custom_transform(), AddSelfLoopsTransform()]))),
#         # "collab": list(TUDataset(root="data", name="COLLAB", transform=T.Compose([Custom_transform(), AddSelfLoopsTransform()]))),
#         # "reddit_binary": list(TUDataset(root="data", name="REDDIT-BINARY", transform=T.Compose([Custom_transform(), AddSelfLoopsTransform()]))),
#     }
#
#     device = torch.device('cpu')
#     path = osp.join(osp.dirname(osp.realpath(__file__)), 'decay_rate_logs')
#     if not osp.exists(path):
#         os.makedirs(path)
#
#     # original_and_rewired = {}
#
#     for key, dataset in datasets.items():
#         log_file = f'{path}/{key}_{args.rewiring}.log'
#         setup_logging(log_file)
#
#         print(f"TESTING: {key} ({args.rewiring})")
#
#         rewired_dataset = deepcopy(dataset)
#
#         if args.rewiring == "fosr":
#             for idx, data in tqdm(enumerate(rewired_dataset), total=len(rewired_dataset), desc="Rewiring"):
#                 edge_index, edge_type, _ = fosr.edge_rewire(data.edge_index.numpy(),
#                                                             num_iterations=args.num_iterations)
#                 data.edge_index = torch.tensor(edge_index)
#                 data.edge_type = torch.tensor(edge_type)
#         elif args.rewiring == "sdrf":
#             for idx, data in tqdm(enumerate(rewired_dataset), total=len(rewired_dataset), desc="Rewiring"):
#                 data.edge_index, data.edge_type = sdrf.sdrf(data, loops=args.num_iterations,
#                                                             remove_edges=False, is_undirected=True)
#         elif args.rewiring == "digl":
#             for idx, data in tqdm(enumerate(rewired_dataset), total=len(rewired_dataset), desc="Rewiring"):
#                 data.edge_index = digl.rewire(data, alpha=args.alpha, eps=args.eps)
#                 m = data.edge_index.shape[1]
#                 data.edge_type = torch.tensor(np.zeros(m, dtype=np.int64))
#
#         elif args.rewiring == "borf":
#             print(f"[INFO] BORF hyper-parameter : num_iterations = {args.num_iterations}")
#             print(f"[INFO] BORF hyper-parameter : batch_add = {args.borf_batch_add}")
#             print(f"[INFO] BORF hyper-parameter : batch_remove = {args.borf_batch_remove}")
#             for idx, data in tqdm(enumerate(rewired_dataset), total=len(rewired_dataset), desc="Rewiring"):
#                 data.edge_index, data.edge_type = borf.borf3(data,
#                                                              loops=args.num_iterations,
#                                                              remove_edges=False,
#                                                              is_undirected=True,
#                                                              batch_add=args.borf_batch_add,
#                                                              batch_remove=args.borf_batch_remove,
#                                                              dataset_name=key,
#                                                              graph_index=idx)
#
#         elif args.rewiring == 'rw':
#             for idx, data in tqdm(enumerate(rewired_dataset), total=len(rewired_dataset), desc="Rewiring"):
#                 diameter = compute_diameter(data)
#
#                 if diameter == 1:
#                     k = 0
#                 else:
#                     k = calculate_num_random_edges(data, diameter, data.num_nodes)
#
#                 edge_index, edge_type, _, _, n = gnm_random_graph_v2(k, args.seed, graph=data)
#                 data.edge_index = edge_index
#                 data.edge_type = edge_type
#         else:
#             pass
#
#         # Lists to store the positive shift values for all graphs
#         max_positive_shifts = []
#         avg_positive_shifts = []
#         max_negative_shifts = []
#         avg_negative_shifts = []
#         max_positive_shifts_total = []
#         avg_positive_shifts_total = []
#         max_negative_shifts_total = []
#         avg_negative_shifts_total = []
#         positive_percentages = []
#         negative_percentages = []
#         unchanged_percentages = []
#         positive_percentages_total = []
#         negative_percentages_total = []
#         unchanged_percentages_total = []
#
#         log_message(f"Processing dataset with {len(dataset)} graphs")
#         for i, (data, wired_data) in enumerate(zip(dataset, rewired_dataset)):
#             try:
#                 adj = get_adj(data.edge_index, set_diag=False, symmetric_normalize=False, device=device)
#                 original_diameter = compute_diameter(original_graph=data, rewired_graph=None)
#                 decay_rates_original, decay_rates_original_nan = decay_rate(adj, diameter=original_diameter,
#                                                                             device=device)
#
#                 adj = get_adj(wired_data.edge_index, set_diag=False, symmetric_normalize=False, device=device)
#                 rewired_diameter = compute_diameter(original_graph=None, rewired_graph=wired_data)
#                 decay_rates_wired, decay_rates_wired_nan = decay_rate(adj, diameter=rewired_diameter, device=device)
#
#                 decay_rates_original_nan = torch.tensor(decay_rates_original_nan)
#                 decay_rates_wired_nan = torch.tensor(decay_rates_wired_nan)
#
#                 mask = ~torch.isnan(decay_rates_original_nan) & ~torch.isnan(decay_rates_wired_nan)
#                 filtered_original = decay_rates_original_nan[mask]
#                 filtered_wired = decay_rates_wired_nan[mask]
#
#                 if len(filtered_original) == 0:
#                     log_message(f"Graph {i + 1} has no valid decay rates after filtering.")
#                     log_message("=" * 40)
#                     continue
#
#                 # if len(filtered_original) <= 10:
#                 #     num_extreme_values = len(filtered_original)
#                 # else:
#                 #     num_extreme_values = int(0.1 * len(filtered_original))
#                 #
#                 # # Find the indices of the most negative decay rates in the original distribution
#                 # sorted_indices = torch.argsort(filtered_original)
#                 # most_negative_indices = sorted_indices[filtered_original[sorted_indices] < 0][:num_extreme_values]
#                 #
#                 # # Extract the corresponding values from both filtered distributions
#                 # most_negative_original = filtered_original[most_negative_indices]
#                 # most_negative_wired = filtered_wired[most_negative_indices]
#
#                 positive_mask = filtered_original > 0.0
#                 most_positive_original = filtered_original[positive_mask]
#                 most_positive_wired = filtered_wired[positive_mask]
#
#                 # Calculate how these values have changed
#                 changes_in_positive = most_positive_wired - most_positive_original
#                 # The total change (both negative and positive)
#                 changes_in_total = filtered_wired - filtered_original
#
#                 num_positive_changes = torch.sum(changes_in_positive > 0).item()  # baddies
#                 num_negative_changes = torch.sum(changes_in_positive < 0).item()  # goodies
#                 num_zero_changes = torch.sum(changes_in_positive == 0).item()
#                 total_changes = len(changes_in_positive)
#
#                 num_positive_changes_total = torch.sum(changes_in_total > 0).item()
#                 num_negative_changes_total = torch.sum(changes_in_total < 0).item()
#                 num_zero_changes_total = torch.sum(changes_in_total == 0).item()
#                 total_changes_total = len(changes_in_total)
#
#                 if total_changes > 0:
#                     positive_percentage = (num_positive_changes / total_changes) * 100
#                     negative_percentage = (num_negative_changes / total_changes) * 100
#                     zero_percentage = (num_zero_changes / total_changes) * 100
#                     positive_percentages.append(positive_percentage)
#                     negative_percentages.append(negative_percentage)
#                     unchanged_percentages.append(zero_percentage)
#                     log_message(f'Graph {i + 1}:')
#                     log_message(f'Percentage of positive shifts: {positive_percentage:.2f}%')
#                     log_message(f'Percentage of negative shifts: {negative_percentage:.2f}%')
#                     log_message(f'Percentage of no change: {zero_percentage:.2f}%')
#                 else:
#                     log_message(f"Graph {i + 1}: No valid decay rate changes found.")
#
#                 if total_changes_total > 0:
#                     positive_percentage_total = (num_positive_changes_total / total_changes_total) * 100
#                     negative_percentage_total = (num_negative_changes_total / total_changes_total) * 100
#                     zero_percentage_total = (num_zero_changes_total / total_changes_total) * 100
#                     positive_percentages_total.append(positive_percentage_total)
#                     negative_percentages_total.append(negative_percentage_total)
#                     unchanged_percentages_total.append(zero_percentage_total)
#                     # log_message(f'Graph {i + 1}:')
#                     log_message(f'Percentage of positive shifts total: {positive_percentage_total:.2f}%')
#                     log_message(f'Percentage of negative shifts total: {negative_percentage_total:.2f}%')
#                     log_message(f'Percentage of no change total: {zero_percentage_total:.2f}%')
#                 else:
#                     log_message(f"Graph {i + 1}: No valid decay rate changes found.")
#
#                 # Calculate the maximum and average positive and negative shifts
#                 if num_positive_changes > 0:
#                     max_positive_shift = torch.max(changes_in_positive[changes_in_positive > 0]).item()
#                     avg_positive_shift = torch.mean(changes_in_positive[changes_in_positive > 0]).item()
#
#                     # Store the values for calculating overall averages
#                     max_positive_shifts.append(max_positive_shift)
#                     avg_positive_shifts.append(avg_positive_shift)
#
#                     log_message(f'Maximum positive shift: {max_positive_shift}')
#                     log_message(f'Average positive shift: {avg_positive_shift}')
#                 else:
#                     log_message("No positive shifts found.")
#
#                 if num_negative_changes > 0:
#                     max_negative_shift = torch.min(changes_in_positive[changes_in_positive < 0]).item()
#                     avg_negative_shift = torch.mean(changes_in_positive[changes_in_positive < 0]).item()
#
#                     # Store the values for calculating overall averages
#                     max_negative_shifts.append(max_negative_shift)
#                     avg_negative_shifts.append(avg_negative_shift)
#
#                     log_message(f'Maximum negative shift: {max_negative_shift}')
#                     log_message(f'Average negative shift: {avg_negative_shift}')
#                 else:
#                     log_message("No negative shifts found.")
#
#                 if num_positive_changes_total > 0:
#                     max_positive_shift_total = torch.max(changes_in_total[changes_in_total > 0]).item()
#                     avg_positive_shift_total = torch.mean(changes_in_total[changes_in_total > 0]).item()
#
#                     # Store the values for calculating overall averages
#                     max_positive_shifts_total.append(max_positive_shift_total)
#                     avg_positive_shifts_total.append(avg_positive_shift_total)
#
#                     log_message(f'Maximum positive shift total: {max_positive_shift_total}')
#                     log_message(f'Average positive shift total: {avg_positive_shift_total}')
#                 else:
#                     log_message("No positive shifts found.")
#
#                 if num_negative_changes_total > 0:
#                     max_negative_shift_total = torch.min(changes_in_total[changes_in_total < 0]).item()
#                     avg_negative_shift_total = torch.mean(changes_in_total[changes_in_total < 0]).item()
#
#                     # Store the values for calculating overall averages
#                     max_negative_shifts_total.append(max_negative_shift_total)
#                     avg_negative_shifts_total.append(avg_negative_shift_total)
#
#                     log_message(f'Maximum negative shift total: {max_negative_shift_total}')
#                     log_message(f'Average negative shift total: {avg_negative_shift_total}')
#                 else:
#                     log_message("No negative shifts found.")
#
#                 log_message("=" * 40)
#
#             except Exception as e:
#                 log_message(f"Error while processing graph {i + 1}: {e}")
#                 continue
#
#         overall_avg_positive_percentage = torch.mean(torch.tensor(positive_percentages)).item()
#         overall_avg_negative_percentage = torch.mean(torch.tensor(negative_percentages)).item()
#         overall_avg_unchanged_percentage = torch.mean(torch.tensor(unchanged_percentages)).item()
#         overall_std_positive_percentage = torch.std(torch.tensor(positive_percentages)).item()
#         overall_std_negative_percentage = torch.std(torch.tensor(negative_percentages)).item()
#         overall_std_unchanged_percentage = torch.std(torch.tensor(unchanged_percentages)).item()
#         log_message(f'Overall average of positive percentage: {overall_avg_positive_percentage}')
#         log_message(f'Overall standard deviation of positive percentage: {overall_std_positive_percentage}')
#         log_message(f'Overall average of negative percentage: {overall_avg_negative_percentage}')
#         log_message(f'Overall standard deviation of negative percentage: {overall_std_negative_percentage}')
#         log_message(f'Overall average of unchanged percentage: {overall_avg_unchanged_percentage}')
#         log_message(f'Overall standard deviation of unchanged percentage: {overall_std_unchanged_percentage}')
#
#         if max_positive_shifts:
#             overall_avg_max_positive_shift = torch.mean(torch.tensor(max_positive_shifts)).item()
#             overall_avg_avg_positive_shift = torch.mean(torch.tensor(avg_positive_shifts)).item()
#             overall_std_max_positive_shift = torch.std(torch.tensor(max_positive_shifts)).item()
#             overall_std_avg_positive_shift = torch.std(torch.tensor(avg_positive_shifts)).item()
#             log_message(f'Overall average of max positive shifts: {overall_avg_max_positive_shift}')
#             log_message(f'Overall standard deviation of max positive shifts: {overall_std_max_positive_shift}')
#             log_message(f'Overall average of avg positive shifts: {overall_avg_avg_positive_shift}')
#             log_message(f'Overall standard deviation of avg positive shifts: {overall_std_avg_positive_shift}')
#         else:
#             log_message("No positive shifts found in any graphs.")
#
#         if max_negative_shifts:
#             overall_avg_max_negative_shift = torch.mean(torch.tensor(max_negative_shifts)).item()
#             overall_avg_avg_negative_shift = torch.mean(torch.tensor(avg_negative_shifts)).item()
#             overall_std_max_negative_shift = torch.std(torch.tensor(max_negative_shifts)).item()
#             overall_std_avg_negative_shift = torch.std(torch.tensor(avg_negative_shifts)).item()
#             log_message(f'Overall average of max negative shifts: {overall_avg_max_negative_shift}')
#             log_message(f'Overall standard deviation of max negative shifts: {overall_std_max_negative_shift}')
#             log_message(f'Overall average of avg negative shifts: {overall_avg_avg_negative_shift}')
#             log_message(f'Overall standard deviation of avg negative shifts: {overall_std_avg_negative_shift}')
#         else:
#             log_message("No negative shifts found in any graphs.")
#
#         overall_avg_positive_percentage_total = torch.mean(torch.tensor(positive_percentages_total)).item()
#         overall_avg_negative_percentage_total = torch.mean(torch.tensor(negative_percentages_total)).item()
#         overall_avg_unchanged_percentage_total = torch.mean(torch.tensor(unchanged_percentages_total)).item()
#         overall_std_positive_percentage_total = torch.std(torch.tensor(positive_percentages_total)).item()
#         overall_std_negative_percentage_total = torch.std(torch.tensor(negative_percentages_total)).item()
#         overall_std_unchanged_percentage_total = torch.std(torch.tensor(unchanged_percentages_total)).item()
#         log_message(f'Overall average of positive percentage total: {overall_avg_positive_percentage_total}')
#         log_message(f'Overall standard deviation of positive percentage total: {overall_std_positive_percentage_total}')
#         log_message(f'Overall average of negative percentage total: {overall_avg_negative_percentage_total}')
#         log_message(f'Overall standard deviation of negative percentage total: {overall_std_negative_percentage_total}')
#         log_message(f'Overall average of unchanged percentage total: {overall_avg_unchanged_percentage_total}')
#         log_message(
#             f'Overall standard deviation of unchanged percentage total: {overall_std_unchanged_percentage_total}')
#
#         if max_positive_shifts_total:
#             overall_avg_max_positive_shift_total = torch.mean(torch.tensor(max_positive_shifts_total)).item()
#             overall_avg_avg_positive_shift_total = torch.mean(torch.tensor(avg_positive_shifts_total)).item()
#             overall_std_max_positive_shift_total = torch.std(torch.tensor(max_positive_shifts_total)).item()
#             overall_std_avg_positive_shift_total = torch.std(torch.tensor(avg_positive_shifts_total)).item()
#             log_message(f'Overall average of max positive shifts total: {overall_avg_max_positive_shift_total}')
#             log_message(
#                 f'Overall standard deviation of max positive shifts total: {overall_std_max_positive_shift_total}')
#             log_message(f'Overall average of avg positive shifts total: {overall_avg_avg_positive_shift_total}')
#             log_message(
#                 f'Overall standard deviation of avg positive shifts total: {overall_std_avg_positive_shift_total}')
#         else:
#             log_message("No positive shifts found in any graphs.")
#
#         if max_negative_shifts_total:
#             overall_avg_max_negative_shift_total = torch.mean(torch.tensor(max_negative_shifts_total)).item()
#             overall_avg_avg_negative_shift_total = torch.mean(torch.tensor(avg_negative_shifts_total)).item()
#             overall_std_max_negative_shift_total = torch.std(torch.tensor(max_negative_shifts_total)).item()
#             overall_std_avg_negative_shift_total = torch.std(torch.tensor(avg_negative_shifts_total)).item()
#             log_message(f'Overall average of max negative shifts total: {overall_avg_max_negative_shift_total}')
#             log_message(
#                 f'Overall standard deviation of max negative shifts total: {overall_std_max_negative_shift_total}')
#             log_message(f'Overall average of avg negative shifts total: {overall_avg_avg_negative_shift_total}')
#             log_message(
#                 f'Overall standard deviation of avg negative shifts total: {overall_std_avg_negative_shift_total}')
#         else:
#             log_message("No negative shifts found in any graphs.")
#
#         log_message(f"All graphs processed for dataset {key}.")
#
#     log_message("All datasets processed.")

def node_classification_datasets():
    largest_cc = LargestConnectedComponents()

    # cornell = WebKB(root="data", name="Cornell", transform=largest_cc)
    # wisconsin = WebKB(root="data", name="Wisconsin", transform=largest_cc)
    # texas = WebKB(root="data", name="Texas", transform=largest_cc)
    chameleon = WikipediaNetwork(root="data", name="chameleon", transform=largest_cc)
    # cora = Planetoid(root="data", name="cora", transform=largest_cc)
    # citeseer = Planetoid(root="data", name="citeseer", transform=largest_cc)
    # datasets = {"cornell": cornell, "wisconsin": wisconsin, "texas": texas, "chameleon": chameleon, "cora": cora,
    #             "citeseer": citeseer}
    datasets = {"chameleon": chameleon}

    for name, dataset in datasets.items():
        dataset.data.edge_index = to_undirected(dataset.data.edge_index)

    args = get_args()
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'node_classification_logs')
    os.makedirs(path, exist_ok=True)

    for name, dataset in datasets.items():
        try:
            log_file = f'{path}/{name}_{args.rewiring}_{args.layer_type}.log'
            setup_logging(log_file)
            log_message(f"Processing dataset: {name} using rewiring method: {args.rewiring} with setting of "
                        f"{args.layer_type}")

            # --------------------
            # ORIGINAL GRAPH METRICS
            # --------------------
            # visualize_graph(dataset.data, dataset.data.num_nodes, f'{name}_original', args.rewiring)
            adj = get_adj(dataset.data.edge_index, set_diag=True, symmetric_normalize=False)
            original_diameter = compute_diameter(dataset.data)
            log_message(f"{name} Original Diameter: {original_diameter}")

            # Compute decay rates and over-squashing metrics for the original graph
            decay_rates_original, decay_rates_original_with_nan = decay_rate(adj, diameter=original_diameter)
            Y_pre_orig, Y_avg_orig, Y_var_orig, Y_skew_orig = compute_over_squashing_metrics(decay_rates_original)
            log_message(f"{name} Original Over-Squashing Prevalence: {Y_pre_orig}")
            log_message(f"{name} Original Average Decay Rate: {Y_avg_orig}")
            log_message(f"{name} Original Variance of Decay Rates: {Y_var_orig}")
            log_message(f"{name} Original Skewness of Decay Rates: {Y_skew_orig}")

            # --------------------
            # REWIRED GRAPH METRICS
            # --------------------
            log_message(f"TESTING rewiring for: {name} ({args.rewiring})")
            # Deepcopy to avoid modifying the original graph

            if args.rewiring == "fosr":
                edge_index, edge_type, _ = fosr.edge_rewire(dataset.data.edge_index.numpy(),
                                                            num_iterations=args.num_iterations)
                dataset.data.edge_index = torch.tensor(edge_index)
                dataset.data.edge_type = torch.tensor(edge_type)
            elif args.rewiring == "sdrf":
                dataset.data.edge_index, dataset.data.edge_type = sdrf.sdrf(dataset.data,
                                                                            loops=args.num_iterations,
                                                                            remove_edges=False,
                                                                            is_undirected=True)
            elif args.rewiring == "digl":
                dataset.data.edge_index = digl.rewire(dataset.data, alpha=args.alpha, k=args.k, eps=args.eps)
                m = dataset.data.edge_index.shape[1]
                dataset.data.edge_type = torch.tensor(np.zeros(m, dtype=np.int64))

            elif args.rewiring == "borf":
                # log_message(f"BORF hyper-parameter : num_iterations = {args.num_iterations}")
                # log_message(f"BORF hyper-parameter : batch_add = {args.borf_batch_add}")
                # log_message(f"BORF hyper-parameter : batch_remove = {args.borf_batch_remove}")
                edge_index, edge_type = borf.borf3(dataset.data, loops=args.num_iterations, remove_edges=True,
                                                   is_undirected=True,
                                                   batch_add=args.borf_batch_add, batch_remove=args.borf_batch_remove,
                                                   dataset_name=name, graph_index=0)
                dataset.data.edge_index = edge_index.clone().detach()
                dataset.data.edge_type = edge_type.clone().detach()

            else:
                log_message("No rewiring method specified. Skipping rewiring.")

            log_message(f"Processing Rewired Graph: {name}")
            # visualize_graph(dataset.data, dataset.data.num_nodes, f'{name}_rewired', args.rewiring)
            rewired_adj = get_adj(dataset.data.edge_index, set_diag=True, symmetric_normalize=False)
            rewired_diameter = compute_diameter(dataset.data)
            log_message(f"{name} Rewired Diameter: {rewired_diameter}")

            # Compute decay rates and over-squashing metrics for the rewired graph
            decay_rates_rewired, decay_rates_rewired_with_nan = decay_rate(rewired_adj, diameter=rewired_diameter)
            Y_pre_rew, Y_avg_rew, Y_var_rew, Y_skew_rew = compute_over_squashing_metrics(decay_rates_rewired)
            log_message(f"{name} Rewired Over-Squashing Prevalence: {Y_pre_rew}")
            log_message(f"{name} Rewired Average Decay Rate: {Y_avg_rew}")
            log_message(f"{name} Rewired Variance of Decay Rates: {Y_var_rew}")
            log_message(f"{name} Rewired Skewness of Decay Rates: {Y_skew_rew}")

            # --------------------
            # COMPUTE INDIVIDUAL TREATMENT EFFECTS (ITEs)
            # --------------------
            ITE_pre = Y_pre_rew - Y_pre_orig
            ITE_avg = Y_avg_rew - Y_avg_orig
            ITE_var = Y_var_rew - Y_var_orig
            ITE_skew = Y_skew_rew - Y_skew_orig

            # log_message(f"{name} ITE Prevalence: {ITE_pre:.3f}")
            # log_message(f"{name} ITE Average: {ITE_avg:.3f}")
            # log_message(f"{name} ITE Variance: {ITE_var:.3f}")
            # log_message(f"{name} ITE Skewness: {ITE_skew:.3f}")
            # log_message('=' * 40)

            # --------------------
            # SIGNIFICANCE TESTING FOR ITEs
            # --------------------
            # Convert decay rates to numpy arrays for statistical tests
            decay_original = np.array(decay_rates_original_with_nan)
            decay_rewired = np.array(decay_rates_rewired_with_nan)

            valid_idx = ~np.isnan(decay_original) & ~np.isnan(decay_rewired)
            decay_original = decay_original[valid_idx]
            decay_rewired = decay_rewired[valid_idx]

            if len(decay_original) > 0:
                t_stat, p_value_avg = ttest_rel(decay_rewired, decay_original)
                # print(f"T-statistic: {t_stat:.3f}, P-value: {p_value_avg:.5f}")
            else:
                print("No valid pairs available for the t-test.")

            # McNemar's test for ITE_pre
            positive_original = decay_original > 0
            positive_rewired = decay_rewired > 0
            only_original_positive = np.sum(positive_original & ~positive_rewired)
            only_rewired_positive = np.sum(~positive_original & positive_rewired)
            n12 = only_original_positive
            n21 = only_rewired_positive
            if n12 + n21 > 0:
                stat = (abs(n12 - n21) - 1) ** 2 / (n12 + n21)
                p_value_pre = 1 - chi2.cdf(stat, 1)
            else:
                p_value_pre = 1.0

            # Step 4: Apply Bonferroni correction
            # Since we're testing two metrics (ITE_avg and ITE_pre), adjust alpha
            alpha = 0.05
            alpha_corrected = alpha / 2  # 0.025

            # Determine significance
            sig_avg = "significant" if p_value_avg < alpha_corrected else "not significant"
            sig_pre = "significant" if p_value_pre < alpha_corrected else "not significant"

            # Step 5: Log the results with significance indicators
            log_message(f"{name} ITE Prevalence: {ITE_pre}, p-value: {p_value_pre:.5f} ({sig_pre})")
            log_message(f"{name} ITE Average: {ITE_avg}, p-value: {p_value_avg:.5f} ({sig_avg})")
            log_message(f"{name} ITE Variance: {ITE_var}")
            log_message(f"{name} ITE Skewness: {ITE_skew}")

        except Exception as e:
            logging.error(f"Error processing {name}: {str(e)}")
            continue


def dataset_statistics():
    args = get_args()
    dataset = get_dataset(args)

    num_nodes_list = []
    degree_list = []
    diameter_list = []
    num_edges_list = []
    counter_of_connected = 0
    for data in tqdm(dataset, desc='Processing', total=len(dataset)):
        num_nodes = data.num_nodes
        num_edges = data.edge_index.shape[1] // 2
        max_deg = degree(data.edge_index[0], num_nodes, dtype=torch.long).max()
        diameter = compute_diameter(data)
        num_nodes_list.append(num_nodes)
        num_edges_list.append(num_edges)
        degree_list.append(max_deg)
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

    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of disconnected graphs: {len(dataset) - counter_of_connected}')


def main():
    args = get_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'logs_stats')
    # if not osp.exists(path):
    #     os.makedirs(path)
    # log_file = f'{path}/{args.dataset}.log'
    # setup_logging(log_file)
    # test_and_plot_toy_datasets(seed)
    # test_and_plot_real_datasets_list()
    # graph_level_average_toy()
    # node_classification_datasets()
    # graph_level_average()
    dataset_statistics()


if __name__ == "__main__":
    main()

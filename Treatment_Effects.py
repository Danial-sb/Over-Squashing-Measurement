from typing import List
from typing import Tuple
from typing import Dict
from typing import Any
from scipy import stats
from torch_geometric.datasets import WebKB, Planetoid, WikipediaNetwork
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import Compose, LargestConnectedComponents
import torch
from utils import *
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, chi2
from tqdm import tqdm
import logging
import os
import random
from args import get_args
from torch_geometric.datasets import TUDataset
import os.path as osp
import numpy as np
from Measurment import decay_rate
from preprocessing import digl, sdrf, fosr, borf

def compute_over_squashing_metrics(decay_rates: List[float], num_nodes) -> Tuple[float, float, float, float]:
    """
    Compute over-squashing metrics:
      1. Over-Squashing Prevalence (Y_pre): fraction of node pairs with positive decay rates.
      2. Average Decay Rate (Y_avg): average decay rate over all node pairs.
      3. Variance of Decay Rates (Y_var): variability of the decay rates.
      4. Maximum Decay Rate (Y_max): highest observed decay rate.

    Parameters:
        decay_rates (List[float]): List of decay rates k_{vu} for each node pair.

    Returns:
        Tuple[float, float, float, float]: (Y_pre, Y_avg, Y_var, Y_max)
    """
    if not decay_rates:
        return 0.0, 0.0, 0.0, 0.0

    total_pairs = num_nodes * num_nodes
    if total_pairs == 0:
        return 0.0, 0.0, 0.0, 0.0

    # 1. Prevalence: fraction of pairs with a positive decay rate.
    num_positive = sum(1 for rate in decay_rates if rate > 0)
    Y_pre = num_positive / total_pairs

    # Filter the positive ones only
    positive_decay_rates = [rate for rate in decay_rates if rate > 0]
    N = len(positive_decay_rates)

    # 2. Average decay rate.
    Y_avg = sum(positive_decay_rates) / N

    # 3. Variance of decay rates.
    Y_var = sum((rate - Y_avg) ** 2 for rate in positive_decay_rates) / N

    # 4. Maximum decay rate.
    Y_max = max(positive_decay_rates)

    return Y_pre, Y_avg, Y_var, Y_max

def load_datasets():
    combined_transform = Compose([AddSelfLoopsTransform(), CustomTransform()])
    datasets = {
        "mutag": list(TUDataset(root="data", name="MUTAG", transform=AddSelfLoopsTransform())),
        "enzymes": list(TUDataset(root="data", name="ENZYMES", transform=AddSelfLoopsTransform())),
        "proteins": list(TUDataset(root="data", name="PROTEINS", transform=AddSelfLoopsTransform())),
        "imdb_binary": list(TUDataset(root="data", name="IMDB-BINARY", transform=combined_transform)),
        "collab": list(TUDataset(root="data", name="COLLAB", transform=combined_transform)),
        "reddit_binary": list(TUDataset(root="data", name="REDDIT-BINARY", transform=combined_transform)),
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
                edge_index, edge_type = borf.borf3(data, loops=args.num_iterations, remove_edges=True,
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
                    max_list = List[float],
                    alpha: float = 0.05) -> Dict[str, Tuple[float, float]]:
    """
    Perform one-sample t-tests on the lists of individual treatment effects (ITEs) for each metric.

    Parameters:
        pre_list, ave_list, var_list, max_list (List[float]): Lists of ITEs.
        alpha (float): Overall significance level.

    Returns:
        Dict[str, Tuple[float, float]]: Dictionary with metric names as keys and (t_stat, p_value) as values.
    """
    results = {}
    tests = {
        'prevalence': pre_list,
        'average': ave_list,
        'variance': var_list,
        'max': max_list
    }
    # Bonferroni correction for multiple comparisons.
    corrected_alpha = alpha / len(tests)

    for metric, values in tests.items():
        t_stat, p_val = stats.ttest_1samp(values, 0)
        results[metric] = (t_stat, p_val, corrected_alpha)
    return results

def compute_metrics_for_dataset(dataset: List[Any], name: str, args: Any, device: torch.device,
                                no_rewiring: bool = False):
    """
    Compute individual treatment effects (ITEs) for over-squashing metrics for each graph in the dataset.

    Parameters:
        dataset (List[Any]): List of graph data objects.
        name (str): Name of the dataset.
        args (Any): Arguments with hyperparameters.
        device (torch.device): Computation device.
        no_rewiring (bool): If True, only compute metrics for the original graphs without rewiring.

    Returns:
        Tuple: (ITE metric lists, t-test results)
    """
    ITE_pre_list, ITE_ave_list, ITE_var_list, ITE_max_list = [], [], [], []
    Y_pre_list, Y_ave_list, Y_var_list, Y_max_list = [], [], [], []
    for idx, data in enumerate(tqdm(dataset, desc="Processing decay rates")):
        # Compute original graph metrics.
        try:
            adj = get_adj(data.edge_index, set_diag=False, symmetric_normalize=False, device=device)
            diameter = compute_diameter(data)
            decay_vals, _, num_nodes = decay_rate(adj, diameter=diameter)
            if len(decay_vals) == 0:
                print(f'No decay rate in graph {idx}')
                continue
            Y_pre, Y_avg, Y_var, Y_max = compute_over_squashing_metrics(decay_vals, num_nodes)
            if no_rewiring:
                Y_pre_list.append(Y_pre)
                Y_ave_list.append(Y_avg)
                Y_var_list.append(Y_var)
                Y_max_list.append(Y_max)
                continue

            # Compute rewired graph metrics.
            rewired_data = apply_rewiring([data], args.rewiring, args, name)[0]
            rewired_adj = get_adj(rewired_data.edge_index, set_diag=False, symmetric_normalize=False, device=device)
            rewired_diameter = compute_diameter(rewired_data)
            decay_vals_r, _, num_nodes_r = decay_rate(rewired_adj, diameter=rewired_diameter)
            Y_pre_r, Y_avg_r, Y_var_r, Y_max_r = compute_over_squashing_metrics(decay_vals_r, num_nodes_r)

            # Compute individual treatment effects (ITEs).
            ITE_pre_list.append(Y_pre_r - Y_pre)
            ITE_ave_list.append(Y_avg_r - Y_avg)
            ITE_var_list.append(Y_var_r - Y_var)
            ITE_max_list.append(Y_max_r - Y_max)
        except Exception as e:
            log_message(f"Error processing graph index {idx}: {e}")
            continue

    if no_rewiring:
        return Y_pre_list, Y_ave_list, Y_var_list, Y_max_list

    t_test_results = perform_t_tests(ITE_pre_list, ITE_ave_list, ITE_var_list, ITE_max_list)
    return (ITE_pre_list, ITE_ave_list, ITE_var_list, ITE_max_list), t_test_results


def graph_level_average(no_rewiring: bool = False):
    """
    Computes and logs graph-level over-squashing metrics or treatment effects across multiple datasets.

    Depending on the `no_rewiring` flag, this function either:
    - Computes the baseline over-squashing metrics (prevalence, average, variance, and maximum) on the original graphs, or
    - Computes the average treatment effect (ATE) of a specified rewiring method, including statistical tests to assess significance.

    Args:
        no_rewiring (bool): If True, runs the analysis on the original graphs without any rewiring applied.
                            If False, evaluates the impact of rewiring as specified in the command-line arguments.
    """
    args = get_args()
    datasets = load_datasets()
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'decay_rate_logs')
    os.makedirs(path, exist_ok=True)
    device = torch.device('cpu')

    for name, dataset in datasets.items():
        setup_logging(f'{path}/{name}_{args.rewiring}_{args.layer_type}.log')

        if no_rewiring:
            Y_pre_list, Y_ave_list, Y_var_list, Y_max_list = compute_metrics_for_dataset(dataset, name, args, device)
            Y_pre_mean = torch.mean(torch.tensor(Y_pre_list)).item()
            Y_ave_mean = torch.mean(torch.tensor(Y_ave_list)).item()
            Y_var_mean = torch.mean(torch.tensor(Y_var_list)).item()
            Y_max_mean = torch.mean(torch.tensor(Y_max_list)).item()
            log_message(f"Original Graph Stats:")
            log_message(f"Prevalence: {Y_pre_mean}")
            log_message(f"Average: {Y_ave_mean}")
            log_message(f"Variance: {Y_var_mean}")
            log_message(f"Maximum: {Y_max_mean}")
        else:
            ite_lists, t_test_results = compute_metrics_for_dataset(dataset, name, args, device)
            ITE_pre_list, ITE_ave_list, ITE_var_list, ITE_max_list = ite_lists

            ate_pre = torch.mean(torch.tensor(ITE_pre_list)).item()
            ate_ave = torch.mean(torch.tensor(ITE_ave_list)).item()
            ate_var = torch.mean(torch.tensor(ITE_var_list)).item()
            ate_max = torch.mean(torch.tensor(ITE_max_list)).item()

            for metric, (t_stat, p_val, corr_alpha) in t_test_results.items():
                ate = {'prevalence': ate_pre, 'average': ate_ave, 'variance': ate_var, 'max': ate_max}[metric]
                significance = 'Significant' if p_val < corr_alpha else 'Not Significant'
                log_message(f"{metric.capitalize()} ATE: {ate} (t={t_stat:.2f}, p={p_val:.4f}, {significance})")


def report_average_added_edges():
    """
    Computes and logs the average number of edges added per graph after rewiring for each graph classification dataset.

    For each dataset, rewiring is applied to all graphs, and the difference in edge count before
    and after rewiring is recorded. The average number of added edges is then logged to a file
    in the 'Added_edges/' directory.

    Logs are saved per dataset and rewiring configuration.
    """

    args = get_args()
    datasets = load_datasets()
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'Added_edges')
    os.makedirs(path, exist_ok=True)

    for name, dataset in datasets.items():
        setup_logging(f'{path}/{name}_{args.rewiring}_{args.layer_type}_added_edges.log')
        added_edges_list = []

        for idx, data in enumerate(tqdm(dataset, desc=f"Rewiring {name}")):
            try:
                num_orig_edges = data.edge_index.size(1) // 2
                rewired_data = apply_rewiring([data], args.rewiring, args, name)[0]
                num_rew_edges = rewired_data.edge_index.size(1) // 2
                added_edges = num_rew_edges - num_orig_edges
                added_edges_list.append(added_edges)
            except Exception as e:
                log_message(f"Error processing graph {idx}: {e}")
                continue

        avg_added_edges = torch.mean(torch.tensor(added_edges_list, dtype=torch.float32)).item() if added_edges_list else 0.0
        log_message(f"{avg_added_edges:.2f} edges added on average for dataset {name} under rewiring {args.rewiring} and layer type {args.layer_type}")

def report_average_added_edges_node():
    """
    Computes and logs the average number of edges added per graph after rewiring for each node classification dataset.

    For each dataset, rewiring is applied to all graphs, and the difference in edge count before
    and after rewiring is recorded. The average number of added edges is then logged to a file
    in the 'Added_edges/' directory.

    Logs are saved per dataset and rewiring configuration.
    """
    args = get_args()
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'Added_edges_node')
    os.makedirs(path, exist_ok=True)

    largest_cc = LargestConnectedComponents()

    cornell = WebKB(root="data", name="Cornell", transform=largest_cc)
    wisconsin = WebKB(root="data", name="Wisconsin", transform=largest_cc)
    texas = WebKB(root="data", name="Texas", transform=largest_cc)
    chameleon = WikipediaNetwork(root="data", name="chameleon", transform=largest_cc)
    cora = Planetoid(root="data", name="cora", transform=largest_cc)
    citeseer = Planetoid(root="data", name="citeseer", transform=largest_cc)
    datasets = {"cornell": cornell, "wisconsin": wisconsin, "texas": texas, "chameleon": chameleon, "cora": cora,
                "citeseer": citeseer}
    for name, dataset in datasets.items():
        dataset.data.edge_index = to_undirected(dataset.data.edge_index)

    for name, dataset in datasets.items():
        setup_logging(f'{path}/{name}_{args.rewiring}_{args.layer_type}_added_edges.log')
        num_orig_edges = dataset.data.edge_index.size(1) // 2
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
            edge_index, edge_type = borf.borf3(dataset.data, loops=args.num_iterations, remove_edges=True,
                                               is_undirected=True,
                                               batch_add=args.borf_batch_add, batch_remove=args.borf_batch_remove,
                                               dataset_name=name, graph_index=0)
            dataset.data.edge_index = edge_index.clone().detach()
            dataset.data.edge_type = edge_type.clone().detach()

        else:
            log_message("No rewiring method specified. Skipping rewiring.")
        num_rew_edges = dataset.data.edge_index.size(1) // 2
        added_edges = num_rew_edges - num_orig_edges
        log_message(f"{added_edges:.2f} edges added for dataset {name} under rewiring {args.rewiring} and layer type {args.layer_type}")

def plot_distributions(args):
    """
    Plot the distribution of Y_pre, Y_mean, Y_var, and Y_skew with LaTeX-rendered labels.
    """

    device = torch.device('cpu')
    datasets = load_datasets()
    sns.set_theme(style="white")
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    plt.figure(figsize=(12, 8))

    for name, dataset in datasets.items():
        Y_pre_list, Y_ave_list, Y_var_list, Y_max_list = compute_metrics_for_dataset(dataset, name, args,
                                                                                      device=device)
        metrics = {
            r"$\mathbf{Y}_{\textbf{pre}}$": Y_pre_list,
            r"$\mathbf{Y}_{\textbf{mean}}$": Y_ave_list,
            r"$\mathbf{Y}_{\textbf{var}}$": Y_var_list,
            r"$\mathbf{Y}_{\textbf{max}}$": Y_max_list
        }
        for i, (label, values) in enumerate(metrics.items(), 1):
            plt.subplot(2, 2, i)
            sns.histplot(values, kde=True, bins=30)
            plt.xlabel(label, fontsize=14)
            plt.ylabel(r"\text{Density}", fontsize=14)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)

        plt.tight_layout()

        path = osp.join(osp.dirname(osp.realpath(__file__)), 'Y_distributions')
        os.makedirs(path, exist_ok=True)
        plt.savefig(f'{path}/{name}.pdf', dpi=300, bbox_inches='tight')
        plt.show()
        plt.clf()

def node_classification_treatment_effects(rewiring : bool = True):
    """
    Computes over-squashing metrics and treatment effects for node classification datasets.

    For each dataset, the function logs decay-based metrics (prevalence, average, variance, maximum)
    on the original graph and, optionally, on its rewired version. It also computes individual
    treatment effects (ITEs) and performs statistical significance tests (t-test, McNemar's test)
    to evaluate rewiring impact.

    Args:
        rewiring (bool): If True, applies the selected rewiring method before computing treatment effects.
    """

    largest_cc = LargestConnectedComponents()

    cornell = WebKB(root="data", name="Cornell", transform=largest_cc)
    wisconsin = WebKB(root="data", name="Wisconsin", transform=largest_cc)
    texas = WebKB(root="data", name="Texas", transform=largest_cc)
    chameleon = WikipediaNetwork(root="data", name="chameleon", transform=largest_cc)
    cora = Planetoid(root="data", name="cora", transform=largest_cc)
    citeseer = Planetoid(root="data", name="citeseer", transform=largest_cc)
    datasets = {"cornell": cornell, "wisconsin": wisconsin, "texas": texas, "chameleon": chameleon, "cora": cora,
                "citeseer": citeseer}

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
            adj = get_adj(dataset.data.edge_index, set_diag=True, symmetric_normalize=False)
            original_diameter = compute_diameter(dataset.data)
            orig_edges = dataset.data.edge_index.size(1) // 2
            log_message(f"{name} Original Diameter: {original_diameter}")

            # Compute decay rates and over-squashing metrics for the original graph
            decay_rates_original, decay_rates_original_with_nan, num_nodes = decay_rate(adj, diameter=original_diameter)
            Y_pre_orig, Y_avg_orig, Y_var_orig, Y_max_orig = compute_over_squashing_metrics(decay_rates_original, num_nodes)
            log_message(f"{name} Original Over-Squashing Prevalence: {Y_pre_orig}")
            log_message(f"{name} Original Average Decay Rate: {Y_avg_orig}")
            log_message(f"{name} Original Variance of Decay Rates: {Y_var_orig}")
            log_message(f"{name} Original Maximum Decay Rate: {Y_max_orig}")
            if rewiring:
                # --------------------
                # REWIRED GRAPH METRICS
                # --------------------
                log_message(f"TESTING rewiring for: {name} ({args.rewiring})")

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
                    edge_index, edge_type = borf.borf3(dataset.data, loops=args.num_iterations, remove_edges=True,
                                                       is_undirected=True,
                                                       batch_add=args.borf_batch_add, batch_remove=args.borf_batch_remove,
                                                       dataset_name=name, graph_index=0)
                    dataset.data.edge_index = edge_index.clone().detach()
                    dataset.data.edge_type = edge_type.clone().detach()

                else:
                    log_message("No rewiring method specified. Skipping rewiring.")

                log_message(f"Processing Rewired Graph: {name}")
                rewired_adj = get_adj(dataset.data.edge_index, set_diag=True, symmetric_normalize=False)
                rewired_diameter = compute_diameter(dataset.data)
                rew_edges = dataset.data.edge_index.size(1) // 2
                log_message(f"{name} Rewired Diameter: {rewired_diameter}")

                # Compute decay rates and over-squashing metrics for the rewired graph
                decay_rates_rewired, decay_rates_rewired_with_nan, num_nodes = decay_rate(rewired_adj, diameter=rewired_diameter)
                Y_pre_rew, Y_avg_rew, Y_var_rew, Y_max_rew = compute_over_squashing_metrics(decay_rates_rewired, num_nodes)
                log_message(f"{name} Rewired Over-Squashing Prevalence: {Y_pre_rew}")
                log_message(f"{name} Rewired Average Decay Rate: {Y_avg_rew}")
                log_message(f"{name} Rewired Variance of Decay Rates: {Y_var_rew}")
                log_message(f"{name} Rewired Maximum Decay Rate: {Y_max_rew}")

                # --------------------
                # COMPUTE INDIVIDUAL TREATMENT EFFECTS (ITEs)
                # --------------------
                ITE_pre = Y_pre_rew - Y_pre_orig
                ITE_avg = Y_avg_rew - Y_avg_orig
                ITE_var = Y_var_rew - Y_var_orig
                ITE_max = Y_max_rew - Y_max_orig

                # --------------------
                # SIGNIFICANCE TESTING FOR ITEs
                # --------------------
                # Convert decay rates to numpy arrays for statistical tests
                decay_original = np.array(decay_rates_original_with_nan)
                decay_rewired = np.array(decay_rates_rewired_with_nan)

                valid_idx = ~np.isnan(decay_original) & ~np.isnan(decay_rewired)
                decay_original = decay_original[valid_idx]
                decay_rewired = decay_rewired[valid_idx]

                both_positive = (decay_original > 0) & (decay_rewired > 0)
                decay_orig_pos = decay_original[both_positive]
                decay_rew_pos = decay_rewired[both_positive]

                eps = 1e-12
                log_orig = np.log(decay_orig_pos + eps)
                log_rew = np.log(decay_rew_pos + eps)
                if len(log_orig) > 0:
                    t_stat, p_value_avg = ttest_rel(log_rew, log_orig)
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
                log_message(f"{name} ITE Maximum: {ITE_max}")
                log_message(f"Number of added edges: {rew_edges - orig_edges}")


        except Exception as e:
            logging.error(f"Error processing {name}: {str(e)}")
            continue


def main():
    args = get_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'decay_rate_logs')
    # if not osp.exists(path):
    #     os.makedirs(path)
    # log_file = f'{path}/{args.dataset}/{args.rewiring}/{args.layer_type}.log'
    # setup_logging(log_file)
    # test_and_plot_toy_datasets(seed)
    # test_and_plot_real_datasets_list()
    # node_classification_datasets()
    graph_level_average()
    # report_average_added_edges()
    # report_average_added_edges_node()
    # dataset_statistics()
    # plot_distributions(args)
    # graph_level_average()
    # graph_level_average_toy()
    # check_disconnected_graphs()


if __name__ == "__main__":
    main()

import torch
import os
import os.path as osp
from utils import log_message, setup_logging
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from scipy.stats import spearmanr
import seaborn as sns

def latex_installed():
    try:
        subprocess.run(["latex", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception:
        return False

# Configure matplotlib to use LaTeX if available; otherwise, use built-in mathtext.
if latex_installed():
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
else:
    print("LaTeX is not installed. Falling back to built-in mathtext rendering.")
    plt.rcParams['text.usetex'] = False

# Function to compute mean and standard deviation
def compute_stats(values):
    # Convert values to a tensor, ignoring None values
    valid_values = torch.tensor([v for v in values if v is not None], dtype=torch.float32)
    if valid_values.numel() == 0:  # If Tensor is empty
        return 0.0, 0.0
    mean = torch.mean(valid_values)
    std = torch.std(valid_values) if valid_values.numel() > 1 else torch.tensor(0.0)
    return mean.item(), std.item()

def stats(data):
    results = {}
    for dataset in data:
        results[dataset] = {}
        for rewiring in data[dataset]:
            results[dataset][rewiring] = {}
            for metric in data[dataset][rewiring]:
                mean, std = compute_stats(data[dataset][rewiring][metric])
                if metric == "Gain":
                    mean_str = f"{mean:.1f}"
                    std_str = f"{std:.1f}"
                else:
                    mean_str = f"{mean:.2e}"
                    std_str = f"{std:.2e}"
                results[dataset][rewiring][metric] = f"{mean_str} ± {std_str}"

    # Print results for all datasets
    for dataset in results:
        log_message(f"\n{dataset} Dataset Results:")
        for rewiring in results[dataset]:
            log_message(f"\n{rewiring}:")
            for metric in results[dataset][rewiring]:
                log_message(f"  {metric}: {results[dataset][rewiring][metric]}")

def plot_scatter(data):
    # Define the rewiring types and the ATE metrics we are interested in
    rewiring_types = ["FoSR", "DIGL", "SDRF", "BORF", "GTR"]
    ate_metrics = ["ATE_pre", "ATE_mean", "ATE_std", "ATE_max"]
    # ate_metrics = ["ITE_pre", "ITE_mean", "ITE_std", "ITE_max"]

    # Mapping of ATE metric names to LaTeX-formatted labels
    latex_labels = {
        "ATE_pre": r"$\textbf{ATE Prevalence}$",
        "ATE_mean": r"$\textbf{ATE Intensity}$",
        "ATE_std": r"$\textbf{ATE Variability}$",
        "ATE_max": r"$\textbf{ATE Extremity}$",
    }
    # latex_labels = {
    #     "ITE_pre": r"$\textbf{ITE Prevalence}$",
    #     "ITE_mean": r"$\textbf{ITE Intensity}$",
    #     "ITE_std": r"$\textbf{ITE Variability}$",
    #     "ITE_max": r"$\textbf{ITE Extremity}$",
    # }

    # Gather data over datasets for each rewiring type and each ATE metric
    plot_data = {metric: {rew: [] for rew in rewiring_types} for metric in ate_metrics}

    for dataset in data.values():
        for rew in rewiring_types:
            if rew in dataset:
                for metric in ate_metrics:
                    if metric in dataset[rew]:
                        values = dataset[rew][metric]
                        plot_data[metric][rew].extend(values)

    # Create a separate plot for each ATE metric
    for metric in ate_metrics:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        ax.grid(False)
        for j, rew in enumerate(rewiring_types):
            # y_vals = plot_data[metric][rew]
            y_vals = [val for val in plot_data[metric][rew] if val is not None]
            # x values: centered at j with some jitter for clarity
            x_vals = np.ones(len(y_vals)) * j + np.random.uniform(-0.1, 0.1, size=len(y_vals))
            ax.scatter(x_vals, y_vals, s=50, alpha=0.7, label=r"$\mathrm{" + rew + r"}$")
        # Set x-axis ticks and labels using LaTeX formatting
        ax.set_xticks(range(len(rewiring_types)))
        ax.set_xticklabels([r"$\mathrm{\textbf{" + rew + r"}}$" for rew in rewiring_types], fontsize=20)
        # ax.set_title(latex_labels[metric], fontsize=14)
        # ax.set_xlabel(r"$\text{Rewiring Type}$", fontsize=12)
        ax.set_ylabel(latex_labels[metric], fontsize=18)

        for label in ax.get_yticklabels():
            label.set_fontsize(18)
            label.set_fontweight('bold')

        # ax.legend()
        fig.tight_layout()
        # Save each figure separately
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'scatter_plots')
        os.makedirs(path, exist_ok=True)
        fig.savefig(f"{path}/scatter_graph_plot_{metric}.pdf", dpi=300)
        plt.show()

def correlation_based_on_rewiring(data):
    """
    Perform Spearman correlation analysis between each ATE type and Gain for each rewiring technique.

    Parameters:
    - data: A dictionary containing datasets, rewiring techniques, ATE types, and Gain values.

    The function iterates over each ATE type and rewiring technique, collects the corresponding
    ATE and Gain values across all datasets, and computes the Spearman correlation if there are
    at least two valid data pairs. It applies the Bonferroni correction for multiple testing based
    on the total number of intended tests (16).
    """
    # Define ATE types and rewiring techniques
    ate_types = ["ATE_pre", "ATE_mean", "ATE_std", "ATE_max"]
    # ate_types = ["ITE_pre", "ITE_mean", "ITE_std", "ITE_max"]
    rewiring_techniques = ["FoSR", "DIGL", "SDRF", "BORF", "GTR"]

    # Define significance levels
    alpha = 0.05
    m = len(ate_types) * len(rewiring_techniques)  # Total number of tests = 4
    alpha_adj = alpha / m

    # Inform the user about the significance levels
    log_message(f"Using alpha={alpha} for individual tests")
    log_message(f"Using Bonferroni corrected alpha={alpha_adj:.6f} for multiple testing correction")

    # Iterate over each ATE type
    for ate_type in ate_types:
        log_message(f"\n### Correlation Analysis for {ate_type}")

        # Iterate over each rewiring technique
        for rewiring_technique in rewiring_techniques:
            ate_values = []
            gain_values = []

            # Collect data across datasets
            for dataset in data:
                if rewiring_technique in data[dataset]:
                    ate_list = data[dataset][rewiring_technique][ate_type]
                    gain_list = data[dataset][rewiring_technique]["Gain"]

                    # Collect pairs where ATE is not None
                    for ate, gain in zip(ate_list, gain_list):
                        if ate is not None:
                            ate_values.append(ate)
                            gain_values.append(gain)

            log_message(f"\nRewiring Technique: {rewiring_technique}")
            if len(ate_values) >= 2:
                # Compute Spearman correlation
                corr, p_value = spearmanr(ate_values, gain_values)
                log_message(f"Spearman Correlation: {corr:.6f}")
                log_message(f"P-value: {p_value:.6f}")
                log_message(f"Number of samples: {len(ate_values)}")

                # Determine significance
                significance = "Yes" if p_value <= alpha else "No"
                significance_bonf = "Yes" if p_value <= alpha_adj else "No"
                log_message(f"Significant (alpha={alpha}): {significance}")
                log_message(f"Significant (Bonferroni, alpha={alpha_adj:.6f}): {significance_bonf}")
            else:
                log_message(f"Not enough data (samples: {len(ate_values)})")
            log_message("-" * 50)

def visualize_correlations(alpha=0.05, m=4,
                           save_path="correlation_node_heatmap.pdf"):
    """
    Visualize Spearman correlation coefficients between ATE types and Gain for different rewiring techniques using a heatmap.
    Significant correlations (after Bonferroni correction) are marked with an asterisk (*).

    Parameters:
    - results: A dictionary with correlation and p-value for each ATE type and rewiring technique.
    - ate_types: List of ATE types (e.g., ["ATE_pre", "ATE_mean", "ATE_std", "ATE_max"]).
    - rewiring_techniques: List of rewiring techniques (e.g., ["FoSR", "DIGL", "SDRF", "BORF"]).
    - alpha: Original significance level (default: 0.05).
    - m: Number of tests for Bonferroni correction (default: 16).
    - save_path: Path to save the heatmap figure (default: "correlation_heatmap.png").
    """
    # Compute Bonferroni-corrected alpha
    alpha_adj = alpha / m
    ate_types = ["ITE_pre", "ITE_mean", "ITE_std", "ITE_max"]
    latex_labels = {
        "ITE_pre": r"$\textbf{Pre.}$",
        "ITE_mean": r"$\textbf{Int.}$",
        "ITE_std": r"$\textbf{Var.}$",
        "ITE_max": r"$\textbf{Ext.}$"
    }
    rewiring_techniques = ["FoSR", "DIGL", "SDRF", "BORF"]
    rewiring_latex = {
        "FoSR": r"$\textbf{FoSR}$",
        "DIGL": r"$\textbf{DIGL}$",
        "SDRF": r"$\textbf{SDRF}$",
        "BORF": r"$\textbf{BORF}$"
    }

# if you want generate graph stats uncomment the following lines and comment out the previous ones.
    # ate_types = ["ATE_pre", "ATE_mean", "ATE_std", "ATE_max"]
    # latex_labels = {
    #     "ATE_pre": r"$\textbf{Pre.}$",
    #     "ATE_mean": r"$\textbf{Int.}$",
    #     "ATE_std": r"$\textbf{Var.}$",
    #     "ATE_max": r"$\textbf{Ext.}$",
    # }
    # rewiring_techniques = ["FoSR", "DIGL", "SDRF", "BORF", "GTR"]
    # rewiring_latex = {
    #     "FoSR": r"$\textbf{FoSR}$",
    #     "DIGL": r"$\textbf{DIGL}$",
    #     "SDRF": r"$\textbf{SDRF}$",
    #     "BORF": r"$\textbf{BORF}$",
    #     "GTR": r"$\textbf{GTR}$"
    # }
    results = {
        "ATE_pre": {
            "FoSR": {"correlation": -0.485947, "p_value": 0.057293},
            "DIGL": {"correlation": -0.143138, "p_value": 0.504615},
            "SDRF": {"correlation": +0.012413, "p_value": 0.755097},
            "BORF": {"correlation": -0.576600, "p_value": 0.175382},
            "GTR": {"correlation": -0.408844, "p_value": 0.047293},
        },
        "ATE_mean": {
            "FoSR": {"correlation": -0.381947, "p_value": 0.884946}, #ops
            "DIGL": {"correlation": +0.044522, "p_value": 0.836344},
            "SDRF": {"correlation": -0.117493, "p_value": 0.584538},
            "BORF": {"correlation": -0.290914, "p_value": 0.484530},
            "GTR": {"correlation": 0.164017, "p_value": 0.443779},
        },
        "ATE_std": {
            "FoSR": {"correlation": -0.340088, "p_value": 0.103943},
            "DIGL": {"correlation": +0.211977, "p_value": 0.343615},
            "SDRF": {"correlation": -0.098641, "p_value": 0.646546},
            "BORF": {"correlation": -0.181822, "p_value": 0.666524},
            "GTR": {"correlation": 0.043346, "p_value": 0.840614},
        },
        "ATE_max": {
            "FoSR": {"correlation": -0.253813, "p_value": 0.231397},
            "DIGL": {"correlation": +0.205107, "p_value": 0.336327},
            "SDRF": {"correlation": +0.013152, "p_value": 0.951363}, #ops
            "BORF": {"correlation": -0.132540, "p_value": 0.754383},
            "GTR": {"correlation": 0.048138, "p_value": 0.823251}
        },
    }

    node_results = {
        "ITE_pre": {
            "FoSR": {"correlation": -0.238793, "p_value": 0.424158},
            "DIGL": {"correlation": +0.371429, "p_value": 0.622787},
            "SDRF": {"correlation": -0.366510, "p_value": 0.344234},
            "BORF": {"correlation": -0.648485, "p_value": 0.036806},
        },
        "ITE_mean": {
            "FoSR": {"correlation": -0.105264, "p_value": 0.959997},
            "DIGL": {"correlation": +0.942857, "p_value": 0.000309},
            "SDRF": {"correlation": -0.366510, "p_value": 0.344234},
            "BORF": {"correlation": +0.608301, "p_value": 0.047024},
        },
        "ITE_std": {
            "FoSR": {"correlation": -0.194036, "p_value": 0.547886},
            "DIGL": {"correlation": +0.828571, "p_value": 0.004805},
            "SDRF": {"correlation": -0.365161, "p_value": 0.185026},
            "BORF": {"correlation": +0.175692, "p_value": 0.519302},
        },
        "ITE_max": {
            "FoSR": {"correlation": +0.189475, "p_value": 0.375946},
            "DIGL": {"correlation": +0.942857, "p_value": 0.004805},
            "SDRF": {"correlation": -0.366510, "p_value": 0.344234},
            "BORF": {"correlation": +0.180702, "p_value": 0.155625},
        },
    }

    n_ate = len(ate_types)
    n_rew = len(rewiring_techniques)
    correlation_matrix = np.zeros((n_ate, n_rew))
    p_value_matrix = np.zeros((n_ate, n_rew))

    # Fill matrices with data from results
    for i, ate in enumerate(ate_types):
        for j, rew in enumerate(rewiring_techniques):
            correlation_matrix[i, j] = node_results[ate][rew]["correlation"]
            p_value_matrix[i, j] = node_results[ate][rew]["p_value"]

    # Determine significance after Bonferroni correction
    significance_matrix = p_value_matrix < alpha_adj

    # Create annotations with correlation values and asterisks for significant results
    annotations = [
        [fr"$\mathbf{{{correlation_matrix[i, j]:+.2f}}}^{{*}}$" if significance_matrix[i, j]
         else fr"$\mathbf{{{correlation_matrix[i, j]:+.2f}}}$"
         for j in range(n_rew)] for i in range(n_ate)
    ]

    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    tick_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    ax = sns.heatmap(
        correlation_matrix,
        annot=annotations,
        fmt="",  # Use pre-formatted strings
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=[rewiring_latex[rew] for rew in rewiring_techniques],
        yticklabels=[latex_labels[ate] for ate in ate_types],
        cbar_kws={
            'ticks': tick_vals,  # <‑‑ here
            # 'label': r'\textbf{Spearman Correlation}',  # optional label
        },
        annot_kws = {"size": 52}
    )
    # plt.title(
    #     rf"Correlation between ATE Types and Gain by Rewiring Technique\\Significant correlations
    #     (Bonferroni, $\alpha={alpha_adj:.4f}$) marked with *")

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=46, fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=46, fontweight='bold')
    cbar = ax.collections[0].colorbar

    # Update the colorbar label: bigger and bold
    # cbar.set_label(r'\textbf{Spearman Correlation}', fontsize=32, fontweight='bold')

    from matplotlib.ticker import FuncFormatter

    def format_ticks_with_sign(x, _):
        return f"${x:+.1f}$"  # Uses LaTeX math mode

    cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_ticks_with_sign))

    # Set bold font for tick labels (LaTeX math still used)
    for label in cbar.ax.get_yticklabels():
        label.set_fontsize(44)
        label.set_fontweight('bold')

    plt.tight_layout()

    # Save the figure with high resolution (300 DPI)
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'Correlation_Figures')
    os.makedirs(path, exist_ok=True)
    plt.savefig(f'{path}/{save_path}', dpi=300)
    plt.show()


if __name__ == "__main__":
    data = {
        "MUTAG": {
            "FoSR": { # GCN, GIN, R-GCN, R-GIN #
                "ATE_pre": [-1.98e-2, -3.29e-2, -1.98e-2, -2.13e-2],
                "ATE_mean": [-1.98e-2, -8.95e-3, -1.98e-2, -1.21e-2],
                "ATE_std": [-3.41e-2, -2.66e-2, -3.41e-2, -1.42e-2],
                "ATE_max": [-5.32e-2, -4.19e-2, -5.32e-2, -8.21e-3],
                "Gain": [7.9, 0.3, 15.2, 3.1]
            },
            "DIGL": {
                "ATE_pre": [-2.20e-1, -5.93e-1, -5.93e-1, -5.85e-1],
                "ATE_mean": [-8.30e-2, -1.09e-1, -1.09e-1, -1.08e-1],
                "ATE_std": [-7.98e-2, -1.06e-1, -1.06e-1, -1.06e-1],
                "ATE_max": [-3.21e-1, -4.54e-1, -4.54e-1, -4.52e-1],
                "Gain": [-0.8, 2.0, 4.1, -2.6]
            },
            "SDRF": {
                "ATE_pre": [-1.02e-2, -1.02e-2, -1.02e-2, -1.02e-2],
                "ATE_mean": [2.28e-3, 2.28e-3, 2.28e-3, 2.28e-3],
                "ATE_std": [2.87e-3, 2.87e-3, 2.87e-3, 2.87e-3],
                "ATE_max": [2.42e-2, 2.42e-2, 2.42e-2, 2.42e-2],
                "Gain": [1.1, -0.7, -2.3, -0.3]
            },
            "BORF": {
                "ATE_pre": [-8.83e-2, -7.10e-2],
                "ATE_mean": [-4.54e-2, -9.85e-3],
                "ATE_std": [-5.00e-2, -2.63e-3],
                "ATE_max": [-2.23e-1, -4.51e-2],
                "Gain": [3.7, 1.2]
            },
            "GTR": {
                "ATE_pre": [-4.29e-2, 4.79e-2, -3.87e-2, 2.83e-2],
                "ATE_mean": [1.16e-1, 8.26e-2, 1.19e-1, 4.07e-2],
                "ATE_std": [2.71e-2, 4.63e-3, 3.13e-2, -1.61e-2], #NS
                "ATE_max": [9.85e-2, 4.56e-2, 1.07e-1, -3.31e-2], #NS
                "Gain": [7.0, -0.1, 16.3, 3.0]
            }
        },
        "PROTEINS": {
            "FoSR": {
                "ATE_pre": [-6.42e-2, -2.34e-2, -9.83e-3, -6.42e-2],
                "ATE_mean": [-4.12e-2, -2.62e-2, -1.65e-2, -4.12e-2],
                "ATE_std": [-5.04e-2, -3.23e-2, -2.37e-2, -5.04e-2],
                "ATE_max": [-1.07e-1, -5.10e-2, -3.68e-2, -1.07e-1],
                "Gain": [2.4, 4.3, 4.2, 4.1]
            },
            "DIGL": {
                "ATE_pre": [-3.09e-1, -2.26e-1, -3.60e-1, -4.81e-1],
                "ATE_mean": [-9.02e-2, -4.93e-2, -1.00e-1, -1.25e-1],
                "ATE_std": [-8.31e-2, -4.19e-2, -9.66e-2, -1.20e-1],
                "ATE_max": [-3.01e-1, -1.28e-1, -3.81e-1, -4.90e-1],
                "Gain": [-0.3, 0.0, -1.3, 0.8]
            },
            "SDRF": {
                "ATE_pre": [-2.72e-2, -2.58e-2, -2.58e-2, -1.40e-2],
                # "ATE_mean": [None, None, None, None],
                "ATE_mean": [-2.19e-4, -3.66e-4, -3.66e-4, -6.18e-4],
                "ATE_std": [5.64e-3, 4.03e-3, 4.03e-3, 8.72e-4],
                "ATE_max": [1.60e-2, 1.21e-2, 1.21e-2, 6.70e-3],
                "Gain": [0.0, -1.0, -0.4, 0.2]
            },
            "BORF": {
                "ATE_pre": [1.06e-2, 1.96e-2],
                "ATE_mean": [-4.39e-3, -8.41e-3],
                "ATE_std": [-1.62e-2, -3.13e-2],
                "ATE_max": [-1.09e-1, -1.54e-1],
                "Gain": [0.4, 0.5]
            },
            "GTR": {
                "ATE_pre": [-3.45e-2, 2.93e-3, -1.63e-3, 2.93e-3],
                "ATE_mean": [-1.17e-2, 2.22e-3, -4.17e-3, 2.22e-3],
                "ATE_std": [-4.97e-2, -3.42e-2, -4.19e-2, -3.42e-2],
                "ATE_max": [-1.38e-1, -6.98e-2, -9.86e-2, -6.98e-2],
                "Gain": [1.6, 2.3, 6.2, 5.1]
            }
        },

        "ENZYMES": {
            "FoSR": {
                "ATE_pre": [-1.41e-3, 9.11e-3, -3.67e-2, -3.67e-2],
                "ATE_mean": [-1.23e-2, -2.91e-3, -3.68e-2, -3.68e-2],
                "ATE_std": [-2.28e-2, -1.27e-2, -5.54e-2, -5.54e-2],
                "ATE_max": [-2.76e-2, -5.52e-3, -1.38e-1, -1.38e-1],
                "Gain": [-2.6, -4.6, 7.0, 6.4]
            },
            "DIGL": {
                "ATE_pre": [-2.67e-1, -1.67e-1, -1.67e-1, -5.22e-1],
                "ATE_mean": [-8.31e-2, -3.06e-2, -3.06e-2, -1.25e-1],
                "ATE_std": [-7.93e-2, -2.65e-2, -2.65e-2, -1.27e-1],
                "ATE_max": [-3.34e-1, -9.10e-2, -9.10e-2, -5.75e-1],
                "Gain": [-0.1, 1.9, -0.3, -1.4]
            },
            "SDRF": {
                "ATE_pre": [-1.34e-2] * 4,
                "ATE_mean": [3.07e-4] * 4, # Not significant
                "ATE_std": [2.41e-3] * 4,
                "ATE_max": [1.27e-2] * 4,
                "Gain": [0.7, 2.0, 4.8, 0.5]
            },
            "BORF": {
                "ATE_pre": [3.85e-2, 3.49e-3],
                "ATE_mean": [-3.39e-2, -3.57e-2],
                "ATE_std": [-2.91e-2, -2.93e-2],
                "ATE_max": [-1.59e-1, -1.33e-1],
                "Gain": [0.2, 1.7]
            },
            "GTR": {
                "ATE_pre": [1.76e-2, 2.39e-2, 9.28e-3, -6.64e-3],
                "ATE_mean": [1.15e-2, 1.28e-2, 1.03e-2, 2.04e-2],
                "ATE_std": [-3.61e-2, -2.82e-2, -3.57e-2, -2.72e-2],
                "ATE_max": [-9.60e-2, -6.42e-2, -9.95e-2, -6.05e-2],
                "Gain": [-0.1, -3.2, 12.7, 11.0]
            }
        },
        "IMDB": {
            "FoSR": {
                "ATE_pre": [-4.12e-3, -1.43e-2, -1.43e-2, -1.43e-2],
                "ATE_mean": [-6.31e-2, -1.57e-1, -1.57e-1, -1.57e-1],
                # "ATE_std": [None, None, None, None],
                "ATE_std": [-5.04e-4, -2.86e-2, -2.86e-2, -2.86e-2], #Not significant
                "ATE_max": [-4.49e-2, -3.92e-3, -3.92e-3, -3.92e-3],
                "Gain": [-0.1, 1.1, 14.0, 1.3]
            },
            "DIGL": {
                "ATE_pre": [-6.28e-1] * 4,
                "ATE_mean": [-3.12e-1] * 4,
                "ATE_std": [-1.57e-1] * 4,
                "ATE_max": [-5.49e-1] * 4,
                "Gain": [0.2, -6.6, -0.4, -4.9]
            },
            "SDRF": {
                "ATE_pre": [-4.00e-2, -2.36e-2, -1.08e-2, -7.06e-2],
                "ATE_mean": [-4.90e-2, -2.89e-2, -1.70e-2, -8.05e-2],
                "ATE_std": [2.32e-2, 2.55e-2, 1.99e-2, 6.14e-3], # Not significant
                "ATE_max": [1.62e-1, 1.41e-1, 1.08e-1, 1.26e-1],
                "Gain": [-0.3, -0.4, 3.6, 1.3]
            },
            "BORF": {
                "ATE_pre": [-5.14e-2, -5.14e-2],
                "ATE_mean": [-7.61e-2, -7.61e-2],
                "ATE_std": [-3.71e-1, -3.71e-1],
                "ATE_max": [-1.41e-1, -2.82e-1],
                "Gain": [0.4, 1.2]
            },
            "GTR": {
                "ATE_pre": [-1.98e-2, -3.29e-2, -1.98e-2, -2.13e-2],
                "ATE_mean": [-1.98e-2, -8.95e-3, -1.98e-2, -1.21e-2],
                "ATE_std": [-3.41e-2, -2.66e-2, -3.41e-2, -1.42e-2],
                "ATE_max": [-5.32e-2, -4.19e-2, -5.32e-2, -8.21e-3],
                "Gain": [0.2, 1.1, 15.0, 2.5]
            }
        },
        "COLLAB": {
            "FoSR": {
                "ATE_pre": [9.64e-3, 8.84e-3, 7.50e-3, 9.64e-3],
                "ATE_mean": [-1.00e-2, -2.61e-2, -3.30e-3, -1.00e-2],
                "ATE_std": [-5.83e-3, -1.93e-2, -9.75e-4, -5.83e-3],
                "ATE_max": [-1.57e-2, -1.13e-2, -1.77e-2, -1.57e-2],
                "Gain": [0.1, 0.2, 37.0, 1.3]
            },
            "DIGL": {
                "ATE_pre": [-5.57e-1, -5.57e-1, -5.31e-1, -5.31e-1],
                "ATE_mean": [-2.56e-1, -2.56e-1, -2.55e-1, -2.55e-1],
                "ATE_std": [-1.93e-1, -1.93e-1, -1.91e-1, -1.91e-1],
                "ATE_max": [-9.10e-1, -9.10e-1, -8.88e-1, -8.88e-1],
                "Gain": [-18.2, -19.4, -16.7, -18.5]
            },
            "SDRF": {
                "ATE_pre": [5.13e-3, 3.95e-3, 6.48e-3, 6.48e-3],
                "ATE_mean": [-5.65e-3, -2.93e-2, -1.63e-2, -1.63e-2],
                "ATE_std": [-1.87e-3, -1.01e-2, -5.25e-3, -5.25e-3],
                "ATE_max": [-1.52e-2, -5.44e-2, -3.91e-2, -3.91e-2],
                "Gain": [-0.3, 0.0, 34.6, 0.9]
            },
            "GTR": {
                "ATE_pre": [1.24e-2, 1.63e-2, 1.63e-2, 1.55e-2],
                "ATE_mean": [-9.70e-4, 1.83e-3, 1.83e-3, 9.19e-4],
                "ATE_std": [-5.51e-4, -9.33e-4, -9.33e-4, -1.09e-3],
                "ATE_max": [4.09e-4, 2.05e-2, 2.05e-2, 2.16e-2],
                "Gain": [-0.7, 0.3, -0.09, 1.9]
            }
        },
        "REDDIT-BINARY": {
            "FoSR": {
                "ATE_pre": [4.54e-3, 5.33e-3, 4.54e-3, 7.11e-3], #NS
                "ATE_mean": [1.87e-3, 3.91e-3, 1.87e-3, 1.37e-2],
                "ATE_std": [1.89e-3, 3.07e-3, 1.89e-3, 8.51e-3],
                "ATE_max": [1.36e-2, 2.10e-2, 1.36e-2, 5.46e-2],
                "Gain": [2.1, 0.5, 26.7, 1.7]
            },
            "DIGL": {
                "ATE_pre": [-8.16e-2, -1.40e-1, -1.60e-2, -1.40e-1],
                "ATE_mean": [4.11e-2, 6.60e-2, 5.68e-2, 6.60e-2],
                "ATE_std": [5.31e-2, 7.23e-2, 5.21e-2, 7.23e-2],
                "ATE_max": [5.08e-1, 6.96e-1, 6.20e-1, 6.96e-1],
                "Gain": [-18.3, -10.7, -10.7, -13.5]
            },
            "SDRF": {
                "ATE_pre": [-1.67e-3, -1.67e-3, -7.40e-3, -1.67e-3], #NS
                "ATE_mean": [-5.02e-4, -5.02e-4, -2.74e-3, -5.02e-4], #NS
                "ATE_std": [-2.62e-4, -2.62e-4, -1.80e-3, -2.62e-4], #NS
                "ATE_max": [1.04e-3, 1.04e-3, -4.28e-3, 1.04e-3], #NS
                "Gain": [0.3, -0.3, -8.8, -1.1]
            },
            "GTR": {
                "ATE_pre": [1.89e-2, 1.89e-2, 1.99e-2, 1.89e-2],
                "ATE_mean": [1.23e-2, 1.23e-2, 1.83e-2, 1.23e-2],
                "ATE_std": [8.52e-3, 8.52e-3, 1.13e-2, 8.52e-3],
                "ATE_max": [3.96e-2, 3.96e-2, 6.00e-2, 3.96e-2],
                "Gain": [0.7, 0.2, 30.3, 2.44]
            }
        }
    }
    node_data = {
        "CORA": {
            "FoSR": {
                "ITE_pre": [1.64e-2, -1.52e-2],
                "ITE_mean": [1.59e-2, -3.59e-2],
                "ITE_std": [1.53e-2, -2.44e-2],
                "ITE_max": [7.20e-2, -1.94e-1],
                "Gain": [-0.8, -0.9],
            },
            "DIGL": {
                "ITE_pre": [6.45e-1],
                "ITE_mean": [1.98e-1],
                "ITE_std": [1.59e-1],
                "ITE_max": [2.17e0],
                "Gain": [1.3],
            },
            "SDRF": {
                "ITE_pre": [5.45e-6, 5.45e-6],
                "ITE_mean": [5.33e-5, 5.33e-5],
                "ITE_std": [3.18e-6, 3.18e-6],
                "ITE_max": [1.92e-5, 1.92e-5],
                "Gain": [-0.3, 1.0],
            },
            "BORF": {
                "ITE_pre": [-6.70e-5, -6.90e-5],
                "ITE_mean": [-7.28e-4, -8.75e-4],
                "ITE_std": [1.50e-4, 1.88e-4],
                "ITE_max": [1.19e-2, 1.22e-2],
                "Gain": [0.8, 3.8],
            }
        },
        "CITESEER": {
            "FoSR": {
                "ITE_pre": [-6.01e-5, -1.02e-4],
                "ITE_mean": [5.98e-4, 1.29e-3],
                "ITE_std": [2.35e-3, 4.11e-3],
                "ITE_max": [1.72e-2, 2.35e-2],
                "Gain": [0.0, 2.4],
            },
            "DIGL": {
                "ITE_pre": [1.80e-1],
                "ITE_mean": [1.39e-1],
                "ITE_std": [1.53e-2],
                "ITE_max": [1.28e0],
                "Gain": [1.0],
            },
            "SDRF": {
                "ITE_pre": [None, None],
                "ITE_mean": [None, None],
                "ITE_std": [None, None],
                "ITE_max": [None, None],
                "Gain": [None, None],
            },
            "BORF": {
                "ITE_pre": [2.98e-06, 9.93e-07],  # Not significant
                "ITE_mean": [-4.92e-6, -1.61e-6], # Not significant
                "ITE_std": [-1.37e-5, -4.61e-6],
                "ITE_max": [0.0, 0.0],
                "Gain": [1.5, 3.8],
            }
        },
        "TEXAS": {
            "FoSR": {
                "ITE_pre": [3.99e-2, 1.11e-1],
                "ITE_mean": [1.46e-2, 1.64e-2],
                "ITE_std": [1.09e-2, 1.23e-2],
                "ITE_max": [1.27e-1, 5.51e-2],
                "Gain": [1.8, -6.5],
            },
            "DIGL": {
                "ITE_pre": [3.25e-2],
                "ITE_mean": [2.41e-2],
                "ITE_std": [3.80e-2],
                "ITE_max": [3.01e-1],
                "Gain": [-0.8],
            },
            "SDRF": {
                "ITE_pre": [1.13e-1, 1.13e-1],
                "ITE_mean": [2.17e-3, 2.17e-3],
                "ITE_std": [5.31e-3, 5.31e-3],
                "ITE_max": [8.65e-2, 8.65e-2],
                "Gain": [-0.3, -3.2],
            },
            "BORF": {
                "ITE_pre": [8.52e-2, -1.06e-2],
                "ITE_mean": [1.38e-3, 1.45e-4], #NS
                "ITE_std": [1.92e-3, -2.34e-4],
                "ITE_max": [5.97e-2, -6.65e-3],
                "Gain": [5.2, 9.6],
            }
        },
        "CORNELL": {
            "FoSR": {
                "ITE_pre": [8.94e-2, 2.09e-4], #NS
                "ITE_mean": [2.59e-2, 1.97e-2],
                "ITE_std": [2.56e-2, 1.19e-2],
                "ITE_max": [2.01e-1, 6.85e-2],
                "Gain": [-1.3, -0.9],
            },
            "DIGL": {
                "ITE_pre": [-3.03e-2],
                "ITE_mean": [1.97e-1],
                "ITE_std": [2.16e-1],
                "ITE_max": [1.70e0],
                "Gain": [5.0],
            },
            "SDRF": {
                "ITE_pre": [None, None],
                "ITE_mean": [None, None],
                "ITE_std": [None, None],
                "ITE_max": [None, None],
                "Gain": [None, None],
            },
            "BORF": {
                "ITE_pre": [-5.77e-2, -6.16e-2],
                "ITE_mean": [5.68e-2, 7.88e-2],
                "ITE_std": [9.72e-1, 4.16e-1],
                "ITE_max": [2.22e0, 2.33e0],
                "Gain": [9.3, 12.1],
            }
        },
        "WISCONSIN": {
            "FoSR": {
                "ITE_pre": [8.01e-2, 7.99e-3],
                "ITE_mean": [2.60e-2, -1.40e-4],
                "ITE_std": [2.33e-2, -2.67e-4],
                "ITE_max": [2.59e-1, 2.31e-2],
                "Gain": [3.7, 0.0],
            },
            "DIGL": {
                "ITE_pre": [-4.04e-1],
                "ITE_mean": [-7.43e-3],
                "ITE_std": [-7.44e-3],
                "ITE_max": [-1.00e-1],
                "Gain": [-2.4],
            },
            "SDRF": {
                "ITE_pre": [2.93e-2, -1.34e-1],
                "ITE_mean": [8.14e-4, -3.24e-3],
                "ITE_std": [1.08e-3, -2.11e-3],
                "ITE_max": [3.49e-2, -2.13e-2],
                "Gain": [0.7, 0.2],
            },
            "BORF": {
                "ITE_pre": [-2.40e-2, -3.89e-2],
                "ITE_mean": [4.00e-4, 2.51e-4],
                "ITE_std": [-5.71e-4, -1.33e-3],
                "ITE_max": [-7.89e-3, -1.20e-2],
                "Gain": [5.7, 6.4],
            }
        },
        "CHAMELEON": {
            "FoSR": {
                "ITE_pre": [1.89e-1, 1.75e-1],
                "ITE_mean": [5.88e-3, 1.12e-2],
                "ITE_std": [2.81e-2, 2.86e-2],
                "ITE_max": [1.23e-1, 1.01e-1],
                "Gain": [0.1, -1.8],
            },
            "DIGL": {
                "ITE_pre": [4.60e-1],
                "ITE_mean": [5.19e-2],
                "ITE_std": [1.11e-1],
                "ITE_max": [1.14e0],
                "Gain": [-0.7],
            },
            "SDRF": {
                "ITE_pre": [-2.78e-4, -2.61e-4],
                "ITE_mean": [-2.64e-4, -1.27e-4],
                "ITE_std": [-3.03e-4, -7.68e-5],
                "ITE_max": [5.12e-4, 6.45e-4],
                "Gain": [0.2, 0.3],
            },
            "BORF": {
                "ITE_pre": [-8.20e-3, -8.18e-3],
                "ITE_mean": [-3.38e-2, -3.30e-2],
                "ITE_std": [-2.10e-2, -2.06e-2],
                "ITE_max": [-9.96e-2, -9.75e-2],
                "Gain": [2.3, 7.2],
            }
        }
    }

    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'correlation_results')
    # os.makedirs(path, exist_ok=True)
    # setup_logging(f'{path}/graph_correlation_results.log')
    # stats(data)
    # plot_scatter(data)
    # correlation_based_on_rewiring(data)
    visualize_correlations()
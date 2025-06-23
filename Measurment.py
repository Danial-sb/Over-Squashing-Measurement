import matplotlib
import torch
from scipy.stats import linregress
from torch_geometric.utils import add_self_loops

matplotlib.use('Agg')  # Use the Agg backend for plotting

def decay_rate(
    adj: torch.Tensor,
    diameter: int,
    device: str = 'cpu'):
    """
    Compute the decay rates for all pairs of nodes in an adjacency matrix by fitting a linear model
    on the log of the matrix powers. Returns the decay rates along with the mean corrected p-value,
    mean R^2 value, and mean uncorrected p-value.

    Parameters:
            :param adj: adjacency matrix
            :param diameter: diameter of the graph
            :param device: device (gpu/cpu)
    Returns:
        Tuple[List[float], float, float, float]:
            - List of decay rates.
    """
    adj = adj.to(device)

    # ----------- pre-compute matrix powers and logs -------------------------
    k_range = torch.arange(diameter, 2 * diameter, device=device, dtype=torch.float32)
    A_powers = torch.stack([torch.matrix_power(adj, int(k)) for k in k_range], dim=2)
    log_A_uv = torch.log(A_powers)
    sum_A_kv_l = torch.sum(A_powers, dim=0)
    log_A_kv_l = torch.log(sum_A_kv_l)

    n = adj.size(0)
    slopes = []

    v_idx, u_idx = torch.nonzero(
        torch.ones(n, n, dtype=torch.bool, device=device),
        as_tuple=True)

    for v, u in zip(v_idx.tolist(), u_idx.tolist()):
            log_A_vu_k = log_A_uv[v, u, :] - log_A_kv_l[v, :]

            slope, _, r_value, p_value, _ = linregress(k_range.cpu().numpy(), log_A_vu_k.cpu().numpy())
            slopes.append(slope)

    slopes_matrix = torch.full((n, n), float("nan"), device=device)
    slopes_matrix[v_idx, u_idx] = torch.tensor(slopes, device=device, dtype=torch.float32)

    decay_rates = slopes_matrix[~torch.isnan(slopes_matrix)].tolist()

    decay_rates_with_nan = slopes_matrix.flatten().tolist()

    return decay_rates, decay_rates_with_nan, n
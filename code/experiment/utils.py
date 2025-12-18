from collections.abc import Callable
from typing import Any
from tqdm import tqdm
import numpy as np
import networkx as nx

import torch
import torch.nn.functional as F


def get_device() -> str:
    """
    Select the compute device, preferring CUDA, and clear its cache.

    Returns
    -------
    device:
        "cuda" if a GPU is available.

    Raises
    ------
    RuntimeError
        If no CUDA device is available.

    Notes
    -----
    - This function currently *requires* a GPU. If you ever want a CPU
    fallback, this is the place to change that behavior.
    - Before returning, it calls `torch.cuda.empty_cache()` to reduce
    fragmentation across runs.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return "cuda"
    raise RuntimeError("No GPU available. Please run on a machine with CUDA support.")


def graph_to_gpu_tensors(
    graph: nx.DiGraph,
    weight: str = "weight",
    device: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, list[Any], int]:
    """
    Convert a NetworkX DiGraph into (edge_index, edge_weight) tensors.

    This path uses NumPy as an intermediate and then moves tensors to
    the chosen device.

    Parameters
    ----------
    graph:
        Directed NetworkX graph.
    weight:
        Name of the edge attribute holding weights. If missing on an edge,
        defaults to 1.0.
    device:
        Target device ("cuda", "cpu", or "auto"). If "auto", uses `get_device()`.

    Returns
    -------
    edge_index:
        Long tensor of shape (2, E) with [0, :] = sources, [1, :] = targets.
    edge_weight:
        Float tensor of shape (E,) with edge weights.
    node_list:
        List of node labels in the order they map to tensor indices.
    node_count:
        Number of nodes in the graph.
    """
    if device == "auto":
        device = get_device()

    node_list = list(graph.nodes())
    node_count = len(node_list)
    node_to_index = {node: idx for idx, node in enumerate(node_list)}

    edges = list(graph.edges(data=True))
    source_indices = np.empty(len(edges), dtype=np.int64)
    target_indices = np.empty(len(edges), dtype=np.int64)
    weight_values = np.empty(len(edges), dtype=np.float32)

    for i, (source, target, edge_data) in enumerate(edges):
        source_indices[i] = node_to_index[source]
        target_indices[i] = node_to_index[target]
        weight_values[i] = float(edge_data.get(weight, 1.0))

    edge_index = torch.from_numpy(
        np.stack([source_indices, target_indices])
    ).to(device)
    edge_weight = torch.from_numpy(weight_values).to(device)

    return edge_index, edge_weight, node_list, node_count


def graph_to_adjacency_tensors(
    graph: nx.Graph,
    weight: str = "weight",
    device: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, list[Any], int]:
    """
    Convert a NetworkX graph to sparse-like adjacency tensors on a device.

    This is similar to `graph_to_gpu_tensors`, but allocates the tensors
    directly on the target device instead of going through NumPy first.

    Parameters
    ----------
    graph:
        NetworkX graph (directed or undirected).
    weight:
        Name of the edge attribute holding weights (default "weight").
    device:
        Target device ("cuda", "cpu", or "auto"). If "auto", uses `get_device()`.

    Returns
    -------
    edge_index:
        Long tensor of shape (2, E) with source and target indices.
    edge_weight:
        Float tensor of shape (E,) with edge weights.
    node_list:
        List of node labels in index order.
    node_count:
        Number of nodes in the graph.
    """
    if device == "auto":
        device = get_device()

    node_list = list(graph.nodes())
    node_count = len(node_list)
    node_to_index = {node: idx for idx, node in enumerate(node_list)}

    edges = list(graph.edges(data=True))
    edge_index = torch.empty((2, len(edges)), dtype=torch.long, device=device)
    edge_weight = torch.empty(len(edges), dtype=torch.float32, device=device)

    for i, (source, target, edge_data) in enumerate(edges):
        edge_index[0, i] = node_to_index[source]  # row = source
        edge_index[1, i] = node_to_index[target]  # col = target
        edge_weight[i] = float(edge_data.get(weight, 1.0))

    return edge_index, edge_weight, node_list, node_count


def sparse_matmul(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    vectors: torch.Tensor,
    node_count: int,
) -> torch.Tensor:
    """
    Sparse matrix-vector multiply using edge list + scatter_add.

    Interprets `edge_index` and `edge_weight` as a sparse adjacency A and
    computes:

        y = A @ x (or batched y = A @ Xᵀ)

    Parameters
    ----------
    edge_index:
        Long tensor of shape (2, E) with [0] = source, [1] = target indices.
    edge_weight:
        Float tensor of shape (E,) with edge weights.
    vectors:
        Either:
        - 1D tensor of shape (N,), or
        - 2D tensor of shape (B, N) (batch of vectors).
    node_count:
        Number of nodes N.

    Returns
    -------
    result:
        Tensor of shape (N,) if input was 1D, else (B, N).
    """
    source_nodes, target_nodes = edge_index
    single_vector = False

    # Normalize input to batched shape (B, N)
    if vectors.ndim == 1:
        vectors = vectors.unsqueeze(0)
        single_vector = True
    batch_size = vectors.shape[0]

    # Expand indices for batched operation
    source_expanded = source_nodes.unsqueeze(0).expand(batch_size, -1)
    target_expanded = target_nodes.unsqueeze(0).expand(batch_size, -1)
    target_expanded = target_expanded.to(dtype=torch.long, device=vectors.device)

    # Gather x[target] (or X[:, target]) for each batch
    gathered = torch.gather(vectors, 1, target_expanded)
    weighted = gathered * edge_weight.unsqueeze(0)
    weighted = weighted.to(dtype=vectors.dtype, device=vectors.device)

    # Scatter-add contributions onto sources
    result = torch.zeros(
        batch_size, node_count, device=vectors.device, dtype=vectors.dtype
    )
    result.scatter_add_(
        1,
        source_expanded.to(dtype=torch.long, device=vectors.device),
        weighted,
    )

    return result.squeeze(0) if single_vector else result


def normalize_rows(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    node_count: int,
) -> torch.Tensor:
    """
    Row-normalize edge weights to make adjacency stochastic.

    For each source node i, this computes:

        w_ij ← w_ij / sum_j w_ij

    so that the outgoing weights from each node sum to 1. This is used to
    build random-walk style transition matrices.

    Parameters
    ----------
    edge_index:
        Long tensor of shape (2, E): [0] = sources, [1] = targets.
    edge_weight:
        Float tensor of shape (E,) with edge weights.
    node_count:
        Number of nodes.

    Returns
    -------
    normalized_edge_weight:
        Float tensor of shape (E,), where weights are row-normalized.
    """
    source_nodes, _ = edge_index
    row_sums = torch.zeros(node_count, device=edge_weight.device).scatter_add_(
        0, source_nodes, edge_weight
    )
    row_sums = torch.clamp(row_sums, min=1e-9)
    return edge_weight / row_sums[source_nodes]


def batched_power_iteration(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    node_count: int,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
    damping_factor: float = 0.0,
    batch_size: int = 1,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Batched power iteration to approximate a dominant eigenvector.

    Runs multiple random initializations in parallel and returns the
    “best” converged vector (based on residuals).

    Parameters
    ----------
    edge_index:
        Long tensor of shape (2, E): [0] = sources, [1] = targets.
    edge_weight:
        Float tensor of shape (E,) with (optionally normalized) weights.
    node_count:
        Number of nodes N (dimension of the vector).
    max_iterations:
        Maximum number of power iterations before stopping.
    tolerance:
        Convergence threshold on L1 differences and relative change.
    damping_factor:
        If > 0, applies a PageRank-style damping step:
            x ← (1 - d) A x + d * 1/N
    batch_size:
        Number of independent initial vectors to run in parallel.
    verbose:
        If True, shows progress and convergence logs.

    Returns
    -------
    vector:
        1D tensor of shape (N,) representing the selected eigenvector
        approximation (L1-normalized).
    """
    # Random positive initial vectors, L1-normalized per batch
    vectors = torch.randn(
        batch_size, node_count, device=edge_weight.device, dtype=torch.float32
    )
    vectors = F.normalize(torch.abs(vectors), p=1, dim=1)

    has_converged = False
    has_numerical_issues = False
    next_vectors = vectors.clone()

    iteration_range = range(max_iterations)
    if verbose:
        iteration_range = tqdm(iteration_range, desc="Power iteration", leave=False)

    previous_diff = torch.tensor(float("inf"), device=edge_weight.device)
    stagnation_count = 0

    for iteration in iteration_range:
        next_vectors = sparse_matmul(edge_index, edge_weight, vectors, node_count)

        # PageRank-style damping
        if damping_factor > 0:
            next_vectors = (
                (1 - damping_factor) * next_vectors
                + damping_factor * (torch.ones_like(next_vectors) / node_count)
            )

        # Keep values positive and L1-normalize
        next_vectors = torch.clamp(next_vectors, min=1e-9)
        next_vectors = F.normalize(next_vectors, p=1, dim=1)

        # Convergence metrics
        convergence_diff = torch.max(torch.norm(next_vectors - vectors, p=1, dim=1))
        relative_diff = convergence_diff / torch.max(
            torch.norm(vectors, p=1, dim=1)
        )

        stagnation_condition = torch.abs(convergence_diff - previous_diff) < tolerance * 0.1
        stagnation_count = stagnation_count + 1 if stagnation_condition else 0
        previous_diff = convergence_diff

        has_converged = (
            convergence_diff < tolerance
            or relative_diff < tolerance
            or stagnation_count >= 3
        )

        # **Important**: make sure we keep the most recent iterate
        vectors = next_vectors

        if has_converged:
            if verbose:
                tqdm.write(
                    f"Converged after {iteration + 1} iterations "
                    f"(diff: {convergence_diff:.2e})"
                )
            break

        # Check for numerical instability
        has_numerical_issues = torch.any(torch.isnan(vectors)) or torch.any(
            torch.isinf(vectors)
        )
        if has_numerical_issues:
            if verbose:
                tqdm.write("Numerical instability detected, returning best result")
            break

    if not has_converged and not has_numerical_issues and verbose:
        tqdm.write(f"Reached maximum iterations ({max_iterations})")

    # Select the vector with the smallest residual between the last two iterates
    residuals = torch.norm(vectors - next_vectors, p=1, dim=1)
    best_index = torch.argmin(residuals)
    return vectors[best_index]


def memory_optimized_operation(operation: Callable) -> Callable:
    """
    Decorator for GPU functions: clear CUDA cache before & after.

    Intended for long-running GPU-heavy operations to reduce the risk
    of fragmentation and out-of-memory errors when running many experiments.

    Usage
    -----
    @memory_optimized_operation
    def some_gpu_heavy_fn(...):
        ...

    Notes
    -----
    - If no CUDA is available, this decorator becomes a no-op.
    - It does *not* change function semantics, only memory behavior.
    """
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        result = operation(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result
    return wrapper
